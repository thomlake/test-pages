# Meta-Chats

> *Disclaimer: This document is highly speculative and based entirely on anecdotal observations and extremely limited qualitative evaluation. Take it with a grain of salt.*

## Introduction

I recently ported some prompts developed for `text-davinci-003` to `gpt-3.5-turbo`. Although both models are based on [GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5), there are several key differences that had to be taken into consideration.

1. **Cost:** `gpt-3.5-turbo` is 1/10th the cost of `text-davinci-003` ($0.02 vs $0.002 / 1K tokens). This is the primary motivator for the switch.
2. **Interface:** `text-davinci-003` uses the [completions](https://platform.openai.com/docs/guides/completion) endpoint, which works differently than the recently introduced [chat](https://platform.openai.com/docs/guides/chat) endpoint used to access `gpt-3.5-turbo`.
3. **Fine-tuning:** The OpenAI [docs](https://platform.openai.com/docs/models/gpt-3-5) state `gpt-3.5-turbo` "has been optimized for chat". Presumably, `text-davinci-003` uses instructions fine-tuning (IFT), while `gpt-3.5-turbo` use IFT and reinforcement learning from human feedback (RLHF). Unfortunately, we don't know the exact details because OpenAI 🤷, but this seems to be the general consensus.

It was straightforward to port single-shot tasks like classification or QA. Interestingly, porting chat-like applications took more work. With a bit of exploration, I was able to find a simple prompt modification that seems to work fairly well.

## Chat

An example prompt for chat is given below.

```python
# Chat Prompt
"""You are a helpful bot.

User: {user[0]}
Assistant: {assistant[0]}
...
User: {user[-1]}
Assistant:"""
```

The chat endpoint uses a message-based input format mimics this pattern.

```python
messages = [
	{"role": "system", "content": "You are a helpful bot."},
	{"role": "user", "content": user[0]},
	{"role": "assistant", "content": assistant[0]},
	...,
	{"role": "user", "content": user[-1]},
]
```

Behind the scenes, the messages are converted into a chat prompt like the one above. In reality, this prompt uses a custom markup language called ChatML (more details are available [here](https://github.com/openai/openai-python/blob/main/chatml.md) or [here](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/chatgpt)), but the example above is close enough for this discussion. What's important to emphasize is that in the constructed prompt, messages are attributed to a single entity which must be one of `"system"`, `"user"`, and `"assistant"`, and the `gpt-3.5-turbo` model has been specialized for this sort of two-party turn-taking context.

## Augmented Chat

In practical applications, it is often necessary to augment prompts with additional information for the model to condition on during generation (like [RAG](https://arxiv.org/abs/2005.11401) or [ReAct](https://arxiv.org/abs/2210.03629)). For example, the original chat prompt could be augmented to include readings from an external sensor.

```python
# Augmented Chat Prompt
"""You are a helpful bot.

User: {user[0]}
Sensor: {sensor[0]}
Assistant: {assistant[0]}
...
User: {user[-1]}
Sensor: {sensor[-1]}
Assistant:"""
```

The most obvious way to encode this additional information in the message based format is to use `"role": "sensor"`. This isn't an option in practice because the OpenAI chat API only accepts the roles `"system"`, `"user"`, and `"assistant"`. However, even without this limitation, it is unclear the extent to which changing from a two-party to a three-party context featuring previously unseen roles would degrade performance since that is not the context the model was fine-tuned for. Several alternative approaches are discussed below.

### Strategy 1: System Messages

An alternative strategy would be to use system messages and the `"name"` field mentioned [here](https://github.com/openai/openai-python/blob/main/chatml.md#few-shot-prompting).

```python
messages = [
	{"role": "system", "content": "You are a helpful bot."},
	{"role": "user", "content": user[0]},
	{"role": "system", "name": "sensor", "content": sensor[0]},  # <-- "name": "sensor"
	{"role": "assistant", "content": assistant[0]},
	...,
	{"role": "user", "content": user[-1]},
	{"role": "system", "name": "sensor", "content": sensor[-1]},  # <-- "name": "sensor"
]
```

Qualitative experimentation suggests this makes the model more likely to ignore instructions in the first system message. Intuitively, this makes sense. If the model is fine-tuned in contexts where there is only a single system message containing instructions, it seems plausible that interleaving system messages would have a resetting effect.

### Strategy 2: Mixed-User Messages

A second alternative is to simply augment the user messages with additional information. Doing this would result in something like the following prompt for a user message.

```python
# Mixed-User Message
"""User: {user}
Sensor: {sensor}"""
```

```python
messages = [
	{"role": "system", "content": "You are a helpful bot."},
	{"role": "user", "content": mixed[0]},
	{"role": "assistant", "content": assistant[0]},
	...,
	{"role": "user", "content": mixed[-1]},
]
```

While this seems to work better than the system message approach, explicitly including input from multiple sources in a context where the model expects a single source appears to easily confuse the model. For example, the model would often respond with something like *"I'm not sure what you mean by Sensor..."*, and then go on to ask for clarification.

### Strategy 3: Meta-Chats

An approach that appears to be more effective and flexible is to lift the conversation context up a level. Specifically, rather than using a context where an assistant is conversing directly with a user, a context can be adopted where an assistant is conversing with a meta-user who is relaying details of a separate chat.

```python
# Meta-User Message
"""The user input "{user}"

The updated sensor reading is "{sensor}"

How should the bot reply?"""
```

```python
messages = [
	{"role": "system", "content": "You are a helpful bot."},
	{"role": "user", "content": meta[0]},
	{"role": "assistant", "content": assistant[0]},
	...,
	{"role": "user", "content": meta[-1]},
]
```

This is a subtle change compared to the mixed-message format, but initial experiments suggest it has a positive impact on quality. 


## Discussion

I can come up with several reasonable hypotheses for why porting augmented chat prompts was difficult, and why adopting a meta-chat context seems to alleviate this issue.

1. The difficulty is simply an artifact of the particular way messages get linearized by OpenAI. Having more control over this would eliminate the need for the meta-user.
2. The difficulty can be attributed to a mismatch between the two-party turn-taking chat fine-tuning context and the augmented multi-party chat context.

Hypothesis 2 seems more plausible given the issues encountered when using system messages and the mixed message format. If so, I think this could point to useful ways to think more generally about how to frame downstream task prompts so they match fine-tuning contexts. This would also imply that while techniques such as IFT and RLHF have undoubtedly made modern LLMs significantly more user-friendly, they have not eliminated the need for appropriate framing.

Of course, I also find Hypothesis 2 more interesting, so my plausibility assessment could just be my own bias showing. Furthermore, as stated in the *disclaimer*, all this is purely speculative and based entirely on limited anecdotal observations. Significantly more research and evaluation would be required to make stronger statements.
