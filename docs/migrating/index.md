# Migrating

Two guides for the two most common starting points. Each maps
concept-for-concept, shows a side-by-side rewrite, and calls out what
you lose (there is always something) and what you gain.

<div class="grid cards" markdown>

-   __From LangChain__

    ---

    You already have chains, tools, and an `AgentExecutor`. Here's the
    concept-to-concept mapping, a full before/after, and an honest read
    on where LangChain still wins (large integration catalog).

    [:octicons-arrow-right-24: Read the LangChain guide](from-langchain.md)

-   __From vanilla asyncio + provider SDK__

    ---

    You wrote your own `messages.create` loop with a `for` over tool
    calls. Here's the same loop as a `ReActCognition` and every
    resilience/cancel/budget primitive you were about to write
    yourself.

    [:octicons-arrow-right-24: Read the asyncio guide](from-vanilla-asyncio.md)

</div>

If your starting point is LlamaIndex, this round doesn't have a
dedicated guide — the shapes differ enough (LlamaIndex is
retrieval-first, agentkit is composition-first) that a direct
mapping would mislead more than help. Read the [Cheatsheet](../cheatsheet.md)
and [Why agentkit](../why.md) instead, and open an issue if a proper
guide would help.
