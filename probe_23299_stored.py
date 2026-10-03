"""What #23299's `else` branch does to a Context that already holds a budget.

A stored non-positive value is reachable from the released code: at 169e450aa a
fresh run with max_iterations=-5 stores -5, because -5 is truthy and the old
`or DEFAULT_MAX_ITERATIONS` passes it through. Seed the store directly here, so the
case is testable against either commit without depending on which one wrote it.

No network and no model.

    PYTHONPATH=llama-index-core .venv/bin/python probe_23299_stored.py
"""

import asyncio
from typing import List

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.workflow import Context

ANSWER = ChatMessage(role=MessageRole.ASSISTANT, content="the answer is 42")


def build(counts: dict) -> FunctionAgent:
    def generator(messages: List[ChatMessage], **kwargs) -> ChatMessage:
        counts["llm_calls"] += 1
        return ANSWER

    return FunctionAgent(
        name="agent",
        description="test",
        llm=MockFunctionCallingLLM(response_generator=generator),
    )


async def case(stored, asked) -> None:
    counts = {"llm_calls": 0}
    agent = build(counts)
    ctx = Context(agent)
    await ctx.store.set("max_iterations", stored)
    try:
        await agent.run(user_msg="test", ctx=ctx, max_iterations=asked)
        outcome = "ok, no error"
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {str(exc).split('!')[0][:52]}"
    print(
        f"store holds {str(stored):>3}, run asks {str(asked):>3}   llm_calls={counts['llm_calls']}  {outcome}"
    )


async def main() -> None:
    for stored, asked in ((-5, 5), (0, 5), (2, 0), (2, 5), (20, 1)):
        await case(stored, asked)


if __name__ == "__main__":
    asyncio.run(main())
