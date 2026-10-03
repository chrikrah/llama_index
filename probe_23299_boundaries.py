"""Check #23299's validation on the paths its tests do not cover.

Cases: a negative budget, a reused Context carrying a valid budget, and a reused
Context handed a zero at run time. No network and no model.

    PYTHONPATH=llama-index-core .venv/bin/python probe_23299_boundaries.py
"""

import asyncio
from typing import List

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.workflow import Context
from llama_index.core.workflow.errors import WorkflowRuntimeError

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


async def case(label: str, **kwargs) -> None:
    counts = {"llm_calls": 0}
    agent = build(counts)
    ctx = kwargs.pop("ctx_first", None)
    if ctx is not None:
        ctx = Context(agent)
        await agent.run(user_msg="warm", ctx=ctx, max_iterations=2)
        before = counts["llm_calls"]
        kwargs["ctx"] = ctx
    else:
        before = 0
    try:
        await agent.run(user_msg="test", **kwargs)
        outcome = "ok, no error"
    except ValueError as exc:
        outcome = f"ValueError: {exc}"
    except WorkflowRuntimeError as exc:
        outcome = f"WorkflowRuntimeError: {str(exc).split('!')[0]}"
    print(f"{label:52} llm_calls_after={counts['llm_calls'] - before}  {outcome}")


async def main() -> None:
    await case("fresh Context, max_iterations=0", max_iterations=0)
    await case("fresh Context, max_iterations=-5", max_iterations=-5)
    await case("fresh Context, max_iterations=1", max_iterations=1)
    await case("reused Context warmed at 2, then 5", ctx_first=True, max_iterations=5)
    await case("reused Context warmed at 2, then 0", ctx_first=True, max_iterations=0)
    await case("reused Context warmed at 2, then -5", ctx_first=True, max_iterations=-5)


if __name__ == "__main__":
    asyncio.run(main())
