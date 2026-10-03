"""Does a reused Context keep the first run's max_iterations?

Bears on run-llama/llama_index#23297 and #23051: `_init_context` wraps the whole
block in `if not await ctx.store.get("max_iterations", default=None)`, so a value
already in the store makes every line inside it unreachable, validation included.

No network and no model: MockFunctionCallingLLM returns a scripted list.

    PYTHONPATH=llama-index-core .venv/bin/python probe_context_reuse_budget.py
"""

import asyncio
from typing import List

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.workflow import Context
from llama_index.core.workflow.errors import WorkflowRuntimeError

TOOL_CALL = ChatMessage(
    role=MessageRole.ASSISTANT,
    content="calling tool",
    additional_kwargs={
        "tool_calls": [
            ToolSelection(tool_id="one", tool_name="side_effect", tool_kwargs={})
        ]
    },
)
FINAL = ChatMessage(role=MessageRole.ASSISTANT, content="the answer is 42")


def build(counts: dict) -> FunctionAgent:
    responses: List[ChatMessage] = [TOOL_CALL, TOOL_CALL, TOOL_CALL, TOOL_CALL, FINAL]
    index = 0

    def generator(messages: List[ChatMessage], **kwargs) -> ChatMessage:
        nonlocal index
        counts["llm_calls"] += 1
        response = responses[min(index, len(responses) - 1)]
        index += 1
        return response

    def side_effect() -> str:
        counts["tool_runs"] += 1
        return "done"

    return FunctionAgent(
        name="agent",
        description="test",
        tools=[side_effect],
        llm=MockFunctionCallingLLM(response_generator=generator),
    )


async def run(label: str, agent, counts: dict, **kwargs) -> None:
    before = counts["llm_calls"]
    try:
        await agent.run(user_msg="test", **kwargs)
        outcome = "ok"
    except WorkflowRuntimeError as exc:
        outcome = f"raised {str(exc).split('!')[0]}"
    print(f"{label:46} llm_calls_this_run={counts['llm_calls'] - before}  {outcome}")


async def main() -> None:
    counts = {"llm_calls": 0, "tool_runs": 0}
    agent = build(counts)
    ctx = Context(agent)
    await run(
        "run 1 on a fresh Context, max_iterations=2",
        agent,
        counts,
        ctx=ctx,
        max_iterations=2,
    )
    await run(
        "run 2 on the SAME Context, max_iterations=5",
        agent,
        counts,
        ctx=ctx,
        max_iterations=5,
    )

    counts2 = {"llm_calls": 0, "tool_runs": 0}
    agent2 = build(counts2)
    await run(
        "control: fresh Context each run, max_iterations=5",
        agent2,
        counts2,
        max_iterations=5,
    )


if __name__ == "__main__":
    asyncio.run(main())
