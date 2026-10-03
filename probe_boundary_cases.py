"""Three boundary cases the max_iterations gate does not cover, measured.

Case A: early_stopping_method='generate' at the boundary.
Case B: a return_direct tool call at the boundary.
Case C: max_iterations=0.

No network and no model: MockFunctionCallingLLM returns a scripted list, and the
counter wraps the generator so each call is recorded.

    PYTHONPATH=llama-index-core .venv/bin/python probe_boundary_cases.py
"""

import asyncio
from typing import List

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow.errors import WorkflowRuntimeError

FINAL = ChatMessage(role=MessageRole.ASSISTANT, content="the answer is 42")


def tool_call(name: str) -> ChatMessage:
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        content="calling tool",
        additional_kwargs={
            "tool_calls": [ToolSelection(tool_id="one", tool_name=name, tool_kwargs={})]
        },
    )


class Counter:
    def __init__(self) -> None:
        self.llm_calls = 0
        self.tool_runs = 0


def build(
    responses: List[ChatMessage], counter: Counter, *, direct: bool
) -> FunctionAgent:
    index = 0

    def generator(messages: List[ChatMessage], **kwargs) -> ChatMessage:
        nonlocal index
        counter.llm_calls += 1
        response = responses[min(index, len(responses) - 1)]
        index += 1
        return response

    def side_effect() -> str:
        counter.tool_runs += 1
        return "done"

    tool = FunctionTool.from_defaults(
        side_effect, name="side_effect", return_direct=direct
    )
    return FunctionAgent(
        name="agent",
        description="test",
        tools=[tool],
        llm=MockFunctionCallingLLM(response_generator=generator),
    )


async def run(label: str, agent: FunctionAgent, counter: Counter, **kwargs) -> None:
    try:
        response = await agent.run(user_msg="test", **kwargs)
        outcome = f"ok response={str(response.response)[-12:]!r}"
    except WorkflowRuntimeError as exc:
        outcome = f"raised {str(exc).split('!')[0]}"
    print(
        f"{label:52} llm_calls={counter.llm_calls} tool_runs={counter.tool_runs}  {outcome}"
    )


async def main() -> None:
    c = Counter()
    await run(
        "A max_iterations=1, tool call, generate",
        build([tool_call("side_effect"), FINAL], c, direct=False),
        c,
        max_iterations=1,
        early_stopping_method="generate",
    )

    c = Counter()
    await run(
        "B max_iterations=1, return_direct tool call",
        build([tool_call("side_effect"), FINAL], c, direct=True),
        c,
        max_iterations=1,
    )

    c = Counter()
    await run(
        "C max_iterations=0, tool round then answer",
        build([tool_call("side_effect"), FINAL], c, direct=False),
        c,
        max_iterations=0,
    )


if __name__ == "__main__":
    asyncio.run(main())
