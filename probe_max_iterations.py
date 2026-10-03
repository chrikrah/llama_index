"""Count LLM calls and tool side effects at the max_iterations boundary.

No network and no model: MockFunctionCallingLLM returns a scripted list, and the
counter wraps the generator so each call is recorded.

Run with: python probe_max_iterations.py
"""

import asyncio
from typing import List

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.mock import MockFunctionCallingLLM
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


class Counter:
    def __init__(self) -> None:
        self.llm_calls = 0
        self.tool_runs = 0


def build(responses: List[ChatMessage], counter: Counter) -> FunctionAgent:
    index = 0

    def generator(messages: List[ChatMessage], **kwargs) -> ChatMessage:
        nonlocal index
        counter.llm_calls += 1
        msg = responses[min(index, len(responses) - 1)]
        index += 1
        return msg

    def side_effect() -> str:
        counter.tool_runs += 1
        return "done"

    return FunctionAgent(
        name="agent",
        description="test",
        tools=[side_effect],
        llm=MockFunctionCallingLLM(response_generator=generator),
    )


async def _report(label: str, runner, counter: Counter) -> None:
    try:
        response = await runner
        outcome = f"ok response={str(response.response)[-12:]!r}"
    except WorkflowRuntimeError as exc:
        outcome = f"raised {str(exc).split('!')[0]}"
    print(
        f"{label:44} llm_calls={counter.llm_calls} tool_runs={counter.tool_runs}  {outcome}"
    )


async def case(label: str, responses: List[ChatMessage], max_iterations: int) -> None:
    counter = Counter()
    agent = build(responses, counter)
    await _report(
        label, agent.run(user_msg="test", max_iterations=max_iterations), counter
    )


async def case_multi(
    label: str, responses: List[ChatMessage], max_iterations: int
) -> None:
    counter = Counter()
    agent = build(responses, counter)
    workflow = AgentWorkflow(agents=[agent], root_agent="agent")
    await _report(
        label, workflow.run(user_msg="test", max_iterations=max_iterations), counter
    )


async def main() -> None:
    print("single agent (BaseWorkflowAgent.parse_agent_output)")
    await case("max_iterations=1, first response final", [FINAL], 1)
    await case("max_iterations=1, first response tool call", [TOOL_CALL, FINAL], 1)
    await case("max_iterations=2, tool then final", [TOOL_CALL, FINAL], 2)
    print()
    print("AgentWorkflow (AgentWorkflow.parse_agent_output)")
    await case_multi("max_iterations=1, first response final", [FINAL], 1)
    await case_multi(
        "max_iterations=1, first response tool call", [TOOL_CALL, FINAL], 1
    )
    await case_multi("max_iterations=2, tool then final", [TOOL_CALL, FINAL], 2)


if __name__ == "__main__":
    asyncio.run(main())
