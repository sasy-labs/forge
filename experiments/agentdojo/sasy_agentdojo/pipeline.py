"""Assemble an AgentDojo pipeline whose tools run behind the SASY monitor.

AgentDojo's own `AgentPipeline.from_config` only knows the defenses that ship
with the benchmark, so a new one normally means forking the package — which is
what Progent did. It is not necessary: the pipeline is just an ordered list of
elements, and the benchmark functions accept any `BasePipelineElement`. Build
the list here, hand it to `benchmark_suite_with_injections`, and the upstream
package stays untouched.

The shape is the stock no-defense pipeline with one substitution:

    SystemMessage → InitQuery → llm → ToolsExecutionLoop([monitor, llm])

where `monitor` is `SasyToolsExecutor` in place of `ToolsExecutor`.
"""

from __future__ import annotations

from typing import Any, Callable

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, get_llm, load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop
from agentdojo.models import MODEL_PROVIDERS, ModelsEnum

from .graph import ConversationRecorder
from .monitor import PolicyDecision, ResponseMonitor, SasyToolsExecutor


def build_pipeline(
    suite: str,
    model: str,
    directory: frozenset[str] = frozenset(),
    *,
    model_id: str | None = None,
    system_message_name: str | None = None,
    fail_closed: bool = True,
    on_decision: Callable[[str, dict[str, Any], PolicyDecision], None] | None = None,
    name_suffix: str = "sasy",
    llm: BasePipelineElement | None = None,
    check_response: bool = False,
    repeat_user_prompt: bool = False,
) -> tuple[AgentPipeline, ConversationRecorder]:
    """Return the pipeline and the recorder it writes the graph through.

    The recorder is returned so the caller can reset it between tasks: each
    AgentDojo task is a fresh conversation and must not inherit the previous
    one's node ids.

    `llm` overrides the model element. AgentDojo resolves a provider from the
    model name, which only covers the names in its own `ModelsEnum`; passing an
    element built elsewhere is how a deployment it has never heard of — an
    Azure deployment, a self-hosted endpoint — gets used without editing the
    benchmark.
    """
    if llm is None:
        llm = get_llm(MODEL_PROVIDERS[ModelsEnum(model)], model, model_id, "tool")
    recorder = ConversationRecorder(agent_name=f"agentdojo-{suite}")
    monitor = SasyToolsExecutor(
        suite=suite,
        recorder=recorder,
        directory=directory,
        fail_closed=fail_closed,
        on_decision=on_decision,
    )
    elements: list[BasePipelineElement] = [
        SystemMessage(load_system_message(system_message_name)),
        InitQuery(),
        llm,
        # `repeat_user_prompt` puts an extra `InitQuery()` inside the loop, so
        # the user's request is restated after every batch of tool output. It is
        # AgentDojo's own defense and it composes with the monitor rather than
        # competing with it — the two act at different points, one on what the
        # model reads and one on what it is allowed to do.
        ToolsExecutionLoop(
            [monitor, InitQuery(), llm] if repeat_user_prompt else [monitor, llm]
        ),
    ]
    if check_response:
        # Last, so it sees the answer the user would actually have received.
        elements.append(
            ResponseMonitor(
                suite=suite,
                recorder=recorder,
                directory=directory,
                fail_closed=fail_closed,
                on_decision=on_decision,
            )
        )
    pipeline = AgentPipeline(elements)
    pipeline.name = f"{model}-{name_suffix}"
    return pipeline, recorder
