"""SASY policy enforcement for AgentDojo agents.

    from sasy_agentdojo.pipeline import build_pipeline

    pipeline, recorder = build_pipeline("banking", "gpt-4o-2024-05-13", directory)

See the README in the parent directory for what the pieces do and how to run
the benchmark.

The names below are imported on first use, so `sasy_agentdojo.presets` and
`sasy_agentdojo.policy_metadata` can be used without AgentDojo installed.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "ConversationRecorder": ".graph",
    "PolicyDecision": ".monitor",
    "SasyToolsExecutor": ".monitor",
    "build_llm": ".providers",
    "build_pipeline": ".pipeline",
    "build_policy_metadata": ".policy_metadata",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_EXPORTS[name], __name__)
    return getattr(module, "build" if name == "build_policy_metadata" else name)
