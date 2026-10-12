"""Model endpoints AgentDojo does not know about.

AgentDojo picks a provider by looking the model name up in its own
`ModelsEnum`, so an Azure deployment — whose name is chosen by whoever created
it and need not resemble a model id — has no route through `get_llm`. Since
`OpenAILLM` takes any client exposing `chat.completions.create`, and the
`AzureOpenAI` client does, one is built here and handed to `build_pipeline`.

The deployment name goes where the model name normally goes, which is how the
Azure API addresses a deployment.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import openai
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM


def load_env_file(path: Path) -> dict[str, str]:
    """Read a `KEY=value` file without disturbing the process environment."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        match = re.match(r'^\s*([A-Z0-9_]+)\s*=\s*"?([^"#]*)"?\s*$', line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def build_llm(
    provider: str,
    deployment: str,
    env_file: Path | None = None,
    *,
    temperature: float | None = 0.0,
) -> BasePipelineElement:
    """An `OpenAILLM` wired to `provider`.

    `azure` reads the endpoint from `AZURE_OPENAI_API_BASE`,
    `AZURE_OPENAI_ENDPOINT` or `AZURE_API_BASE` and the key from
    `AZURE_OPENAI_API_KEY` or `AZURE_API_KEY` (the last two of each are the
    names the tau2 cells use), from the environment, falling back to
    `env_file`; `deployment` names the Azure
    deployment. `openai` uses `OPENAI_API_KEY` and treats `deployment` as the
    model id.
    """
    env = {**load_env_file(env_file), **os.environ} if env_file else dict(os.environ)

    if provider == "azure":
        endpoint = (env.get("AZURE_OPENAI_API_BASE") or env.get("AZURE_OPENAI_ENDPOINT")
                    or env.get("AZURE_API_BASE"))
        api_key = env.get("AZURE_OPENAI_API_KEY") or env.get("AZURE_API_KEY")
        if not (endpoint and api_key):
            raise SystemExit("Azure needs an endpoint (AZURE_OPENAI_API_BASE or "
                             "AZURE_API_BASE) and a key (AZURE_OPENAI_API_KEY or "
                             "AZURE_API_KEY)")
        client = openai.AzureOpenAI(
            azure_endpoint=endpoint.rstrip("/"),
            api_key=api_key,
            api_version=env.get("AZURE_API_VERSION") or "2024-12-01-preview",
            max_retries=5,
        )
        return OpenAILLM(client, deployment, temperature=temperature)

    if provider == "openai":
        api_key = env.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is not set")
        return OpenAILLM(openai.OpenAI(api_key=api_key, max_retries=5), deployment,
                         temperature=temperature)

    raise SystemExit(f"unknown provider {provider!r}")
