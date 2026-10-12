# AgentDojo

This case study evaluates prompt injection defenses on [AgentDojo](https://github.com/ethz-spylab/agentdojo) using the `tool_knowledge` attack. It compares AgentDojo baseline defenses with two runtime policies:

- **FORGE (FIDES IFC Policy):** an information-flow policy following the IFC approach used by FIDES.
- **FORGE (provenance policy):** tracks action and parameter provenance, allowing actions bound to user intent and rejecting actions or parameters derived from untrusted content.

Each policy is evaluated with and without AgentDojo's repeat-user-prompt baseline defense. Both policies use the same evaluation harness.

## Run

Complete the [root setup](../../README.md#run-evaluations). From the repository root:

```bash
uv run python scripts/evaluation/run_benchmark.py --targets agentdojo \
  --agentdojo-suite workspace --agentdojo-user-task user_task_0 \
  --agentdojo-repeats 1 --agentdojo-threads 1
```

Run from the repository root. Omit the suite/task/repeat/thread overrides to use the full default comparison. The runner detects `experiments/agentdojo/.venv/bin/python`; override it with `--agentdojo-python` or `AGENTDOJO_PYTHON`. Live runs require access to the configured GPT-4o deployment and engine authentication.

## Saved results

```bash
uv run python scripts/evaluation/summarize_agentdojo.py results/agentdojo/manifest.json
uv run python scripts/evaluation/audit_agentdojo_effect_asr.py \
  --source results/agentdojo --output results/agentdojo-effect-audit.json
```

The summary reports the paper's native and effect-based ASR. The separate audit also checks the final run's population, benchmark version, and source hashes.

**ASR (AgentDojo)** is the benchmark's native attack success rate. **ASR (Effect)** uses the same attacked population, but treats a scorer-positive Slack task-5 attempt as failed when every identifiable attacker-goal call was refused by policy. Output-only attacks remain included.

Utility is reported separately for clean and attacked tasks. Sweeps are matched across defense arms before aggregation. The entry point is [`run.py`](run.py); the runtime policy is in [`policies/agentdojo_policy.dl`](policies/agentdojo_policy.dl).
