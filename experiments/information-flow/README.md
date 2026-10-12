# Information flow

A Langroid agent reads a simulated corporate filesystem and sends simulated email. An untrusted document attempts to redirect it into disclosing confidential information. A multi-agent scenario passes the malicious material through a research assistant's summary.

Two policies illustrate different restrictions:

- [`mls_policy.dl`](mls_policy.dl) enforces multilevel security: no reads above the caller's clearance and no writes below its clearance.
- [`toxic_flow_policy.dl`](toxic_flow_policy.dl) blocks external email when its dependency graph includes both untrusted input and sensitive data. This allows more benign actions, but does not provide every confidentiality guarantee of the multilevel policy.

## Run

Complete the [root setup](../../README.md#run-evaluations), including Azure credentials. Langroid uses `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_API_BASE`; configure these alongside the runner's `AZURE_API_*` settings and provide a `gpt-4.1-mini` deployment. From the repository root:

```bash
uv run python scripts/evaluation/run_benchmark.py --targets infoflow \
  --example-num-trials 1 --example-max-concurrency 1
```

The runner compares natural-language-policy and instrumented conditions across the configured policies and clearances. Instrumented agents receive policy feedback without the additional natural-language policy. Files, users, and email destinations are simulated.

## Saved results

```bash
uv run python scripts/evaluation/summarize_infoflow.py results/infoflow/manifest.json
```

The summary reports task completion, policy violations, mean runtime, and mean cost per scenario. The entry point is [`information_flow/main.py`](information_flow/main.py); simulated tools and data are in [`information_flow/tools.py`](information_flow/tools.py) and [`information_flow/filesystem.py`](information_flow/filesystem.py).
