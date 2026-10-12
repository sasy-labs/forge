# MALADE

This case study adapts [MALADE](https://github.com/jihyechoi77/malade), a Langroid multi-agent pharmacovigilance system. Agents use FDA drug labels to assess drug–outcome associations. The evaluation compares natural-language instructions with runtime enforcement of FDA-access approval.

The [policy](malade_policy.dl) requires an FDAHandler agent to register before accessing FDA labels. The orchestration driver may perform the initial NDC drug-category discovery before an agent message exists. Representative drug answers are constrained to the candidates returned by that discovery step.

## Run

Complete the [root setup](../../README.md#run-evaluations), including Azure and OpenFDA credentials. MALADE's Langroid clients use `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_API_BASE`, in addition to the runner's `AZURE_API_*` settings. The default Azure setup needs deployments named `gpt-4.1` and `text-embedding-3-small`; leave `AZURE_OPENAI_DEPLOYMENT_NAME` unset so chat and embedding clients select their respective model names.

MALADE also needs Qdrant for concurrent vector search. Start a local instance using the version from the development environment:

```bash
docker run -d --name forge-evaluation-qdrant \
  -p 127.0.0.1:6333:6333 qdrant/qdrant:v1.15.1
export QDRANT_API_URL=http://localhost:6333
```

Alternatively, configure an existing Qdrant instance with `QDRANT_API_URL` and, if required, `QDRANT_API_KEY`. Without a URL, Langroid falls back to embedded local storage, which is unsuitable for the concurrent benchmark. From the repository root:

```bash
uv run python scripts/evaluation/run_benchmark.py --targets malade \
  --example-num-trials 1 --example-max-concurrency 1
```

Remove the trial/concurrency overrides for the runner's standard evaluation budget. Both conditions run the same application; the instrumented condition receives policy feedback instead of the additional natural-language FDA-access policy.

## Saved results

```bash
uv run python scripts/evaluation/summarize_malade.py results/malade/manifest.json
```

The summary reports prediction correctness, FDA-access compliance, mean runtime, and mean cost per trial. An FDA lookup with no matches is distinct from a policy denial; incomplete trials are reported separately from completed predictions.

The main entry point is [`malade/omop_observability.py`](malade/omop_observability.py); FDA handling is in [`malade/doc/fda_handler.py`](malade/doc/fda_handler.py).
