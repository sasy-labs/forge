# Policy translation

The paper studies five independent translations of the airline policy and an
earlier set of five translations. Each policy is evaluated on airline tasks
8, 43, and 47, with five trials per task. Construction traces come from ten
different tasks, disjoint from these evaluation tasks.

## Saved results

The paper artifacts are in [results/translation](../../results/translation):

| Set | Mean reward across policies | Standard deviation | Artifacts |
| --- | ---: | ---: | --- |
| Round 2, main appendix table | 0.893 | 0.138 | Five policies, functors, construction captures, labels, convergence reports, and held-out simulation results |
| Round 1, earlier configuration | 0.853 | 0.185 | Five policies, functors, and held-out simulation results |
| Manually authored reference | 1.000 | — | Held-out simulation results |

Round 2's per-policy rewards are 0.933, 1.000, 1.000, 0.867, and 0.667.
The saved `round2/summary.json` records these results and the construction
task selection. Each `round*/run*/grade.json` contains the fifteen held-out
simulations. The reference results are in `reference/grade.json`; the historical
reference policy source is not included.

Round 1's offline harness supplied an empty tool-call list (`tools_json = "[]"`).
Rules reading that field were therefore checked against an incomplete event
representation. Round 1 and Round 2 are reported separately. Round 1's construction captures
are not included.

The construction task specification is [`spec.md`](spec.md).

## Run a new experiment

First follow the repository [setup instructions](../../README.md). Translation
also needs the Claude Code CLI, `jq`, `make`, Soufflé, and a C++ compiler. It
uses the repository's `policy-compiler` plugin and the public SASY implementation
for the event schema, preprocessing, and enforcement. Leave `SASY_REPO_ROOT` unset to use the configured implementation.

After setup, start the local engine in a separate terminal:

```bash
make serve
```

This uses the pinned public engine, local benchmark authentication and TLS,
and provider credentials from the process environment or `.env-credentials`.
It listens on `127.0.0.1:11089`. The benchmark runner normally manages its own
engine; use this separate server for the standalone translation harness.

In the experiment terminal, pass the generated local settings with uv's
`--env-file` options. Capturing graph slices requires an identity with
`observability-reader`; setup provides an admin certificate with that role.

Run the independent-trace protocol, using the default five trials on each
held-out task:

```bash
uv run --env-file .env --env-file .env-credentials python experiments/translation/run_variant_set.py \
  --runs 5 --out results/new-translation \
  --capture-cert certs/admin.crt --capture-key certs/admin.key \
  --grade-tasks 8 43 47 --trials 5 \
  --trajectory-mode independent --repair-rounds 0
```

Select available agent and user models with `--agent-llm` and `--user-llm`.
The translation oracle uses Anthropic by default; `--oracle-provider litellm --oracle-model MODEL`
selects another configured provider. The translator runs through Claude Code.

The stages are capture, sample, label, translate, and grade. Every branch gets
its own construction traces, while grading uses tasks excluded from capture.
The runner copies and hashes the policy and evidence before grading, checks recorded labels, and verifies task coverage.

## Evaluate a saved policy

Grade a translated policy again without rerunning construction:

```bash
uv run python experiments/translation/tau2_grade.py \
  --policy results/translation/round2/run1/airline_policy.dl \
  --task-ids 8 --task-ids 43 --task-ids 47 \
  --agent-llm gpt-4.1 --user-llm gpt-4.1 \
  --trials 5 --workers 8 --out-dir results/translation-regrade
```

The grader copies the policy and companion functors, resolves shared includes,
and records the exact copy it evaluates. For policy checks, see `experiments/translation/generated_policy_fidelity.py` and `scripts/policy_test_harness.py`.
