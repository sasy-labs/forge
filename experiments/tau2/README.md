# tau2: airline and retail

This case study runs the [tau2 benchmark](https://github.com/sierra-research/tau2-bench) with runtime policies for airline and retail customer service. The local benchmark is in [`../tau2-bench`](../tau2-bench/); its upstream changes are listed in [`PATCHES.md`](../tau2-bench/PATCHES.md).

The paired conditions use identical agent prompts, including the domain's natural-language policy, and the same user simulator instructions. The simulator receives an additional clarification to avoid ending the conversation while authorizing an action that still needs execution. The instrumented condition checks the domain policy; the comparison condition checks an allow-all policy.

The adapter is packaged as `forge-tau2` (Python module `forge_tau2`), separate from the upstream `tau2` benchmark. Its command remains `tau2-instrumented`.

The reference policies and C++ helpers are [`airline_policy.dl`](airline_policy.dl), [`airline_functors.cpp`](airline_functors.cpp), [`retail_policy.dl`](retail_policy.dl), and [`retail_functors.cpp`](retail_functors.cpp).

## Run

Complete the [root setup](../../README.md#run-evaluations). From the repository root:

```bash
uv run python scripts/evaluation/run_benchmark.py \
  --targets tau2 --domain airline --model gpt --user-model opus \
  --task-ids 1 --num-trials 1 --max-concurrency 1
```

For the full comparison, omit `--task-ids`, select `--domain all --model all`, and use `--num-trials 4`. The named paper results use Opus as the user simulator; pass `--user-model opus` explicitly because the runner's current default is GPT. Model aliases and exact deployment names are recorded in each new run's manifest.

## Saved results

```bash
uv run python scripts/evaluation/summarize_tau2.py results/tau2/manifest.json
```

The summary includes success, pass^k, mean runtime, mean cost, and task subsets with and without a denied action. These subsets are defined by the instrumented trajectory and paired with the corresponding comparison tasks. Benchmark reward is not a complete policy-compliance score: some global rules, including retail write locking, are not checked for every task. Inspect policy denials and traces alongside task reward.
