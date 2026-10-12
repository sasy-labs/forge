# The Trace Loop: Capture → Label → Converge → Live Verify

Use this whenever a live/instrumented target system is available
(Mode L in the core workflow). The loop replaces guesswork about
graph structure with observed traces, and replaces "looks right" with
convergence against labeled decision points.

```
1. CAPTURE   live run, allow-all policy   → decisions JSONL
2. CONVERT   decisions → test scenarios   → captured_tests.json
3. EXAMINE   read the scenarios FIRST     → K-constraints
4. LABEL     strong-LLM oracle (cached)   → expected/labels
5. CONVERGE  harness vs labels, iterate   → mismatch report → fix policy
```

Steps 1–2 need a running server (`make serve`) and the target
experiment's LLM credentials. Steps 3 and 5 are local checks; step 4
makes model calls unless the verdicts are already cached.

## 1. Capture natural traces

Run the instrumented target with the **allow-all** policy so denials
don't truncate the traces, and decision-point capture enabled:

```bash
# tau2 (airline / retail):
SASY_POLICY_PATH=$PWD/experiments/translation/allow_all.dl \
SASY_CAPTURE_FILE=$PWD/outputs/capture/run.jsonl \
uv run tau2-instrumented run --domain airline \
    --task-ids 8 --num-trials 1

# information-flow (toxic-flow scenarios):
SASY_POLICY_PATH=$PWD/experiments/translation/allow_all.dl \
SASY_CAPTURE_FILE=$PWD/outputs/capture/run.jsonl \
uv run python experiments/information-flow/information_flow/main.py \
    --instrumented True --policy_label "Toxic Flow" --trials 1
```

Every reference-monitor decision appends a JSON line: tool call,
backward slice (full node + edge data), decision. Works for any
instrumented system — the hook lives in
`sasy.reference_monitor.check_tool_call` (env-gated, zero overhead
when unset).

Capturing **under enforcement** (candidate policy instead of
allow-all) also works and records real denials — useful for the
final verification, less useful for label generation because denials
cut conversations short.

## 2. Convert to scenarios

```bash
uv run python experiments/translation/capture_to_scenarios.py --capture outputs/capture/run.jsonl \
    --out outputs/loop/captured_tests.json
```

Dedups near-identical decision points (slice-shape hash), preserves
the live decision + provenance, leaves `expected: null`. Captured
scenarios carry `roles_unknown: true` — role-gated clauses are NOT
exercised by them; author those by hand (references/scenarios.md).

## 3. Examine the traces BEFORE writing rules

Read the captured scenarios and answer, with evidence:

- What distinct `msg.agent` values appear, and which participant
  produces each? (This is how you DISCOVER that e.g. a
  user-simulator or helper agent also speaks with
  `agent_role = user` — and therefore which `msg.agent` filter
  every content rule needs.)
- What `agent_role` does each participant use?
- What does a typical backward slice look like for each tool you
  need to gate — which tool results and user turns are actually in
  the chain?
- What do tool-result payloads actually contain (field names,
  formats) for the fields your rules will read?

Record the answers as the K-constraints block (core workflow step
4). Constraints must cite trace evidence ("in
captured_tests.json, `get_user_details` results carry `membership`
as a bare string").

## 4. Label with the strong oracle

```bash
uv run python experiments/translation/oracle_label.py --scenarios outputs/loop/captured_tests.json \
    --policy-nl path/to/english_policy.md
# taint/graph-structured policies — labels become ADVISORY:
uv run python experiments/translation/oracle_label.py --scenarios ... --policy-nl ... --policy-class taint
```

Defaults: `claude-opus-5` via `ANTHROPIC_API_KEY`
(`SASY_ORACLE_MODEL` overrides), linear-history rendering, and a
content-hash verdict cache at
`outputs/oracle-cache.json` so re-runs are free.

**Label accounting.** Model labels provide empirical evidence rather than
a correctness guarantee. Graph and taint-policy labels are advisory:

- non-weak oracle labels populate `expected` and are enforced;
- `oracle_weak: true` labels (set by `--policy-class taint` or graph
  context rendering) are advisory: convergence surfaces
  disagreements as `disputed-advisory` instead of failing on them.

## 5. Converge

```bash
uv run python experiments/translation/policy_convergence.py --policy experiments/my_policy.dl \
    --scenarios outputs/loop/captured_tests.json \
    --report outputs/loop/convergence.json
```

For each mismatch the report gives: the harness decision and which
`DenialReason` rules fired, the label + its reasoning, the live
decision, and the rendered linear context (exactly what the oracle
saw). Fix the policy (or the scenario, if the label is provably
wrong) and re-run; the oracle cache makes iteration free.

**Dispute protocol** for `disputed-advisory` entries (advisory label
disagrees with the policy):

- Interactive: show the user the rendered context and both
  positions; let them decide.
- Non-interactive (`-p`): scripted/spec-derived expectations beat
  advisory oracle labels. Record the resolution in the scenario:

  ```json
  "ack": {"resolution": "label-wrong",
          "note": "scripted ground truth: this flow is benign"}
  ```

  (`resolution` ∈ `label-wrong` | `policy-correct-by-design`.)

Converged = zero mismatches against trusted labels AND every
disputed entry acked. `make policy-finalize` enforces this.

## Grading a finished policy

The loop ends at convergence. Measuring a finished policy against a
live benchmark is the caller's job, not this skill's — see
`experiments/translation/run_variant_set.py`, which pins each policy per
session so several can be graded at once against one server.

If a live run fails while offline convergence is green, that is
usually a scenario gap rather than a rule bug: capture the failing
run, convert it, label it, and fold the new scenarios into the
suite.

## When there is no live system (Mode S)

Skip steps 1–5; author all scenarios by hand
(references/scenarios.md) and rely on
validate + harness + round-trip audit. State explicitly in your
report that graph-structure assumptions are unverified by traces.
