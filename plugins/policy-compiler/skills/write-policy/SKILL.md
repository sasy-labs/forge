---
name: write-policy
description: This skill should be used when the user asks to "write a policy", "generate a policy", "translate English to Datalog", "create Souffle rules", or provides an English security policy to compile.
---

# Write Soufflé Policy

Translate an English security policy into validated Soufflé Datalog
for the SASY Policy Enforcement Platform, verified against test
scenarios — and, when a live instrumented system is available,
against captured real traces labeled by a strong-LLM oracle.

**Paths:** `references/...` files live in this skill's own
directory (under the plugin root). All other paths are relative to
the repository root. Read reference files at the step that names
them — not all up front.

**Enforcement:** the plugin ships hooks. Every `.dl` write is
auto-validated (failures are fed back to you immediately), and a
session that wrote a policy cannot finish until
`make policy-finalize` passes for it (step 12). Don't fight the
hooks; they run the same commands the workflow requires anyway.

Every step below is MANDATORY unless marked optional.
"Interactive" = a human is present; "`-p`" = non-interactive mode,
where you must never wait for input.

---

## Phase A — Ground

### 1. Read the schema and config

```bash
cat policies/common_policy.dl
cat config/auth_config.yaml
cat config/transforms.json
```

`common_policy.dl` defines ALL input/output relations, functors,
and helpers. **Never invent predicates.**

### 2. Determine inputs, output file, and mode

- `FILE=` in the request → use that exact path. Otherwise:
  interactive → ask with AskUserQuestion (suggest
  `experiments/<slug>_policy.dl`); `-p` → derive
  `experiments/<slug>_policy.dl` and proceed.
- Name the policy `<name>_policy.dl` so the companion functor file
  `<name>_functors.cpp` is auto-discovered.
- **Detect the mode:**
  - **Mode L (live/trace-grounded)** — capture artifacts exist
    (a `*.decisions.jsonl` / `captures/` dir / converted
    `*captured*tests*.json` near the spec or output dir), or the
    request names a runnable instrumented target (a tau2 domain,
    an information-flow scenario, any system instrumented with the
    sasy SDK).
  - **Mode S (static)** — neither; work from the spec and source
    code alone.

### 3. Obtain real traces BEFORE writing any rule (Mode L)

If captured artifacts exist, use them. Otherwise run the capture
pipeline (steps 1–2 of `references/trace-loop.md`): start the
target under `experiments/translation/allow_all.dl` with `SASY_CAPTURE_FILE` set,
run a task or two, convert with `experiments/translation/capture_to_scenarios.py`.

This step exists so structural facts are **discovered, not
assumed**. In Mode S, skip — and treat every structural assumption
as unverified (state this in your report).

### 4. Agent analysis → K-constraints

Goal: a numbered **K-constraints block** — concrete, checkable
facts every rule must respect — written as a comment at the top of
the policy file before any rule.

- **Mode L:** examine the captured scenarios FIRST (checklist in
  `references/trace-loop.md` step 3): every distinct `msg.agent`
  value and who produces it; `agent_role` per participant; what
  backward slices actually contain for each gated tool; real
  tool-result field names/formats. Then confirm against an agent
  summary (`agent_summary.md`, if present) and the agent source.
- **Mode S:** read the agent summary if one exists — apply it; else
  read the agent source + instrumentation for the same facts.

**The classic trap:** `msg.agent_role = $UserRole()` does NOT
identify "the end user" in multi-party systems — simulators,
wrappers, and other agents also emit user-role messages. If a rule
should only match messages from a specific source, it needs a
`msg.agent` filter too. Your K-block must state which `msg.agent`
value(s) each kind of rule filters on, with evidence (trace or
source line).

Example K-block:

```datalog
/* ── K-Constraints (verified against captured traces) ──
   K1: every SentMessage rule matching user content filters
       msg.agent = "<main agent's name from the traces>" AND
       msg.agent_role = $UserRole() (other parties also emit
       $UserRole() turns with different agent values — see
       captured_tests.json)
   K2: tool-call/result rules filter msg.agent on the same value
   K3: get_user_details results carry membership as a bare
       string field "membership"
   ─────────────────────────────────────────────────────── */
```

### 5. Structured spec C1..Cn

Rewrite the English policy as a numbered clause list — one atomic
authorization decision per clause:

```
C1: Allow role=admin to use tool=delete_user
C2: Deny with error for non-admins attempting delete_user
C3: Inject "inject_api_key" for authorized api.example.com requests
```

- Compound sentences become separate clauses; use concrete role,
  tool, host, and argument names; include both allow and deny
  intent.
- **Identify content checks:** clauses conditioned on *what the
  user said* (intent, reasons, preferences — "only cancel if the
  user made a booking mistake") need a content check on
  `SentMessage` tuples in the dependency chain; the tool-call args
  reflect what the agent decided, not what the user asked.
- **Graph assumptions:** for each clause using `CurrentDepends` /
  `SentMessage` / `ToolResult`, note the assumed slice shape — in
  Mode L, cite trace evidence; in Mode S, mark unverified.
- Interactive: present the clause list for confirmation. `-p`:
  print it, self-check it covers the whole spec (count the spec's
  normative statements), continue.

### 5a. Deployment-conditional constructs — default to NOT using them

Three constructs only do anything if the deployment around the
policy implements them. Used anywhere else they do not fail — they
evaluate to nothing, so the rule that depends on one is silently
vacuous and reads as coverage while enforcing nothing. That is the
worst failure mode this skill has, so the default is off:

| Construct | Needs | Use only when |
|---|---|---|
| `@ask` (`kind="ask"`) | a host that surfaces a prompt and re-submits the answer | the request asks for it, or the codebase already handles ask-kind reasons |
| `@llm_check_fn` | `LLM_ENABLED=true` and a reachable oracle | the request explicitly asks for a semantic check |
| `ActionMetadata(idx, rel, a, b)` | the *application* passing `metadata=` into `check_tool_call` | you found the call site that supplies the facts your rule reads |

Establish support before you rely on one, from the code or from the
request — never from the fact that the relation is declared in
`common_policy.dl`. Declaration means the engine can carry it, not
that this deployment produces it. `ActionMetadata` is the sharp
case: it is declared, but the SDK only forwards whatever the
application hands it, so for an application that passes nothing the
relation is permanently empty.

If a clause genuinely needs one and support is absent, say so in
the report and implement the clause another way (a C++ functor over
message contents, an argument check) rather than emitting a rule
that cannot fire.

**What `@ask` means.** Read it as *soft-unauthorized* — the action is
not authorized to proceed autonomously and needs a human decision —
not as a weaker allow. It does not set `Unauthorized`, so the engine
returns a non-deny at the Datalog layer and downgrades the action to
a user prompt (precedence: deny > ask > allow), but that is a
mechanism detail, not the meaning. Reach for it where the policy
genuinely wants a human in the loop, never as a hedge on a clause you
are unsure how to encode: on a host that does not implement asking,
an ask-kind reason is indistinguishable from allowing the action.

## Phase B — Generate

### 6. Select patterns

Read `references/patterns.md` now. Map every clause to patterns:

| # | Pattern | # | Pattern |
|---|---------|---|---------|
| 1 | Role-based (RBAC) | 5 | Graph-based conditional |
| 2 | Blacklist / annotated deny | 6 | Taint tracking |
| 3 | Tool-based authorization | 7 | Credential transform |
| 4 | Argument conditions | 8 | Message content checks |

### 7. Generate the policy (+ companion functors)

Read `references/syntax.md` for the syntax/relations reference, and
`references/functors.md` before writing any content check.

- File template:

  ```datalog
  /* <Policy Name> */
  #include "<relative path to>/policies/common_policy.dl"

  /* ── K-Constraints ... (from step 4) */
  /* ── Structured Spec ... (from step 5) */
  ```

- **Citation grounding:** every rule carries `/* C_n: ... */`;
  every clause has ≥1 rule; no rule without a clause. Verify each
  rule against every K-constraint before moving on.
- Content checks default to **custom C++ functors**
  (`<name>_functors.cpp`, keyword matching); `@llm_check_fn` only
  on explicit user request.
- Denials use the annotated form (`// @deny_message:` /
  `// @suggestion:` lines) so denied agents get remediation hints.

### 8. Validate

```bash
make souffle-validate FILE=<path-to-policy.dl>
```

Run it yourself and show the result (the PostToolUse hook also
re-validates on every write and feeds errors back). Fix and re-run
until `✓ valid`. Error decoding table: `references/syntax.md`.

## Phase C — Verify behavior

### 9. Build the scenario suite

Format details: `references/scenarios.md`.

- **Mode L:** the converted captured scenarios are the base; label
  them with the oracle (`experiments/translation/oracle_label.py` — see
  `references/trace-loop.md` step 4; use `POLICY_CLASS=taint` for
  taint/graph-structured policies so labels stay advisory). Then
  **author top-ups** for anything capture didn't exercise:
  role-gated clauses (capture can't see roles), rare branches, and
  one impersonation probe per content rule (right content, wrong
  `msg.agent`).
- **Mode S:** author everything: ≥1 positive + ≥1 negative per
  clause, plus the impersonation probes.

### 10. Run tests / convergence loop

```bash
make policy-test POLICY=<policy> SCENARIOS=<tests>
```

All scenarios must pass. In Mode L additionally:

```bash
uv run python experiments/translation/policy_convergence.py --policy <policy> --scenarios <captured+labeled>
```

Iterate on the mismatch report (it shows which deny rule fired, the
label's reasoning, and the rendered context) until converged. For
`disputed-advisory` entries follow the dispute protocol in
`references/trace-loop.md` step 5 — interactive: ask the user;
`-p`: scripted/spec-derived expectations win over advisory oracle
labels, and you record an `ack` in the scenario file either way.

### 11. Blind round-trip audit

1. Read ONLY the generated Soufflé (with its `/* C_n */` comments).
2. WITHOUT looking at the original English, reconstruct the policy
   intent in plain English, rule by rule.
3. Compare against the structured spec; explain any mismatch.
4. Fix the Soufflé if the reconstruction reveals a bug.
5. Record `// Round-trip audit: PASS (<date>)` in the policy header.

### 12. Finalize

```bash
make policy-finalize POLICY=<policy> SCENARIOS=<tests> [TRACES=<captured+labeled>]
```

Bundles validate + config-sync (transform ids must exist in
`config/transforms.json` — missing roles in `auth_config.yaml` are
warnings) + scenario harness + convergence, and writes the
completion marker the Stop hook checks. **The session cannot end
until this passes for the current policy content.** If it fails,
fix and re-run; editing the policy afterwards re-arms the gate.

## Phase D — Optional hardening + report

### 13–14. Paraphrase check / mutation testing (optional)

Procedures in `references/paraphrase-mutation.md`. Skip silently in
`-p` mode unless the invocation asks for them.

### 15. Report

Interactive: show the validated policy, test results, and (Mode L)
the convergence summary. `-p`: end with a structured status —

```
status: success | failure
policy_file / functors_file / scenarios_file
mode: L | S
clauses: <n>  k_constraints: <n>
harness: <passed>/<total>
convergence: converged | n/a (+ disputes acked: <n>)
errors: [...]
```

— and a full worked example is in `references/example.md`.
