# Local patches vs upstream

Vendored copy of
[`sierra-research/tau2-bench`](https://github.com/sierra-research/tau2-bench)
at tag **v1.0.1** (commit
[`fc0055d`](https://github.com/sierra-research/tau2-bench/commit/fc0055d)).
Everything is upstream byte for byte except the patches below; domain data
(including the airline and retail `policy.md` and `tasks.json`) is unchanged.

Left out: `web/`, `.github/`, upstream's `uv.lock`, voice data (`data/voice/`,
`tasks_voice.json`, `audio_difficulty.json`), `banking_knowledge` data,
`data/tau2/results`, and `data/tau2/domains/retail/task_issues/`.
`data/simulations/` holds local run outputs. `pyproject.toml` includes
packaging changes for this repository. Local changes are marked with
`SASY patch` comments.

## Patches

1. **`TAU2_SKIP_NL_POLICY`** (`domains/airline/environment.py`,
   `domains/retail/environment.py`). When set, the domain does not load
   `policy.md`: airline keeps only the sentence giving the current time
   (`2024-05-15 15:00:00 EST`), retail gets an empty policy. Used to measure an
   agent whose rules are enforced only by the policy engine.

2. **Replay skips denied calls** (`environment/environment.py`,
   `Environment.set_state`). A recorded call whose response is a policy denial
   (`policy_denial=True`, or `error=True` with content starting `[BLOCKED]`)
   is not re-executed. A denied call never ran, so replaying it would change
   the database and, under `strict=True`, raise on the output mismatch.

3. **Policy denials are not tool errors** (`data_model/message.py`,
   `orchestrator/orchestrator.py`). `ToolMessage` gains
   `policy_denial: bool = False` (also compared in `__eq__`); the orchestrator
   counts an error only when `error and not policy_denial`. Without this, ten
   denials end a simulation with `TOO_MANY_ERRORS`.

4. **Model-call robustness** (`utils/llm_utils.py`, `generate`):
   - an empty assistant turn (no text, no tool call) is retried up to
     `TAU2_EMPTY_RESPONSE_ATTEMPTS` times (default 3, raised from 1 in
     v1) with a larger `max_tokens`, then replaced by a placeholder so the
     conversation continues (except with `tool_choice="required"`);
   - a per-request timeout of 120 s (`TAU2_REQUEST_TIMEOUT_S`);
   - litellm `num_retries` defaults to 1 (`TAU2_NUM_RETRIES`; upstream 3);
   - the cost lookup uses the requested model name with its provider prefix,
     so costs are not recorded as 0.

5. **`modify_pending_order_items` chain-collision fix**
   (`domains/retail/tools.py`). The original mutation loop searched
   `order.items` by `item_id` after earlier mutations had already changed
   ids, causing a chain collision when A→B and B→C shared an id (the first
   mutation changes A to B, so the second finds the already-mutated item).
   The validation loop now captures distinct original `OrderItem` object
   references and their validated replacement variants; the mutation loop
   applies changes through those captured references without any id search.
   Atomicity: validation failures raise before any mutation or payment.

6. **User simulator task-completion protocol** (v1)
   (`data/tau2/user_simulator/simulation_guidelines.md`,
   `simulation_guidelines_tools.md`). Consent or requesting a write is not
   task completion. The simulated user must never emit `###STOP###`,
   `###TRANSFER###`, or `###OUT-OF-SCOPE###` in a message that authorizes or
   requests an outstanding operation; it must wait for the agent to report
   actual completion or transfer. This prevents premature conversation
   termination that loses the agent's confirming turn.

7. **Golden-action failure tracking, fail-closed** (`evaluator/evaluator_env.py`).
   Both `EnvironmentEvaluator` and `FullDuplexEnvironmentEvaluator` share a
   common helper (`_env_evaluate_common`). The invalid-reference check is
   fail-closed: a failed golden action whose tool type is WRITE **or
   unknown** forces `db_reward=0` and sets `invalid_reference=True` in
   `RewardInfo.info`. Only a known READ (or THINK/GENERIC) failure is
   treated as an expected probe. `gold_action_failures` in `info` carries
   structured failure records with tool type metadata. The audit CLI
   (`scripts/evaluation/audit_references.py`) uses strict initialization,
   rejects nonexistent requested task IDs, and exits nonzero on failed
   writes or unknown types.
