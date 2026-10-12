#!/usr/bin/env bash
# Stop hook: a session that wrote a policy may not finish until
# `make policy-finalize` has passed for the CURRENT content of
# every policy it wrote.
#
# Arming: validate_policy.sh appends each successfully-validated
# policy path to .sasy-policy-run/pending.list. This hook checks
# that .sasy-policy-run/finalized/<sha256(policy)>.ok exists —
# content-hash keying means editing a policy after finalizing it
# re-blocks until finalize is re-run.
#
# Escape hatches: stop_hook_active (loop guard, set by Claude
# Code on the retry pass), SASY_SKIP_POLICY_GATE=1 (operator
# override), absent/empty pending list (no policy written this
# session — the gate never bothers unrelated work).
# Concurrency: the marker directory is per-session when
# SASY_POLICY_RUN_DIR is set. Several translation sessions can share
# one checkout (a variant set runs N of them at once), and a single
# project-root directory would let one session's pending policy block
# another's Stop hook, or one session's finalize satisfy another's
# gate.
set -uo pipefail

INPUT=$(cat)

command -v jq >/dev/null 2>&1 || exit 0
ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false')
[[ "$ACTIVE" == "true" ]] && exit 0
[[ "${SASY_SKIP_POLICY_GATE:-}" == "1" ]] && exit 0

DIR="${CLAUDE_PROJECT_DIR:-}"
[[ -z "$DIR" || ! -d "$DIR" ]] && exit 0

RUN_DIR="${SASY_POLICY_RUN_DIR:-$DIR/.sasy-policy-run}"
PENDING="$RUN_DIR/pending.list"
[[ -s "$PENDING" ]] || exit 0

MISSING=""
while IFS= read -r REL; do
  [[ -z "$REL" ]] && continue
  # validate_policy.sh records a path relative to the project dir
  # when the policy lives inside it, and an absolute path when it
  # does not. Prepending $DIR unconditionally turns the latter into
  # a path that cannot exist, and the `-f` guard below would then
  # skip it — a silent fail-OPEN in a gate that is supposed to fail
  # closed.
  case "$REL" in
    /*) POLICY="$REL" ;;
    *)  POLICY="$DIR/$REL" ;;
  esac
  [[ -f "$POLICY" ]] || continue
  # policy_finalize.py writes the marker keyed on a Python hashlib
  # sha256. Read it back with whichever hasher exists (shasum on
  # macOS, sha256sum on most Linux). This is an enforcement gate,
  # so a hashing failure fails CLOSED — report the policy as
  # not-finalized rather than silently letting the session end.
  SHA=$(shasum -a 256 "$POLICY" 2>/dev/null | cut -d' ' -f1)
  [[ -z "$SHA" ]] && SHA=$(sha256sum "$POLICY" 2>/dev/null | cut -d' ' -f1)
  if [[ -z "$SHA" ]]; then
    MISSING="$MISSING"$'\n'"- $REL (could not hash to check finalize marker)"
    continue
  fi
  if [[ ! -f "$RUN_DIR/finalized/$SHA.ok" ]]; then
    MISSING="$MISSING"$'\n'"- $REL"
  fi
done < "$PENDING"

[[ -z "$MISSING" ]] && exit 0

REASON="Policy file(s) were written but not finalized:${MISSING}

Run the finalize gate for each (validate + config-sync + scenario tests + convergence):
  make policy-finalize POLICY=<policy.dl> SCENARIOS=<tests.json> [TRACES=<labeled.json>]

If no scenario file exists yet, write one per the write-policy skill (>=1 positive and >=1 negative scenario per clause). Editing a policy after finalizing re-arms this gate."

# Build the JSON with jq so policy paths containing quotes or
# backslashes can't produce malformed output (which Claude Code
# would treat as a non-block — a fail-open).
jq -n --arg reason "$REASON" '{decision: "block", reason: $reason}'

exit 0
