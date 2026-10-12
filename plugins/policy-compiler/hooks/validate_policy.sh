#!/usr/bin/env bash
# PostToolUse hook (Write|Edit|MultiEdit): auto-validate policy
# artifacts on every write.
#
#   *.dl            → make souffle-validate (sugar + lint + Soufflé);
#                     a failure exits 2 so the error is fed back to
#                     the model immediately (fixes "skips the
#                     validator in -p mode"). Successful policy
#                     writes are recorded in
#                     .sasy-policy-run/pending.list, which arms the
#                     Stop hook (require_finalize.sh).
#   *_functors.cpp  → g++ -fsyntax-only (when g++ + Soufflé headers
#                     are available).
#
# Infra problems (no jq, no make, missing project dir) fail OPEN —
# the hook must never block unrelated work.
# Concurrency: the marker directory is per-session when
# SASY_POLICY_RUN_DIR is set. Several translation sessions can share
# one checkout (a variant set runs N of them at once), and a single
# project-root directory would let one session's pending policy block
# another's Stop hook, or one session's finalize satisfy another's
# gate.
set -uo pipefail

INPUT=$(cat)

command -v jq >/dev/null 2>&1 || exit 0
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // ""')
[[ -z "$FILE_PATH" ]] && FILE_PATH="${TOOL_INPUT_FILE_PATH:-}"
[[ -z "$FILE_PATH" ]] && exit 0

DIR="${CLAUDE_PROJECT_DIR:-}"
[[ -z "$DIR" || ! -d "$DIR" ]] && exit 0

case "$FILE_PATH" in
  *.dl)
    [[ -f "$FILE_PATH" ]] || exit 0
    # Infra fail-open: without make we can't validate, but that's
    # an environment gap, not a policy error — don't block.
    command -v make >/dev/null 2>&1 || exit 0
    # Resolve relative to the project dir for make.
    REL="$FILE_PATH"
    case "$REL" in
      "$DIR"/*) REL="${FILE_PATH#"$DIR"/}" ;;
    esac
    # Arm the finalize gate BEFORE validating. Validation is fallible, and
    # arming afterwards means a policy whose first write fails is never
    # recorded as pending — so if the session stops instead of fixing it,
    # the Stop hook sees nothing outstanding and lets the session end with
    # an invalid, never-finalized .dl on disk. A file that failed to
    # validate is the clearest case of one that still needs finalizing.
    # Arm the finalize gate only for files that look like
    # policies (not schema fragments or unrelated .dl files).
    if grep -q 'common_policy\.dl\|IsAuthorized\|Unauthorized' \
        "$FILE_PATH" 2>/dev/null; then
      RUN_DIR="${SASY_POLICY_RUN_DIR:-$DIR/.sasy-policy-run}"
      mkdir -p "$RUN_DIR"
      PENDING="$RUN_DIR/pending.list"
      touch "$PENDING"
      grep -qxF "$REL" "$PENDING" 2>/dev/null \
        || echo "$REL" >> "$PENDING"
    fi
    OUT=$(cd "$DIR" && make souffle-validate FILE="$REL" 2>&1)
    if [[ $? -ne 0 ]]; then
      echo "Policy validation failed for $REL:" >&2
      echo "$OUT" >&2
      echo "Fix the policy and re-validate with:" >&2
      echo "  make souffle-validate FILE=$REL" >&2
      exit 2
    fi
    ;;
  *_functors.cpp|*/functors.cpp)
    [[ -f "$FILE_PATH" ]] || exit 0
    command -v g++ >/dev/null 2>&1 || exit 0
    INC=""
    for d in /opt/homebrew/include /usr/local/include /usr/include; do
      [[ -f "$d/souffle/SouffleInterface.h" ]] && INC="-I$d" && break
    done
    [[ -z "$INC" ]] && exit 0
    OUT=$(g++ -fsyntax-only -std=c++17 $INC "$FILE_PATH" 2>&1)
    if [[ $? -ne 0 ]]; then
      echo "Functor C++ syntax check failed for $FILE_PATH:" >&2
      echo "$OUT" >&2
      exit 2
    fi
    ;;
esac

exit 0
