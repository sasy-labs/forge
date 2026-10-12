#!/usr/bin/env python3
"""Offline audit of tau2 reference (golden) trajectories.

Validates that every golden action in a task's evaluation_criteria
successfully replays against the domain's environment. Reports failures
and exits nonzero when a golden WRITE action fails (invalid reference)
or when any action has an unknown tool type classification.

Usage:
    uv run python scripts/evaluation/audit_references.py --domain retail
    uv run python scripts/evaluation/audit_references.py --domain airline --task-ids 1 2 3
    uv run python scripts/evaluation/audit_references.py --domain retail --split base --json

Environment:
    TAU2_DATA_DIR: override tau2 data directory (default: experiments/tau2-bench/data)
    LITELLM_LOCAL_MODEL_COST_MAP: set to True for offline use
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure tau2 is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "tau2-bench" / "src"))

from tau2.data_model.tasks import Task
from tau2.environment.toolkit import ToolType, get_tool_types
from tau2.evaluator.evaluator_env import _replay_golden_actions
from tau2.registry import registry


def classify_failure(
    failure: dict,
    tool_types: dict[str, ToolType] | None,
) -> str:
    """Classify a golden action failure.

    Returns one of:
      - "failed_write": a WRITE tool failed (invalid reference)
      - "failed_read": a READ tool failed (expected probe, allowed)
      - "unknown_type": tool type not in the toolkit metadata
      - "failed_other": THINK/GENERIC tool failed
    """
    tt_str = failure.get("tool_type")
    if tt_str is None and tool_types:
        resolved = tool_types.get(failure["name"])
        tt_str = resolved.value if resolved else None
    if tt_str is None:
        return "unknown_type"
    if tt_str == ToolType.WRITE.value:
        return "failed_write"
    if tt_str == ToolType.READ.value:
        return "failed_read"
    return "failed_other"


def audit_task(
    task: Task,
    domain: str,
    tool_types: dict[str, ToolType] | None = None,
) -> dict:
    """Audit a single task's golden actions.

    Returns a dict with:
      - task_id
      - num_actions
      - failures: list of classified failure records
      - has_failed_write: bool
      - has_unknown_type: bool
    """
    result = {
        "task_id": task.id,
        "num_actions": 0,
        "failures": [],
        "has_failed_write": False,
        "has_unknown_type": False,
    }

    if task.evaluation_criteria is None:
        return result
    golden_actions = task.evaluation_criteria.actions or []
    result["num_actions"] = len(golden_actions)
    if not golden_actions:
        return result

    # Build the gold environment
    env_constructor = registry.get_env_constructor(domain)
    gold_env = env_constructor()

    initialization_data = None
    initialization_actions = None
    message_history = []
    if task.initial_state is not None:
        if task.initial_state.initialization_data is not None:
            initialization_data = task.initial_state.initialization_data
        if task.initial_state.initialization_actions is not None:
            initialization_actions = task.initial_state.initialization_actions
        if task.initial_state.message_history is not None:
            message_history = task.initial_state.message_history

    try:
        gold_env.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
            strict=True,
        )
    except Exception as e:
        result["failures"] = [{
            "name": "(initialization)",
            "arguments": {},
            "error": str(e),
            "tool_type": None,
            "category": "init_failure",
        }]
        result["has_unknown_type"] = True
        return result

    failures = _replay_golden_actions(gold_env, golden_actions, tool_types)
    classified = []
    for f in failures:
        category = classify_failure(f, tool_types)
        classified.append({**f, "category": category})
        if category == "failed_write":
            result["has_failed_write"] = True
        if category == "unknown_type":
            result["has_unknown_type"] = True

    result["failures"] = classified
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Audit tau2 reference trajectories for golden action failures"
    )
    parser.add_argument(
        "--domain",
        required=True,
        help="tau2 domain name (e.g. 'retail', 'airline')",
    )
    parser.add_argument(
        "--split",
        default="base",
        help="Task split name (default: 'base')",
    )
    parser.add_argument(
        "--task-ids",
        nargs="*",
        help="Specific task IDs to audit (default: all)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    # Load tasks
    tasks_loader = registry.get_tasks_loader(args.domain)
    tasks = tasks_loader(args.split)
    if args.task_ids:
        id_set = set(args.task_ids)
        all_ids = {str(t.id) for t in tasks}
        missing = id_set - all_ids
        if missing:
            print(f"Error: requested task IDs not found: {sorted(missing)}", file=sys.stderr)
            sys.exit(2)
        tasks = [t for t in tasks if str(t.id) in id_set]

    # Get tool types from the environment
    tool_types: dict[str, ToolType] | None = None
    try:
        env = registry.get_env_constructor(args.domain)()
        if env.tools is not None:
            tool_types = get_tool_types(env.tools)
    except Exception:
        pass

    results = []
    total_tasks = 0
    total_failures = 0
    failed_writes = 0
    unknown_types = 0

    for task in tasks:
        total_tasks += 1
        audit = audit_task(task, args.domain, tool_types)
        if audit["failures"]:
            results.append(audit)
            total_failures += len(audit["failures"])
            if audit["has_failed_write"]:
                failed_writes += 1
            if audit["has_unknown_type"]:
                unknown_types += 1

    if args.json_output:
        output = {
            "domain": args.domain,
            "split": args.split,
            "total_tasks": total_tasks,
            "tasks_with_failures": len(results),
            "total_failures": total_failures,
            "failed_writes": failed_writes,
            "unknown_types": unknown_types,
            "details": results,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Domain: {args.domain}, split: {args.split}")
        print(f"Audited {total_tasks} tasks")
        if not results:
            print("All golden actions replay successfully.")
        else:
            print(f"\n{len(results)} task(s) with failures ({total_failures} total):")
            for r in results:
                print(f"\n  Task {r['task_id']} ({r['num_actions']} actions):")
                for f in r["failures"]:
                    cat = f["category"]
                    marker = "FAIL" if cat in ("failed_write", "unknown_type", "init_failure") else "WARN"
                    print(f"    [{marker}] {f['name']}: {f['error']} [{cat}]")

        if failed_writes:
            print(f"\n{failed_writes} task(s) with failed golden WRITE actions (invalid reference)")
        if unknown_types:
            print(f"{unknown_types} task(s) with unknown tool type classification")

    # Exit nonzero on failed writes or unknown types
    if failed_writes or unknown_types:
        sys.exit(1)


if __name__ == "__main__":
    main()
