"""Convert captured decision points into policy test scenarios.

Reads the JSONL written by the SASY_CAPTURE_FILE hook (see
packages/sasy/sasy/reference_monitor/capture.py), dedups
near-identical decision points, and emits a scenario file in the
policy_test_harness.py format with ``expected: null`` — ready for
oracle_label.py (or hand-authored labels).

Usage:
    python3 experiments/translation/capture_to_scenarios.py \\
        --capture outputs/capture/run.jsonl \\
        [--out path/to/captured_tests.json] \\
        [--dedup slice-shape|exact|none] [--max-per-tool N] \\
        [--entity ENTITY] [--roles r1,r2] \\
        [--policy-description TEXT]

Captured traces cannot see server-derived principal/roles (tau2
runs use NoAuthHook), so role-gated clauses are NOT exercised by
captured scenarios unless --entity/--roles are supplied; the
provenance carries roles_unknown so downstream reports flag this.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from policy_translation.inputs.capture import (
    capture_record_to_scenario,
    exact_key,
    slice_shape_key,
)


def load_capture(path: Path) -> list[tuple[int, dict]]:
    """Read capture JSONL; skip unparseable/incomplete lines."""
    records: list[tuple[int, dict]] = []
    skipped = 0
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not rec.get("fn_name"):
                skipped += 1
                continue
            records.append((line_no, rec))
    if skipped:
        print(
            f"NOTE: skipped {skipped} unparseable/incomplete capture lines",
            file=sys.stderr,
        )
    return records


def select_keys(
    order: list[str],
    by_tool: dict[str, list[str]],
    max_per_tool: int,
    sample: int,
    seed: int | None,
    priority_keys: set[str] | None = None,
) -> tuple[list[str], int]:
    """Choose which deduped decision points to keep.

    Stratified round-robin across tools rather than a flat cut: a
    capture is dominated by whichever read tool the agent calls most,
    so a flat first-N or a uniform draw spends the whole budget there
    and never reaches the gated writes a policy actually has to
    reason about.

    With ``seed`` the order within each tool is shuffled, so repeated
    runs over one capture draw genuinely different samples while
    staying reproducible. Without it the original file order is kept,
    which is what a single deterministic translation wants.

    Returns (selected keys in original file order, dropped count).
    """
    rank = {k: i for i, k in enumerate(order)}
    pools = {tool: list(keys) for tool, keys in by_tool.items() if keys}
    if seed is not None:
        rng = random.Random(seed)
        for keys in pools.values():
            rng.shuffle(keys)
    priority_keys = priority_keys or set()
    for keys in pools.values():
        # Stable partition after optional shuffling: previously disputed rows
        # consume the active-learning budget first, while round-robin still
        # prevents one tool from monopolizing it.
        keys.sort(key=lambda key: key not in priority_keys)

    if max_per_tool:
        for tool in pools:
            pools[tool] = pools[tool][:max_per_tool]

    # Round-robin so every tool is represented before any tool gets
    # a second slot.
    picked: list[str] = []
    tools = sorted(pools)
    if seed is not None:
        random.Random(seed).shuffle(tools)
    depth = 0
    while True:
        added = False
        for tool in tools:
            if sample and len(picked) >= sample:
                break
            if depth < len(pools[tool]):
                picked.append(pools[tool][depth])
                added = True
        if not added or (sample and len(picked) >= sample):
            break
        depth += 1

    dropped = len(order) - len(picked)
    # Restore file order: the sample decides membership, not sequence.
    return sorted(picked, key=lambda k: rank[k]), dropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert capture JSONL to harness scenarios",
    )
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output scenarios JSON (default: <capture>.scenarios.json)",
    )
    parser.add_argument(
        "--dedup",
        choices=["slice-shape", "exact", "none"],
        default="slice-shape",
    )
    parser.add_argument(
        "--max-per-tool",
        type=int,
        default=0,
        help="Keep at most N scenarios per tool (0 = unlimited)",
    )
    parser.add_argument(
        "--entity",
        default="",
        help="Entity to stamp on scenarios (capture can't see server-side identity)",
    )
    parser.add_argument(
        "--roles",
        default="",
        help="Comma-separated roles to stamp on scenarios",
    )
    parser.add_argument(
        "--principal",
        default="",
        help="Principal to stamp on scenarios (default: entity, else a placeholder — live runs always have one)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Keep at most N scenarios in total (0 = unlimited). "
        "Selection is stratified round-robin across tools, so a "
        "small budget still reaches the rare gated writes instead "
        "of filling up with whichever read tool the agent calls "
        "most.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Randomize selection reproducibly. Without it "
        "selection is first-wins by file order, so repeated runs "
        "over one capture return the SAME scenarios — fine for a "
        "single translation, wrong for repeated runs meant to be "
        "independent samples of the pipeline.",
    )
    parser.add_argument(
        "--priority-report",
        type=Path,
        default=None,
        help=(
            "prior convergence/disagreement report; scenarios named in its "
            "mismatches or disputed lists are sampled before ordinary rows"
        ),
    )
    parser.add_argument(
        "--allow-missing-slices",
        action="store_true",
        help="Proceed when captured decisions have no slice. For "
        "exploration only: the resulting file is a non-random subset "
        "of the run and must not back a reported measurement.",
    )
    parser.add_argument("--policy-description", default="")
    args = parser.parse_args()

    if not args.capture.exists():
        print(f"ERROR: capture file not found: {args.capture}")
        sys.exit(1)

    records = load_capture(args.capture)
    if not records:
        print("ERROR: no usable capture records")
        sys.exit(1)

    roles = [r.strip() for r in args.roles.split(",") if r.strip()]

    no_slice = 0
    seen: dict[str, dict] = {}
    order: list[str] = []
    by_tool: dict[str, list[str]] = {}
    per_tool: Counter[str] = Counter()
    dropped_by_limits = 0

    for line_no, rec in records:
        if not rec.get("slice"):
            # Recorded but unusable. Dropping it quietly leaves the run
            # measured against a non-random subset of its own decisions
            # — and slice queries fail on the large or tool-specific
            # graphs most likely to expose a policy mismatch, so the
            # cases that go missing are exactly the informative ones.
            no_slice += 1
            continue
        if args.dedup == "slice-shape":
            key = slice_shape_key(rec)
        elif args.dedup == "exact":
            key = exact_key(rec)
        else:
            key = f"line-{line_no}"

        if key in seen:
            seen[key]["provenance"]["dedup_count"] += 1
            continue

        scenario = capture_record_to_scenario(
            rec,
            key=key,
            entity=args.entity,
            roles=roles,
            principal=args.principal,
            capture_file=str(args.capture),
            line_no=line_no,
        )
        seen[key] = scenario
        order.append(key)
        by_tool.setdefault(rec.get("fn_name", ""), []).append(key)

    priority_names: set[str] = set()
    priority_tools: set[str] = set()
    if args.priority_report is not None:
        try:
            prior = json.loads(args.priority_report.read_text())
            collections: list[object] = []
            if isinstance(prior, dict):
                collections.extend(prior.get(field, []) for field in ("mismatches", "disputed", "episodes", "rows"))
                queue = prior.get("disagreement_queue")
                if isinstance(queue, dict):
                    collections.extend(queue.get(field, []) for field in ("rows", "episodes"))
            for rows in collections:
                if isinstance(rows, list):
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        if row.get("name"):
                            priority_names.add(str(row["name"]))
                        scenario = row.get("scenario")
                        action = scenario.get("action") if isinstance(scenario, dict) else None
                        if isinstance(action, dict) and action.get("fn_name"):
                            priority_tools.add(str(action["fn_name"]))
        except (OSError, json.JSONDecodeError) as e:
            print(f"ERROR: cannot read priority report: {e}", file=sys.stderr)
            sys.exit(1)
    priority_keys = {
        key
        for key, scenario in seen.items()
        if scenario.get("name") in priority_names or (scenario.get("action") or {}).get("fn_name") in priority_tools
    }
    order, dropped_by_limits = select_keys(
        order,
        by_tool,
        args.max_per_tool,
        args.sample,
        args.seed,
        priority_keys,
    )
    per_tool = Counter(seen[k]["action"]["fn_name"] for k in order)
    scenarios = [seen[k] for k in order]
    if not scenarios:
        print("ERROR: no scenarios produced (all records lacked slices or were deduped away)")
        sys.exit(1)

    desc = args.policy_description or (f"captured decision points from {args.capture.name}")
    out_path = args.out or args.capture.with_suffix(".scenarios.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "policy_description": desc,
                "generated_by": "capture_to_scenarios.py",
                "sampling": {
                    "strategy": "tool-stratified-active",
                    "seed": args.seed,
                    "budget": args.sample,
                    "priority_report": (str(args.priority_report) if args.priority_report else None),
                    "priority_rows_selected": sum(key in priority_keys for key in order),
                },
                "scenarios": scenarios,
            },
            f,
            indent=2,
        )
        f.write("\n")

    dupes = sum(s["provenance"]["dedup_count"] - 1 for s in scenarios)
    print(
        f"Wrote {len(scenarios)} scenarios to {out_path} "
        f"(from {len(records)} records; {dupes} duplicates "
        f"folded, {no_slice} without slices"
        + (f", {dropped_by_limits} omitted by sampling limits" if dropped_by_limits else "")
        + ")"
    )
    if no_slice and not args.allow_missing_slices:
        print(
            f"ERROR: {no_slice} of {len(records)} captured decisions have no "
            f"slice, so this file covers a non-random subset of the run. "
            f"Slice queries fail on the largest and most tool-specific "
            f"graphs, which are the ones most likely to expose a policy "
            f"mismatch, so a measurement built on the survivors is biased "
            f"in the direction that hides problems. Fix the capture (the "
            f"usual cause is an identity without the observability-reader "
            f"role), or pass --allow-missing-slices for exploratory use.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not roles:
        print(
            "NOTE: scenarios have no roles (capture cannot see "
            "server-derived identity) — role-gated clauses are "
            "not exercised by this file."
        )


if __name__ == "__main__":
    main()
