"""One row per tau2 cell of a ``run_benchmark.py`` run.

Reads ``<run dir>/manifest.json`` written by ``scripts/evaluation/run_benchmark.py``,
loads each tau2 cell's ``results.json`` and prints, per cell: trials, mean
reward, pass^1 and pass^k (tau2's consistency measure, ``k`` = trials per task),
tool calls the policy blocked, agent errors, trials cut short, and, for the LLM
judge mode, the judge's calls and cost.  AgentDojo, info-flow and MALADE cells
in the same manifest are listed with the command that summarizes them.
``--per-task`` adds a per-task pass count for every (domain, model) across
modes.  ``summarize_tau2.py`` prints the paired enforcement off/on table.

Usage::

    uv run python scripts/evaluation/summarize_benchmark.py results/<timestamp>
    uv run python scripts/evaluation/summarize_benchmark.py --csv results/<ts>/results.csv \\
                                                  results/<ts>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from tau2_results import (
    abnormal_terminations,
    cell_results_file,
    is_success,
    judge_log_for,
    pass_hat_k,
    resolve_results_file,
    termination_counts,
)

# Where each non-tau2 target's results are read.
_OTHER_READERS = {
    "agentdojo": "scripts/evaluation/summarize_agentdojo.py <run dir>/manifest.json",
    "infoflow": "scripts/evaluation/summarize_infoflow.py <run dir>/manifest.json",
    "malade": "scripts/evaluation/summarize_malade.py <run dir>/manifest.json",
}


def _load_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        sys.exit(f"manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text())


def _summarize_judge_log(log_path: Path) -> dict:
    """Aggregate per-call judge stats from a TAU2_LLM_POLICY_LOG_FILE
    JSONL log. Returns total cost / tokens / wall time across the cell."""
    if not log_path.is_file():
        return {}
    n_calls = 0
    cost = 0.0
    tokens_in = 0
    tokens_out = 0
    elapsed = 0.0
    n_llm_calls = 0
    n_deny = 0
    n_error = 0
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_calls += 1
            elapsed += float(rec.get("elapsed_s", 0.0) or 0.0)
            verdict = rec.get("verdict") or {}
            usage = verdict.get("usage") or {}
            cost += float(usage.get("cost_usd", 0.0) or 0.0)
            tokens_in += int(usage.get("tokens_in", 0) or 0)
            tokens_out += int(usage.get("tokens_out", 0) or 0)
            n_llm_calls += int(usage.get("n_llm_calls", 0) or 0)
            if verdict.get("authorized") is False:
                n_deny += 1
            if rec.get("error"):
                n_error += 1
    return {
        "judge_n_calls": n_calls,
        "judge_n_llm_calls": n_llm_calls,
        "judge_cost_usd": cost,
        "judge_tokens_in": tokens_in,
        "judge_tokens_out": tokens_out,
        "judge_elapsed_s": elapsed,
        "judge_n_deny": n_deny,
        "judge_n_error": n_error,
    }


def _summarize_cell(sim_path: Path) -> dict:
    """Return per-cell aggregates from a tau2 simulation JSON.

    ``sim_path`` may name either layout: ``<save_to>/results.json`` (tau2
    1.0) or the older flat ``<save_to>.json``; whichever exists is read.
    """
    sim_path = resolve_results_file(sim_path)
    if not sim_path.is_file():
        return {"missing": True, "n_sims": 0}

    data = json.loads(sim_path.read_text())
    sims = data.get("simulations") or []
    if not sims:
        return {"missing": False, "n_sims": 0}

    # An infrastructure_error simulation (tau2 1.0) carries reward_info
    # null: it is a failed trial here, and ``terminations`` counts it.
    rewards = [(s.get("reward_info") or {}).get("reward") or 0.0 for s in sims]
    by_task: dict[str, list[float]] = defaultdict(list)
    for s, reward in zip(sims, rewards):
        by_task[str(s["task_id"])].append(reward)

    # pass^1 is the success rate; pass^k (k = fewest trials any task has)
    # the chance that k trials of a task all succeed, averaged over tasks.
    counts = [(sum(1 for r in rs if is_success(r)), len(rs)) for rs in by_task.values()]
    k = min(n for _, n in counts)

    n_blocks = 0
    n_agent_errors = 0
    for s in sims:
        if s.get("termination_reason") == "agent_error":
            n_agent_errors += 1
        for m in s.get("messages") or []:
            c = m.get("content")
            if isinstance(c, str) and c.startswith("[BLOCKED]"):
                n_blocks += 1

    terminations = termination_counts(sims)
    summary = {
        "missing": False,
        "n_sims": len(sims),
        "n_tasks": len(by_task),
        "mean_reward": sum(rewards) / len(rewards),
        "k": k,
        "pass_hat_1": pass_hat_k(counts, 1),
        "pass_hat_k": pass_hat_k(counts, k),
        "n_blocked_events": n_blocks,
        "n_agent_errors": n_agent_errors,
        # Every termination_reason, counted; ``n_cut_short`` is the trials
        # that did not end with the agent or user stopping (timeout,
        # infrastructure_error, context_window_exceeded, ...).
        "terminations": terminations,
        "n_cut_short": sum(abnormal_terminations(terminations).values()),
        "by_task": {tid: rs for tid, rs in by_task.items()},
    }

    # The LLM-judge mode's per-call log: <run>/judge_log.jsonl, or beside
    # the run as <run>.judge_log.jsonl for runs written before that.
    judge_log = judge_log_for(sim_path)
    if judge_log is not None:
        summary.update(_summarize_judge_log(judge_log))

    return summary


def _print_table(rows: list[dict]) -> None:
    """ASCII table grouped by domain → model → mode.

    Adds judge-cost columns when any row has them; the SASY/baseline
    rows show ``-`` in those columns for visual alignment.
    """
    has_judge = any(r.get("judge_n_calls", 0) for r in rows)

    cols = [
        ("domain", 8), ("model", 7), ("mode", 11),
        ("n", 4), ("avg", 6), ("pass^1", 7), ("pass^k", 7), ("k", 2),
        ("blocks", 7), ("err", 5), ("cut", 5),
    ]
    if has_judge:
        cols += [
            ("j_calls", 7), ("j_$", 7), ("j_t(s)", 8),
            ("j_tok_in", 9), ("j_tok_out", 9), ("j_deny", 6),
        ]
    header = "  ".join(f"{c[0]:>{c[1]}}" for c in cols)
    print(header)
    print("  ".join("-" * c[1] for c in cols))
    rows_sorted = sorted(rows, key=lambda r: (r["domain"], r["model"], r["mode"]))
    for r in rows_sorted:
        if r.get("missing"):
            print(f'  {r["domain"]:>8}  {r["model"]:>7}  {r["mode"]:>11}  MISSING')
            continue
        if not r.get("n_sims"):
            print(f'  {r["domain"]:>8}  {r["model"]:>7}  {r["mode"]:>11}  NO SIMULATIONS')
            continue
        line = (
            f'  {r["domain"]:>8}'
            f'  {r["model"]:>7}'
            f'  {r["mode"]:>11}'
            f'  {r["n_sims"]:>4}'
            f'  {r["mean_reward"]:>6.3f}'
            f'  {r["pass_hat_1"]:>6.2%}'
            f'  {r["pass_hat_k"]:>6.2%}'
            f'  {r["k"]:>2}'
            f'  {r["n_blocked_events"]:>7}'
            f'  {r["n_agent_errors"]:>5}'
            f'  {r["n_cut_short"]:>5}'
        )
        if has_judge:
            jc = r.get("judge_n_calls", 0)
            if jc:
                line += (
                    f'  {jc:>7}'
                    f'  ${r.get("judge_cost_usd", 0.0):>5.2f}'
                    f'  {r.get("judge_elapsed_s", 0.0):>7.1f}'
                    f'  {r.get("judge_tokens_in", 0):>9}'
                    f'  {r.get("judge_tokens_out", 0):>9}'
                    f'  {r.get("judge_n_deny", 0):>6}'
                )
            else:
                line += "  " + "  ".join(f'{"-":>{c[1]}}' for c in cols[-6:])
        print(line)


def _print_mode_compare(rows: list[dict], modes: Iterable[str]) -> None:
    """Side-by-side reward comparison per (domain, model, task) across modes."""
    by_dmt = {}  # (domain, model) -> {mode -> by_task dict}
    for r in rows:
        if r.get("missing"):
            continue
        by_dmt.setdefault((r["domain"], r["model"]), {})[r["mode"]] = r["by_task"]

    for (domain, model), modes_data in sorted(by_dmt.items()):
        print(f"\n=== {domain} / {model} — per-task pass count by mode ===")
        modes_present = [m for m in modes if m in modes_data]
        if not modes_present:
            print("  no data")
            continue
        # Union of task ids across modes
        all_tids = sorted(
            {tid for d in modes_data.values() for tid in d.keys()},
            key=lambda x: int(x) if x.isdigit() else x,
        )
        header = "task   " + "  ".join(f"{m:>8}" for m in modes_present)
        print(header)
        for tid in all_tids:
            cells = []
            for m in modes_present:
                rs = modes_data[m].get(tid, [])
                if not rs:
                    cells.append("    -")
                else:
                    n = sum(1 for r in rs if r == 1.0)
                    cells.append(f"  {n}/{len(rs)}")
            print(f"  {tid:>3}  " + "  ".join(f"{c:>8}" for c in cells))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Path to results/<timestamp>/")
    parser.add_argument("--csv", type=Path, help="Also write a CSV row-per-cell here.")
    parser.add_argument("--per-task", action="store_true",
                        help="Also print per-task pass counts across modes.")
    args = parser.parse_args()

    manifest = _load_manifest(args.run_dir)
    run_dir = args.run_dir.resolve()
    manifest_run_dir = manifest.get("run_dir")

    rows: list[dict] = []
    others: dict[str, int] = defaultdict(int)
    for cell in manifest["cells"]:
        if cell.get("target", "tau2") != "tau2":
            others[cell["target"]] += 1
            continue
        if cell.get("returncode", 0) != 0:
            print(f"WARN cell {cell['domain']}/{cell['mode']}/{cell['model']}: returncode {cell['returncode']}",
                  file=sys.stderr)
        sim_path = cell_results_file(cell, run_dir, run_dir=manifest_run_dir)
        summary = _summarize_cell(sim_path) if sim_path else {"missing": True}
        rows.append({
            "domain": cell["domain"], "mode": cell["mode"], "model": cell["model"],
            "returncode": cell.get("returncode", 0),
            **summary,
        })

    print(f"\nrun: {args.run_dir.name}  ({len(rows)} cells, "
          f"{manifest.get('num_trials', '?')} trials/task)")
    print()
    _print_table(rows)
    if args.per_task:
        _print_mode_compare(rows, manifest.get("modes") or sorted({r["mode"] for r in rows}))
    for target, n in sorted(others.items()):
        reader = _OTHER_READERS.get(target, "")
        print(f"\n{n} {target} cell(s) not shown; read them with {reader}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        keys = ["domain", "model", "mode", "returncode", "n_sims",
                "mean_reward", "pass_hat_1", "pass_hat_k", "k",
                "n_blocked_events", "n_agent_errors", "n_cut_short", "terminations",
                "judge_n_calls", "judge_n_llm_calls", "judge_cost_usd",
                "judge_tokens_in", "judge_tokens_out", "judge_elapsed_s",
                "judge_n_deny", "judge_n_error"]
        with args.csv.open("w") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                if r.get("missing"):
                    continue
                w.writerow({k: (json.dumps(r[k]) if k == "terminations" and k in r else r.get(k))
                        for k in keys})
        print(f"\nCSV: {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
