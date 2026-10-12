#!/usr/bin/env python3
"""Run the whole tau2 benchmark for several policies, one after another.

Where ``tau2_grade.py`` grades one policy on named tasks, this runs a domain's
entire task set for each of several policies and writes one ``summary.json``
comparing them: mean reward, pass^1 (the mean of the per-task success rates,
where a reward of at least 1.0 is a success),
pass^k (the share of tasks that passed on *every* trial), the per-task means,
and how many tool calls the agent made against how many of them the policy
blocked.

Execution and result handling:

* ``summary.json`` is written after every policy, so a run that dies halfway
  still has the policies that finished.
* A policy whose benchmark fails is recorded as failed with its return code and
  skipped; the remaining policies still run.  So is a policy whose benchmark
  exited zero and left behind something that is not a measurement -- no file,
  truncated JSON, no scored reward, or a task that did not produce ``--trials``
  scored simulations -- with the reason written into its row.  A run that lost
  simulations is never averaged into a row that claims the trials asked for.
* Resuming into an ``--out-dir`` that already holds a ``summary.json`` is
  refused unless the domain, the trials, the workers and both models match what
  that file already says: the header describes every row in it.
* ``--wait-for-pid`` lets one run queue behind another: the tool waits for that
  process to exit before starting, which is how two of these are scheduled
  without overlapping their model quota.
* Each policy's simulation file is named ``full_<policy>_<run token>.json``,
  the token being unique to this invocation, so two campaigns sharing a
  ``--sim-dir`` never delete or read each other's simulations.  The target is
  still deleted before the run, because tau2 resumes into a file of the same
  name.

Each policy is enforced where it lives, not from a copy: a policy's
``#include`` of the shared vocabulary is resolved relative to its own
directory, so moving the file changes what it says.  ``tau2_grade.py`` does
stage a copy, and it stages the *inlined* policy for exactly this reason -- the
copy it enforces has no includes left to resolve.

Usage:
    uv run python experiments/translation/tau2_full_run.py \\
        --policy reference=experiments/tau2/airline_policy.dl \\
        --policy candidate=outputs/run-01/deployed_policy.dl \\
        --trials 4 --workers 50 \\
        --agent-llm vertex_ai/claude-opus-4-5@20251101 \\
        --user-llm vertex_ai/claude-opus-4-5@20251101 \\
        --out-dir outputs/full-run-airline
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# tau2_grade.py is the sibling that owns the process runner and the launcher
# spelling; running this file as a script puts its directory on sys.path, and
# importing it from a test does not, so the entry is added either way.
TRANSLATION_DIR = Path(__file__).resolve().parent
if str(TRANSLATION_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSLATION_DIR))

from tau2_grade import (
    DEFAULT_SIM_DIR,
    DEFAULT_TAU2_BIN,
    REPO_ROOT,
    GradeError,
    completeness_problem,
    run_process,
    run_token,
    clear_simulation_files,
    tasks_in_simulation_file,
    tau2_command,
    written_simulation_file,
)

RUN_SCHEMA = "full-tau2-run/v1"
DEFAULT_DOMAIN = "airline"
# How often the queue check looks at the process it is waiting for.
WAIT_POLL_SECONDS = 30


def parse_policy(entry: str) -> tuple[str, Path]:
    """``name=path``: the name the summary reports, and the policy it names."""
    name, separator, raw = entry.partition("=")
    if not separator or not name or not raw:
        raise GradeError(f"--policy takes name=path, not {entry!r}")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise GradeError(f"--policy {name}: not a file: {path}")
    return name, path


def wait_for_pid(pid: int, *, poll_seconds: int = WAIT_POLL_SECONDS) -> None:
    """Block until that process is gone, so two long runs do not overlap.

    Only ``ProcessLookupError`` means gone.  ``os.kill(pid, 0)`` also raises
    ``PermissionError`` -- a subclass of ``OSError`` -- for a live process owned
    by another user, and reading that as "exited" would start this run on top of
    the run it was queued behind, sharing one model quota between them.  That is
    a process that is very much there, so it is waited for.
    """
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        except PermissionError:
            pass  # alive and owned by someone else
        time.sleep(poll_seconds)
    print(f"process {pid} exited; starting", flush=True)


def summarize(name: str, simulation_file: Path, trials: int) -> dict[str, Any]:
    """The comparison row for one policy, read out of its simulation file.

    ``blocked_calls`` counts tool results the reference monitor refused: tau2
    records a refusal as a tool message whose content begins ``[BLOCKED]``, so
    the count is what the policy stopped, task by task.

    Raises ``GradeError`` when the file is not a measurement of ``trials``
    trials: a shape that is not tau2's, no scored reward at all, or a task that
    produced some other number of scored simulations. The scored tasks must
    exactly match a valid, nonempty top-level task list; missing or malformed
    coverage is refused. Every number in the row
    -- ``sims``, the per-task means, both pass rates -- then counts the same
    scored simulations.
    """
    data = json.loads(simulation_file.read_text())
    # A file that parses as JSON and is not tau2's shape -- a list, a null, a
    # ``simulations`` entry that is not a simulation -- would raise a
    # ``TypeError`` on the first subscript, which is not one of the errors
    # ``run_one`` turns into a failed row: it would travel past ``main`` and
    # abort the campaign.  It is this one policy's failure, so it is a
    # ``GradeError`` that names the shape.
    if not isinstance(data, dict) or not isinstance(data.get("simulations"), list):
        raise GradeError(
            f"{name}: {simulation_file} is not a tau2 simulation file: "
            f"its top level is {type(data).__name__}, with no list of simulations"
        )
    by_task: dict[str, list[float | None]] = collections.defaultdict(list)
    blocked: collections.Counter[str] = collections.Counter()
    tool_calls = 0
    for simulation in data["simulations"]:
        if not isinstance(simulation, dict):
            raise GradeError(
                f"{name}: {simulation_file} holds a simulation that is not an object "
                f"({type(simulation).__name__}), so it is not a tau2 simulation file"
            )
        task = str(simulation["task_id"])
        by_task[task].append((simulation.get("reward_info") or {}).get("reward"))
        for message in simulation["messages"]:
            if message.get("tool_calls"):
                tool_calls += len(message["tool_calls"])
            content = message.get("content") or ""
            if message.get("role") == "tool" and isinstance(content, str) and content.startswith("[BLOCKED]"):
                blocked[task] += 1
    scored = {task: [value for value in values if value is not None] for task, values in by_task.items()}
    rewards = [reward for values in scored.values() for reward in values]
    if not rewards:
        raise GradeError(f"{name}: no simulation in {simulation_file} carried a reward")
    # Use the declared task set, not only the surviving simulations: a task
    # whose workers all disappeared has no scored count to be short.
    problem = completeness_problem(
        {
            "per_task": {
                task: f"{sum(value >= 1.0 for value in values)}/{len(values)}" for task, values in scored.items()
            },
            "sims": len(rewards),
        },
        tasks_in_simulation_file(simulation_file),
        trials,
        tasks_came_from_the_file=True,
    )
    if problem is not None:
        raise GradeError(f"{name}: {problem} ({simulation_file})")
    ordered = sorted(scored.items(), key=lambda item: int(item[0]) if item[0].isdigit() else 0)
    per_task = {task: sum(values) / len(values) for task, values in ordered}
    return {
        "policy": name,
        "sims": len(rewards),
        "tasks": len(by_task),
        "trials": trials,
        "mean_reward": sum(rewards) / len(rewards),
        "pass_hat_1": sum(reward >= 1.0 for reward in rewards) / len(rewards),
        "pass_hat_k_all_trials": (
            sum(1 for values in scored.values() if all(value >= 1.0 for value in values)) / len(scored)
        ),
        "tasks_all_trials_pass": sorted(
            task for task, values in scored.items() if all(value >= 1.0 for value in values)
        ),
        "tasks_all_trials_fail": sorted(
            task for task, values in scored.items() if all(value == 0.0 for value in values)
        ),
        "per_task": per_task,
        "tool_calls": tool_calls,
        "blocked_calls": sum(blocked.values()),
        "blocked_by_task": {task: blocked[task] for task, _ in ordered if blocked[task]},
        "agent": (data.get("info") or {}).get("agent_info"),
        "user": {
            key: value
            for key, value in ((data.get("info") or {}).get("user_info") or {}).items()
            if key != "global_simulation_guidelines"
        },
    }


def run_one(
    name: str,
    policy: Path,
    args: argparse.Namespace,
    tau2_bin: list[str],
    sim_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Benchmark one policy: its summary row, or the record of its failure."""
    # The policy's name says which row the file belongs to; the token says which
    # invocation, so two campaigns sharing a --sim-dir do not delete and read
    # each other's simulations.
    save_to = f"full_{name}_{run_token()}"
    # tau2 resumes into an existing file of the same name, so a previous run's
    # results would be summarized as this one's.
    clear_simulation_files(sim_dir, save_to)
    argv = [
        *tau2_bin,
        "run",
        "--domain",
        args.domain,
        "--num-trials",
        str(args.trials),
        "--max-concurrency",
        str(args.workers),
        "--agent-llm",
        args.agent_llm,
        "--user-llm",
        args.user_llm,
        "--save-to",
        save_to,
    ]
    result = run_process(
        argv,
        cwd=REPO_ROOT,
        env={"SASY_POLICY_PATH": str(policy)},
        log=out_dir / name / "run.log",
    )
    simulation_file = written_simulation_file(sim_dir, save_to)
    missing = not simulation_file.exists()
    if result.returncode != 0 or missing:
        print(
            f"POLICY FAILED: {name} rc {result.returncode} after {result.elapsed_seconds:.0f}s (see {result.log})",
            flush=True,
        )
        # A row says why it has no numbers, and it says the thing that actually
        # happened: the two faults are independent.  A benchmark can die after
        # writing a complete file, and one that exits zero having written
        # nothing says nothing at all unless this does -- that is the case a
        # stale file from an earlier run would otherwise have been summarized
        # as this run's measurement.
        if result.returncode != 0 and missing:
            reason = f"the benchmark exited {result.returncode} and wrote no simulation file at {simulation_file}"
        elif result.returncode != 0:
            reason = f"the benchmark exited {result.returncode}"
        else:
            reason = f"the benchmark wrote no simulation file at {simulation_file}"
        return {
            "policy": name,
            "policy_file": str(policy),
            "failed": True,
            "returncode": result.returncode,
            "reason": reason,
            "argv": argv,
            "log": result.log,
        }
    kept = out_dir / name / "simulations.json"
    kept.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(simulation_file, kept)
        row = summarize(name, simulation_file, args.trials)
    except (GradeError, json.JSONDecodeError, KeyError, OSError) as error:
        # A file that is there and is not a measurement -- truncated JSON from a
        # killed writer, no scored reward, a task that lost trials, a shape that
        # is not tau2's -- is this policy's failure, not the campaign's: the
        # remaining policies still run and the summary says why this one has no
        # row.  It is never averaged into a row that claims the trials asked for.
        print(f"POLICY NOT A MEASUREMENT: {name}: {error} (see {result.log})", flush=True)
        return {
            "policy": name,
            "policy_file": str(policy),
            "failed": True,
            "returncode": result.returncode,
            "reason": f"{type(error).__name__}: {error}",
            "argv": argv,
            "log": result.log,
        }
    row["elapsed_seconds"] = result.elapsed_seconds
    row["policy_file"] = str(policy)
    row["argv"] = argv
    row["simulations"] = str(kept)
    print(
        f"POLICY DONE: {name} mean {row['mean_reward']:.3f} pass^1 {row['pass_hat_1']:.3f} "
        f"pass^{args.trials} {row['pass_hat_k_all_trials']:.3f} sims {row['sims']} "
        f"blocked {row['blocked_calls']} elapsed {row['elapsed_seconds']:.0f}s",
        flush=True,
    )
    return row


def run_all(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    """Every policy in turn, with the summary rewritten after each one."""
    policies = [parse_policy(entry) for entry in args.policy]
    tau2_bin = tau2_command(args.tau2_bin)
    sim_dir = Path(args.sim_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "summary.json"
    header = {
        "domain": args.domain,
        "trials": args.trials,
        "workers": args.workers,
        "agent_llm": args.agent_llm,
        "user_llm": args.user_llm,
    }
    if summary_file.is_file():
        summary: dict[str, Any] = json.loads(summary_file.read_text())
        # Resuming into a summary written under other settings would file this
        # run's rows under the first run's header: one file saying every row
        # was four trials on one model when half of them were not.
        differing = [
            f"{field}: the file says {summary.get(field)!r}, this run is {value!r}"
            for field, value in header.items()
            if summary.get(field) != value
        ]
        if differing:
            raise GradeError(
                f"{summary_file} was written by a run with other settings, and its header describes "
                f"every row in it ({'; '.join(differing)}). Use another --out-dir, or match the settings."
            )
    else:
        summary = {"schema": RUN_SCHEMA, **header, "tasks": "all", "policies": {}}
    if args.wait_for_pid:
        wait_for_pid(args.wait_for_pid)
    failed: list[str] = []
    for name, policy in policies:
        row = run_one(name, policy, args, tau2_bin, sim_dir, out_dir)
        if row.get("failed"):
            failed.append(name)
        summary["policies"][name] = row
        summary_file.write_text(json.dumps(summary, indent=1) + "\n")
    return summary, failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a policy to run, named for the summary; repeat for several",
    )
    parser.add_argument("--trials", type=int, default=4, help="simulations per task (default: 4)")
    parser.add_argument("--workers", type=int, default=1, help="tau2 --max-concurrency (default: 1)")
    parser.add_argument("--agent-llm", required=True, help="the model the agent runs on")
    parser.add_argument("--user-llm", required=True, help="the model the user simulator runs on")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help=f"tau2 domain (default: {DEFAULT_DOMAIN})")
    parser.add_argument("--out-dir", required=True, help="where the logs, the kept simulations and summary.json go")
    parser.add_argument("--sim-dir", default=str(DEFAULT_SIM_DIR), help="where tau2 writes its simulation files")
    parser.add_argument(
        "--tau2-bin", default=DEFAULT_TAU2_BIN, help=f"benchmark launcher (default: {DEFAULT_TAU2_BIN})"
    )
    parser.add_argument(
        "--wait-for-pid",
        type=int,
        default=0,
        metavar="PID",
        help="wait for this process to exit before starting, to queue behind another run (0: start now)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary, failed = run_all(args)
    except GradeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"wrote {Path(args.out_dir).expanduser().resolve() / 'summary.json'} ({len(summary['policies'])} policies)")
    if failed:
        print(f"these policies failed and were skipped: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
