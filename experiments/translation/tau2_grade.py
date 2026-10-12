#!/usr/bin/env python3
"""Grade one policy on tau2 tasks, and write down what was graded.

This is the grading step of the variant-set pipeline as a tool of its own:
point it at a ``.dl`` policy and it runs the tau2 benchmark with that policy
enforced, then writes ``grade.json`` -- the reward, the per-task pass counts,
the command line it ran, and the digest of the exact policy bytes those numbers
came from.

Grading uses an immutable policy copy, invocation-specific simulation files,
and complete trial counts:

* **A copy is graded, not the original.** The policy is written into
  a fresh directory under ``--out-dir/policy`` with its whole ``#include``
  graph resolved into it, and
  the functor sources the SDK loads from beside it are copied there too; that
  copy is what is enforced.  Inlining is what makes the copy say the same thing
  as the original: an ``#include`` is resolved relative to the file that holds
  it, so a byte-for-byte copy in another directory would name a different
  shared vocabulary, or none.  The copy carries no includes at all, it is
  digested, and that digest is what ``grade.json`` reports -- so editing an
  include target afterwards changes nothing about what was measured.
* **A stale file is never read.** tau2 resumes into an existing simulation file
  of the same name, so the target is deleted before the run and nothing is
  parsed unless the benchmark exited zero.  The name is
  ``tau2_grade_<policy digest>_<run token>.json``: the digest says which policy
  the file belongs to and the token which invocation, so two grades of one
  policy sharing a ``--sim-dir`` never delete or read each other's file.
* **Partial results are not a measurement.** Every task must have exactly
  ``--trials`` scored simulations, and the tasks that came back must be the
  tasks the run was over: the ones ``--task-ids`` named, or, when it named
  none, the ones the simulation file's own top-level ``tasks`` list holds.
  Reading that list is what catches a task none of whose simulations were
  scored -- it has no entry among the per-task counts to be short.  A run that
  lost half its simulations to a crashed worker fails here instead of being
  averaged.

The reward reader is ``run_variant_set.read_reward`` and the companion-functor
rule is ``run_variant_set.companion_names``: the same code the variant-set
harness grades with, so a number produced here means what a number produced
there means.

Usage:
    uv run python experiments/translation/tau2_grade.py \\
        --policy experiments/tau2/airline_policy.dl \\
        --task-ids 8 --task-ids 43 --task-ids 47 \\
        --trials 4 --workers 8 \\
        --agent-llm vertex_ai/claude-opus-4-5@20251101 \\
        --user-llm vertex_ai/claude-opus-4-5@20251101 \\
        --out-dir outputs/tau2-grade/airline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRANSLATION_DIR = Path(__file__).resolve().parent

# What a grade record says about itself, and the stamp copied into the reward
# summary so that a pooled number carries the harness that produced it.
GRADE_SCHEMA = "sasy-tau2-grade/v1"
HARNESS_STAMP = "tau2_grade/v1"

DEFAULT_DOMAIN = "airline"
DEFAULT_TAU2_BIN = "uv run tau2-instrumented"
DEFAULT_SIM_DIR = REPO_ROOT / "experiments" / "tau2-bench" / "data" / "simulations"


class GradeError(Exception):
    """A refusal: the grade was not produced, and no number is reported."""


@dataclass(frozen=True)
class ProcessResult:
    argv: list[str]
    returncode: int
    elapsed_seconds: float
    log: str


def run_process(argv: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> ProcessResult:
    """Run one child process with its output streamed into ``log``.

    Every subprocess these drivers start goes through here, so a test drives
    the whole tool by naming a different binary and nothing else is replaced.
    """
    started = time.time()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as handle:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            env={**os.environ, **env},
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return ProcessResult(
        argv=list(argv),
        returncode=completed.returncode,
        elapsed_seconds=round(time.time() - started, 6),
        log=str(log),
    )


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_token() -> str:
    """A token no other invocation of these tools shares.

    ``--sim-dir`` is one shared directory, and the simulation file's name is
    what tau2 resumes into, so a name built out of the policy alone is the same
    name for two runs of that policy -- and each of them deletes and then reads
    the other's file.  Three parts, because no one of them is enough: the
    process id separates two runs on one host, the nanosecond clock separates
    two hosts and two runs of one process, and four random bytes cover the
    clock's real resolution, which is coarse enough that two calls in quick
    succession read the same nanosecond.
    """

    return f"{os.getpid()}-{time.time_ns():x}-{os.urandom(4).hex()}"


def simulation_file_paths(sim_dir: Path, save_to: str) -> tuple[Path, Path]:
    """Where ``tau2 run --save-to <save_to>`` writes its results, newest layout first.

    tau2 1.0 writes ``<sim_dir>/<save_to>/results.json``
    (``tau2.runner.batch.run_domain``); earlier releases wrote the flat file
    ``<sim_dir>/<save_to>.json``.  A launcher given with ``--tau2-bin`` can be
    either, so both are cleared before a run and read after it.
    """
    return sim_dir / save_to / "results.json", sim_dir / f"{save_to}.json"


def clear_simulation_files(sim_dir: Path, save_to: str) -> None:
    """Remove anything tau2 would resume into under this name, in either layout.

    The whole ``<save_to>`` directory goes, not only its ``results.json``:
    tau2 1.0 resumes from what it finds there, and the name is this
    invocation's own (``run_token``), so nothing else lives in it.
    """
    run_dir = sim_dir / save_to
    if run_dir.is_dir() and not run_dir.is_symlink():
        shutil.rmtree(run_dir)
    elif run_dir.exists() or run_dir.is_symlink():
        run_dir.unlink()
    sim_dir.joinpath(f"{save_to}.json").unlink(missing_ok=True)


def written_simulation_file(sim_dir: Path, save_to: str) -> Path:
    """The results file the run left, in whichever layout; the 1.0 path when there is none."""
    current, legacy = simulation_file_paths(sim_dir, save_to)
    if not current.exists() and legacy.exists():
        return legacy
    return current


def tau2_command(spec: str) -> list[str]:
    """The benchmark launcher, written as a command line rather than one path.

    ``uv run tau2-instrumented`` is three words and a bare path is one; both are
    the same thing to a caller, so the argument is split the way a shell would.
    """
    argv = shlex.split(spec)
    if not argv:
        raise GradeError("--tau2-bin is empty")
    return argv


def include_resolver() -> Callable[[Path], str]:
    """The SDK's ``#include`` resolver, the one the benchmark itself loads with.

    Imported here rather than at module import time, for the reason
    ``grading_helpers`` gives: ``--help`` and every argument check of this tool
    run without the ``sasy`` SDK installed.  Reused rather than reimplemented,
    because a second resolver would be a second answer to what the policy says.
    """
    from sasy.policy.api import resolve_includes

    return resolve_includes


def grading_helpers() -> tuple[Callable[..., dict[str, Any]], Callable[[Path], list[str]]]:
    """``run_variant_set``'s reward reader and companion-functor rule.

    Imported here rather than at module import time: ``run_variant_set`` needs
    the ``sasy`` SDK, and ``--help`` and every argument check of this tool run
    without it.  Reused rather than reimplemented, because a second reward
    reader would be a second definition of what a grade is.
    """
    if str(TRANSLATION_DIR) not in sys.path:
        sys.path.insert(0, str(TRANSLATION_DIR))
    import run_variant_set

    return run_variant_set.read_reward, run_variant_set.companion_names


def stage_policy_copy(source: Path, destination_dir: Path, companion_names: Callable[[Path], list[str]]) -> Path:
    """Write the graded policy, includes resolved, into our own tree with its functors.

    The staged file is the *inlined* policy -- ``run_variant_set.inlined_policy``
    does the same thing for the same reason.  A policy's ``#include`` is
    resolved relative to the file that holds it, so copying the bytes into
    another directory either fails to resolve or silently resolves to some other
    file: the copy would not say what the original says, and the digest would
    cover neither. Resolving first makes the staged copy self-contained.

    The SDK loads ``functors.cpp`` (and ``<stem>_functors.cpp``) from beside the
    policy file, so the copy has to carry them; grading the caller's file would
    measure bytes that something else is still free to rewrite.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / source.name
    try:
        inlined = include_resolver()(source)
    except FileNotFoundError as error:
        raise GradeError(f"--policy {source} has an #include that does not resolve: {error}") from error
    if not inlined.endswith("\n"):
        inlined += "\n"
    target.write_text(inlined)
    for name in companion_names(source):
        companion = source.parent / name
        if companion.is_file():
            shutil.copy2(companion, destination_dir / name)
    return target


def build_argv(
    tau2_bin: list[str],
    *,
    domain: str,
    task_ids: list[str],
    trials: int,
    workers: int,
    agent_llm: str,
    user_llm: str,
    save_to: str,
) -> list[str]:
    """The benchmark command line, in the shape run_variant_set uses."""
    argv = [*tau2_bin, "run", "--domain", domain]
    if task_ids:
        argv += ["--task-ids", *task_ids]
    argv += [
        "--num-trials",
        str(trials),
        "--max-concurrency",
        str(workers),
        "--agent-llm",
        agent_llm,
        "--user-llm",
        user_llm,
        "--save-to",
        save_to,
    ]
    return argv


def tasks_in_simulation_file(path: Path) -> list[str]:
    """The task ids the simulation file says the run was over.

    tau2 writes the tasks it ran beside the simulations, as a top-level
    ``tasks`` list whose entries each carry an ``id``.  That list is the only
    record of a task whose every simulation went unscored: such a task has no
    per-task counts to be short, so without it a run that lost one task whole
    reads as complete. Unknown or malformed coverage is refused rather than
    inferred from whichever simulations survived.
    """

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise GradeError(f"cannot read the expected task list in {path}: {error}") from error
    listed = data.get("tasks") if isinstance(data, dict) else None
    if not isinstance(listed, list) or not listed:
        raise GradeError(f"{path}: expected a nonempty top-level task list to establish domain coverage")
    ids: list[str] = []
    seen: set[str] = set()
    for index, task in enumerate(listed):
        task_id = task.get("id") if isinstance(task, dict) else None
        if type(task_id) not in (str, int) or not str(task_id).strip():
            raise GradeError(f"{path}: task list entry {index} has no valid string or integer id")
        normalized = str(task_id)
        if normalized in seen:
            raise GradeError(f"{path}: task list contains duplicate id {normalized!r}")
        ids.append(normalized)
        seen.add(normalized)
    return ids


def completeness_problem(
    reward: dict[str, Any],
    task_ids: list[str],
    trials: int,
    *,
    tasks_came_from_the_file: bool = False,
) -> str | None:
    """Why this run is not a measurement of the tasks it was over.

    ``read_reward`` reports each task as ``"<passed>/<total>"``.  A task whose
    total is not ``trials`` lost simulations; a task that is absent was never
    run, or was run and never scored; a task nobody asked for means the file is
    some other run's.

    ``task_ids`` is the set the run was over.  It is what ``--task-ids`` named,
    and when it named nothing the caller passes the file's own task list
    instead and sets ``tasks_came_from_the_file``, which only changes how the
    mismatch is phrased. An empty ``task_ids`` is refused: coverage cannot be
    established without a record of which tasks were meant to run.
    """
    per_task: dict[str, str] = dict(reward.get("per_task") or {})
    if not per_task:
        return "no simulation carried a reward, so there is nothing to report"
    short = []
    for task, counts in sorted(per_task.items()):
        total = int(str(counts).split("/")[-1])
        if total != trials:
            short.append(f"{task}={total}")
    if short:
        return f"these tasks did not produce {trials} scored simulations each: {sorted(short)}"
    if not task_ids:
        return "no expected task list is available, so domain coverage cannot be established"
    observed, asked = set(per_task), set(task_ids)
    if observed != asked:
        source = "the file's own task list holds" if tasks_came_from_the_file else "the run asked for"
        return f"the simulations are for tasks {sorted(observed)}, but {source} {sorted(asked)}"
    expected = len(per_task) * trials
    if int(reward.get("sims", 0)) != expected:
        return f"the file holds {reward.get('sims')} scored simulations, expected {expected}"
    return None


def grade(args: argparse.Namespace) -> dict[str, Any]:
    """Grade one policy and return the record written to ``grade.json``."""
    policy = Path(args.policy).expanduser().resolve()
    if not policy.is_file():
        raise GradeError(f"--policy is not a file: {policy}")
    out_dir = Path(args.out_dir).expanduser().resolve()
    sim_dir = Path(args.sim_dir).expanduser().resolve()
    tau2_bin = tau2_command(args.tau2_bin)
    task_ids = list(args.task_ids or [])

    read_reward, companion_names = grading_helpers()
    # Each run gets a fresh tree. Reusing a directory would leave removed
    # companions behind, and the SDK could load an old functors.cpp ahead of
    # the current policy's own named companion. Keep prior runs intact.
    staging_root = out_dir / "policy"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix="run-", dir=staging_root))
    graded = stage_policy_copy(policy, staging_dir, companion_names)
    sha256 = digest_file(graded)
    # The digest says which policy this file belongs to; the token says which
    # invocation, so two grades of one policy in a shared --sim-dir do not
    # delete and read each other's simulations.
    save_to = f"tau2_grade_{sha256[:12]}_{run_token()}"
    simulation_file = simulation_file_paths(sim_dir, save_to)[0]
    argv = build_argv(
        tau2_bin,
        domain=args.domain,
        task_ids=task_ids,
        trials=args.trials,
        workers=args.workers,
        agent_llm=args.agent_llm,
        user_llm=args.user_llm,
        save_to=save_to,
    )
    stamp = {
        "policy_sha256": sha256,
        "harness": HARNESS_STAMP,
        "domain": args.domain,
        # Filled in below with the tasks that were actually graded, which with
        # no --task-ids is the simulation file's own task list.  ``tasks_source``
        # says which of the two a reader is looking at, so a stamp naming the
        # whole domain is not read as a stamp naming a chosen handful.
        "tasks": task_ids,
        "tasks_source": "argument" if task_ids else "simulation-file",
        "trials": args.trials,
        "agent_llm": args.agent_llm,
        "user_llm": args.user_llm,
    }
    record: dict[str, Any] = {
        "schema": GRADE_SCHEMA,
        "policy": str(policy),
        "graded_policy": str(graded),
        "policy_sha256": sha256,
        "simulation_file": str(simulation_file),
        "argv": argv,
    }
    if args.dry_run:
        record["status"] = "planned"
        return record

    # tau2 resumes into an existing file of the same name, so a previous
    # round's results would be read as this round's if the command failed.
    clear_simulation_files(sim_dir, save_to)
    result = run_process(
        argv,
        cwd=REPO_ROOT,
        env={"SASY_POLICY_PATH": str(graded)},
        log=out_dir / "grade.log",
    )
    if result.returncode != 0:
        raise GradeError(f"grading exited {result.returncode} for policy {sha256[:12]} (see {out_dir / 'grade.log'})")
    simulation_file = written_simulation_file(sim_dir, save_to)
    record["simulation_file"] = str(simulation_file)
    graded_tasks = task_ids or tasks_in_simulation_file(simulation_file)
    stamp["tasks"] = graded_tasks
    reward = read_reward(simulation_file, task_ids, stamp=stamp)
    problem = completeness_problem(
        reward,
        graded_tasks,
        args.trials,
        tasks_came_from_the_file=not task_ids,
    )
    if problem is not None:
        raise GradeError(f"partial results are not a measurement: {problem} ({simulation_file})")
    record["status"] = "graded"
    record["elapsed_seconds"] = result.elapsed_seconds
    record["log"] = result.log
    record.update(reward)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--policy", required=True, help="the .dl policy to enforce while the benchmark runs")
    parser.add_argument(
        "--task-ids",
        action="append",
        default=[],
        metavar="ID",
        help="a task to grade on; repeat for several. Omit to run the domain's whole task set",
    )
    parser.add_argument("--trials", type=int, default=1, help="simulations per task (default: 1)")
    parser.add_argument("--workers", type=int, default=1, help="tau2 --max-concurrency (default: 1)")
    parser.add_argument("--agent-llm", required=True, help="the model the agent runs on; recorded in the stamp")
    parser.add_argument("--user-llm", required=True, help="the model the user simulator runs on")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help=f"tau2 domain (default: {DEFAULT_DOMAIN})")
    parser.add_argument("--out-dir", required=True, help="where the staged policy, the log and grade.json go")
    parser.add_argument(
        "--tau2-bin", default=DEFAULT_TAU2_BIN, help=f"benchmark launcher (default: {DEFAULT_TAU2_BIN})"
    )
    parser.add_argument("--sim-dir", default=str(DEFAULT_SIM_DIR), help="where tau2 writes its simulation files")
    parser.add_argument("--dry-run", action="store_true", help="stage and plan, but do not run the benchmark")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        record = grade(args)
    except GradeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    out = Path(args.out_dir).expanduser().resolve() / "grade.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {out}")
    if record.get("status") == "graded":
        print(f"  reward {record.get('reward')} over {record.get('sims')} simulations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
