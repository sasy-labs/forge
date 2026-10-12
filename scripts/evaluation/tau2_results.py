"""Where a tau2 run's results file is, and how each simulation ended.

tau2 1.0 writes a run named by ``--save-to <name>`` to
``data/simulations/<name>/results.json`` (``tau2.runner.batch.run_domain``).
Earlier releases wrote the flat file ``data/simulations/<name>.json``, and runs
archived before the upgrade are still in that layout, so a reader is given
either path and takes whichever exists, the directory layout first.

A simulation's ``termination_reason`` is one of ``TERMINATION_REASONS``, the
values of ``tau2.data_model.simulation.TerminationReason`` in 1.0.1.  1.0 added
``timeout``, ``infrastructure_error`` (the run never happened -- an API
disconnect after every retry; such a simulation carries no reward),
``context_window_exceeded`` and ``unexpected_error``.  Summaries count every
reason they see, a value outside this tuple included, instead of leaving any
out.

``run_benchmark.py`` passes tau2 an absolute ``--save-to``.  tau2 builds the run
directory as ``DATA_DIR / "simulations" / <save_to>``, and joining an absolute
path onto a ``pathlib.Path`` replaces everything before it, so the run lands at
``<save_to>/results.json`` -- inside the benchmark's own output directory, not
in the vendored ``data/simulations``.  The manifest records that path both
absolute (``sim_path``) and relative to the manifest (``sim_relpath``);
:func:`cell_results_file` reads either, so a run directory that was moved or
copied elsewhere still resolves.

pass^k is tau2's measure of consistency: for a task run ``n`` times with ``c``
successes, the chance that ``k`` trials drawn without replacement all succeed is
``C(c, k) / C(n, k)``; pass^k is that chance averaged over tasks.  pass^1 is the
plain success rate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from math import comb
from pathlib import Path
from typing import Any

RESULTS_NAME = "results.json"
# The LLM-judge mode's per-call log, inside the run directory.
JUDGE_LOG_NAME = "judge_log.jsonl"

TERMINATION_REASONS = (
    "user_stop",
    "agent_stop",
    "max_steps",
    "timeout",
    "too_many_errors",
    "agent_error",
    "user_error",
    "infrastructure_error",
    "context_window_exceeded",
    "unexpected_error",
)
# The two ways a conversation ends on its own; every other reason is the run
# being cut short, and a summary says how often that happened.
NORMAL_TERMINATIONS = frozenset({"user_stop", "agent_stop"})


def run_base(path: Path) -> Path:
    """``data/simulations/<name>`` for a results path in either layout."""
    return path.parent if path.name == RESULTS_NAME else path.with_suffix("")


def results_candidates(path: Path) -> tuple[Path, Path]:
    """The directory-layout file and the flat file for the same run, in that order."""
    base = run_base(path)
    return base / RESULTS_NAME, base.with_name(base.name + ".json")


def resolve_results_file(path: Path) -> Path:
    """The results file that exists for this run, the directory layout first.

    When neither exists, the directory-layout path is returned, which is where
    tau2 1.0 would have written it.
    """
    current, legacy = results_candidates(path)
    if current.is_file():
        return current
    if legacy.is_file():
        return legacy
    return current


def results_file_for(sim_dir: Path, save_to: str) -> Path:
    """Where ``tau2 run --save-to <save_to>`` wrote its results under ``sim_dir``."""
    return resolve_results_file(sim_dir / save_to / RESULTS_NAME)


def termination_counts(simulations: Iterable[Any]) -> dict[str, int]:
    """How many simulations ended each way; a missing reason counts as ``"missing"``."""
    counts: Counter[str] = Counter()
    for simulation in simulations:
        reason = simulation.get("termination_reason") if isinstance(simulation, dict) else None
        counts[str(reason) if reason else "missing"] += 1
    return dict(sorted(counts.items()))


def abnormal_terminations(counts: dict[str, int]) -> dict[str, int]:
    """The counts for every reason other than the agent or user ending the conversation."""
    return {reason: n for reason, n in counts.items() if reason not in NORMAL_TERMINATIONS}


def cell_results_file(cell: dict, manifest_dir: Path | None = None, *, run_dir: str | None = None) -> Path | None:
    """The results file a manifest cell names, wherever the run now is.

    Paths can be relative (new format) or absolute (legacy). Relative paths are
    resolved against ``manifest_dir``; absolute paths from moved runs are rebased
    using ``run_dir``.

    For moved/copied runs, local files are preferred over the original absolute
    path even when both exist. This ensures operations use the selected run's
    artifacts, not artifacts from the original location.

    Returns the first existing file, else the recorded path, else ``None``.
    """
    sim_path = cell.get("sim_path")
    sim_relpath = cell.get("sim_relpath")

    # 1. Try relative sim_path (new format)
    if sim_path and manifest_dir is not None:
        if not sim_path.startswith("/"):
            candidate = resolve_results_file(manifest_dir / sim_path)
            if candidate.is_file():
                return candidate

    # 2. Try sim_relpath (legacy dual-field format)
    if sim_relpath and manifest_dir is not None:
        candidate = resolve_results_file(manifest_dir / sim_relpath)
        if candidate.is_file():
            return candidate

    if sim_path:
        recorded = resolve_results_file(Path(sim_path))

        # 3. For absolute paths, try rebasing to local FIRST (before original)
        # This ensures moved/copied runs use their local artifacts
        if manifest_dir is not None and run_dir:
            original = Path(sim_path)
            if original.is_absolute():
                try:
                    rel = original.relative_to(Path(run_dir))
                    moved = resolve_results_file(manifest_dir / rel)
                    if moved.is_file():
                        return moved
                except ValueError:
                    pass

        # 4. Try basename fallback for historical layout cases
        if manifest_dir is not None:
            moved = resolve_results_file(manifest_dir / run_base(recorded).name / RESULTS_NAME)
            if moved.is_file():
                return moved

        # 5. Fall back to original absolute path if it exists
        if recorded.is_file():
            return recorded

        return recorded

    if sim_relpath and manifest_dir is not None:
        return resolve_results_file(manifest_dir / sim_relpath)

    return None


def judge_log_for(results_path: Path) -> Path | None:
    """The LLM-judge log for a run, or ``None`` when there is none.

    ``run_benchmark.py`` writes it inside the run directory
    (``<run>/judge_log.jsonl``); runs from before that kept it beside the run
    as ``<run>.judge_log.jsonl``.
    """
    base = run_base(results_path)
    for candidate in (base / JUDGE_LOG_NAME, base.with_name(base.name + ".judge_log.jsonl")):
        if candidate.is_file():
            return candidate
    return None


def is_success(reward: float | None) -> bool:
    """tau2 counts a trial as a success when its reward is 1 (within rounding)."""
    return reward is not None and reward >= 1.0 - 1e-6


def pass_hat_k(counts: Iterable[tuple[int, int]], k: int) -> float | None:
    """tau2's pass^k over tasks given as ``(successes, trials)`` pairs.

    A task with fewer than ``k`` trials has no pass^k and is left out; ``None``
    when no task has ``k`` trials.
    """
    if k < 1:
        raise ValueError(f"pass^k needs k >= 1, got {k}")
    values = [comb(c, k) / comb(n, k) for c, n in counts if n >= k]
    return sum(values) / len(values) if values else None
