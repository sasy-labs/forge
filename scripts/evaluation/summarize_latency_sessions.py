"""Offline stratified session bootstrap for latency sweep results.

Reads the sweep manifest's canonical child benchmark manifests and SDK
latency.jsonl / server_replay.jsonl files. Maps readable session IDs to
task IDs, bootstraps independently within each task stratum, and computes
pooled request-weighted medians for SDK RTT and server total_us.

Default: 2000 replicates, fixed seed (42), 95% CI with p95/p99 descriptive.

Data integrity validation:
- Rejects malformed JSON records (counts them)
- Rejects non-object JSON records and non-string session IDs
- Requires finite, non-negative numeric timing values (rejects booleans/strings)
- Requires session ID fullmatch against pattern
- Returns nonzero exit for missing coverage, data integrity failures, or
  failed/incomplete cells
- Reports counts for malformed/error/missing/unmapped records

Usage::

    # From a completed latency sweep
    uv run python scripts/evaluation/summarize_latency_sessions.py \\
        outputs/latency-sweep/2026-10-05T14-00-00

    # Or via the sweep runner's --reduce-only
    uv run python scripts/evaluation/run_latency_sweep.py \\
        --reduce-only outputs/latency-sweep/2026-10-05T14-00-00
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Session ID pattern: tau2-task-{task_id}-{random_suffix}
# This pattern is set when SASY_SESSION_READABLE=1
# Uses fullmatch to reject partial matches
SESSION_ID_PATTERN = re.compile(r"tau2-task-(\d+)-[a-zA-Z0-9]+")


@dataclass
class LoadStats:
    """Statistics from loading a latency log file."""
    total_lines: int = 0
    valid_records: int = 0
    malformed_json: int = 0
    non_object_json: int = 0
    missing_session: int = 0
    non_string_session: int = 0
    missing_value: int = 0
    invalid_value: int = 0
    error_records: int = 0

    def has_rejected_records(self) -> bool:
        """True if any records were rejected (excluding explicit errors)."""
        return (
            self.malformed_json > 0
            or self.non_object_json > 0
            or self.missing_session > 0
            or self.non_string_session > 0
            or self.missing_value > 0
            or self.invalid_value > 0
        )


@dataclass
class SessionData:
    """Latency data for one session."""
    session_id: str
    task_id: int | None
    sdk_rtts_us: list[float] = field(default_factory=list)
    server_totals_us: list[float] = field(default_factory=list)


@dataclass
class CellLatencies:
    """Aggregated latency data for one concurrency cell."""
    concurrency: int
    sessions: dict[str, SessionData] = field(default_factory=dict)
    unmapped_sessions: set[str] = field(default_factory=set)
    sdk_stats: LoadStats = field(default_factory=LoadStats)
    server_stats: LoadStats = field(default_factory=LoadStats)


@dataclass
class TrialCoverage:
    """Per-task expected vs actual trial coverage from canonical results.

    Canonical results are the tau2 simulation Results JSON, which tracks
    (task_id, trial) pairs independently of SASY session IDs. This allows
    detecting missing trials even when zero-call sessions leave no SDK trace.

    Limitation: The canonical results do not include SASY session IDs, so
    exact trial-to-session identity cannot be recovered. We can detect count
    deficits but cannot trace which specific SASY session corresponds to which
    canonical trial.
    """
    task_id: int
    expected_trials: int  # from trials_per_task
    actual_trials: int  # unique (task_id, trial) pairs found
    completed_trials: int  # trials with non-infrastructure_error termination
    infra_error_trials: int  # trials with infrastructure_error termination
    duplicate_trials: list[int] = field(default_factory=list)  # trial numbers seen >1


@dataclass
class CanonicalCoverageResult:
    """Result of validating canonical trial coverage for a cell."""
    has_child_manifest: bool = False
    has_sim_path: bool = False
    has_results_file: bool = False
    coverage_by_task: dict[int, TrialCoverage] = field(default_factory=dict)
    missing_trials_total: int = 0
    infra_errors_total: int = 0
    duplicate_trials_total: int = 0
    validation_errors: list[str] = field(default_factory=list)


@dataclass
class BootstrapResult:
    """Bootstrap confidence interval result."""
    median: float
    ci_low: float
    ci_high: float
    p95: float
    p99: float
    n_calls: int
    n_sessions: int
    n_tasks: int
    n_replicates: int
    coverage_warnings: list[str] = field(default_factory=list)
    coverage_errors: list[str] = field(default_factory=list)


def _is_valid_timing(value) -> bool:
    """Check if a timing value is valid: numeric int/float, finite, non-negative, not boolean/string."""
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        return False
    if not isinstance(value, (int, float)):
        return False
    try:
        f = float(value)
        return math.isfinite(f) and f >= 0
    except (ValueError, TypeError):
        return False


def parse_session_task_id(session_id: str) -> int | None:
    """Extract task ID from readable session ID pattern.

    Uses fullmatch to reject partial matches like 'prefix-tau2-task-42-abc'.
    """
    match = SESSION_ID_PATTERN.fullmatch(session_id)
    if match:
        return int(match.group(1))
    return None


def load_sdk_latencies(path: Path) -> tuple[dict[str, list[float]], LoadStats]:
    """Load SDK RTT values grouped by session_id from latency.jsonl.

    Returns (data_by_session, stats) where stats counts all parsing outcomes.
    Invalid records are rejected, not silently skipped.
    """
    by_session: dict[str, list[float]] = defaultdict(list)
    stats = LoadStats()

    if not path.exists():
        return dict(by_session), stats

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        stats.total_lines += 1

        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            stats.malformed_json += 1
            continue

        # Must be a JSON object
        if not isinstance(rec, dict):
            stats.non_object_json += 1
            continue

        # Skip explicit error records
        if "error" in rec:
            stats.error_records += 1
            continue

        session = rec.get("session_id")
        if session is None:
            stats.missing_session += 1
            continue
        if not isinstance(session, str):
            stats.non_string_session += 1
            continue

        rtt = rec.get("rtt_us")
        if rtt is None:
            stats.missing_value += 1
            continue

        if not _is_valid_timing(rtt):
            stats.invalid_value += 1
            continue

        by_session[session].append(float(rtt))
        stats.valid_records += 1

    return dict(by_session), stats


def load_server_latencies(path: Path) -> tuple[dict[str, list[float]], LoadStats]:
    """Load server total_us values grouped by session_id from server_replay.jsonl.

    Returns (data_by_session, stats) where stats counts all parsing outcomes.
    Invalid records are rejected, not silently skipped.
    """
    by_session: dict[str, list[float]] = defaultdict(list)
    stats = LoadStats()

    if not path.exists():
        return dict(by_session), stats

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        stats.total_lines += 1

        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            stats.malformed_json += 1
            continue

        # Must be a JSON object
        if not isinstance(rec, dict):
            stats.non_object_json += 1
            continue

        session = rec.get("session_id")
        if session is None:
            stats.missing_session += 1
            continue
        if not isinstance(session, str):
            stats.non_string_session += 1
            continue

        total = rec.get("total_us")
        if total is None:
            stats.missing_value += 1
            continue

        if not _is_valid_timing(total):
            stats.invalid_value += 1
            continue

        by_session[session].append(float(total))
        stats.valid_records += 1

    return dict(by_session), stats


def load_cell_latencies(
    cell_dir: Path,
    expected_task_ids: set[int],
) -> CellLatencies:
    """Load and validate latency data for one cell.

    Returns a CellLatencies with sessions mapped to tasks and load statistics.
    Sessions with IDs not matching the expected task set are recorded as unmapped.
    """
    sdk_log = cell_dir / "latency.jsonl"
    server_log = cell_dir / "server_replay.jsonl"

    sdk_by_session, sdk_stats = load_sdk_latencies(sdk_log)
    server_by_session, server_stats = load_server_latencies(server_log)

    all_sessions = set(sdk_by_session.keys()) | set(server_by_session.keys())

    result = CellLatencies(
        concurrency=0,  # filled by caller
        sessions={},
        unmapped_sessions=set(),
        sdk_stats=sdk_stats,
        server_stats=server_stats,
    )

    for session_id in all_sessions:
        task_id = parse_session_task_id(session_id)
        if task_id is None:
            result.unmapped_sessions.add(session_id)
            continue
        if task_id not in expected_task_ids:
            result.unmapped_sessions.add(session_id)
            continue

        result.sessions[session_id] = SessionData(
            session_id=session_id,
            task_id=task_id,
            sdk_rtts_us=sdk_by_session.get(session_id, []),
            server_totals_us=server_by_session.get(session_id, []),
        )

    return result


def validate_canonical_coverage(
    cell: dict,
    sweep_manifest_dir: Path,
    expected_task_ids: set[int],
    expected_trials_per_task: int | None,
) -> CanonicalCoverageResult:
    """Validate canonical trial coverage from tau2 Results.

    Parses the child manifest to find the sim_path, then reads the tau2
    Results JSON to verify all expected (task_id, trial) pairs are present
    and completed without infrastructure errors.

    Args:
        cell: Cell entry from sweep manifest
        sweep_manifest_dir: Directory containing the sweep manifest
        expected_task_ids: Set of expected task IDs
        expected_trials_per_task: Expected trials per task (from manifest)

    Returns:
        CanonicalCoverageResult with validation status and per-task coverage
    """
    result = CanonicalCoverageResult()

    if type(expected_trials_per_task) is not int or expected_trials_per_task < 1:
        result.validation_errors.append("missing or invalid trials_per_task in cell")
        return result

    # Resolve child manifest path relative to sweep manifest
    child_manifest_path_str = cell.get("child_manifest")
    if not child_manifest_path_str:
        result.validation_errors.append("no child_manifest in cell")
        return result

    child_manifest_path = sweep_manifest_dir / child_manifest_path_str
    if not child_manifest_path.exists():
        result.validation_errors.append(f"child manifest not found: {child_manifest_path}")
        return result

    result.has_child_manifest = True

    try:
        child_manifest = json.loads(child_manifest_path.read_text())
    except json.JSONDecodeError as e:
        result.validation_errors.append(f"invalid JSON in child manifest: {e}")
        return result

    # Find the measured cell (not warmup, returncode=0)
    measured_cells = [
        c for c in child_manifest.get("cells", [])
        if not c.get("warmup") and c.get("returncode") == 0
    ]
    if len(measured_cells) != 1:
        result.validation_errors.append(
            f"expected 1 measured cell, found {len(measured_cells)}"
        )
        return result

    measured_cell = measured_cells[0]
    sim_path_str = measured_cell.get("sim_path")
    if not sim_path_str:
        result.validation_errors.append("no sim_path in measured cell")
        return result

    result.has_sim_path = True

    # Resolve sim_path relative to child manifest directory
    child_manifest_dir = child_manifest_path.parent
    results_path = child_manifest_dir / sim_path_str
    if not results_path.exists():
        result.validation_errors.append(f"results file not found: {results_path}")
        return result

    result.has_results_file = True

    try:
        results_data = json.loads(results_path.read_text())
    except json.JSONDecodeError as e:
        result.validation_errors.append(f"invalid JSON in results: {e}")
        return result

    # Extract simulations and validate coverage
    simulations = results_data.get("simulations", [])
    if not simulations:
        # Check for dir format - simulation_index in results.json
        sim_index = results_data.get("simulation_index", [])
        if sim_index:
            simulations = sim_index

    if not isinstance(simulations, list):
        result.validation_errors.append("simulations must be a list")
        return result

    # Count trials per task
    seen_trials: dict[int, dict[int, list[str]]] = {}  # task_id -> {trial -> [termination_reasons]}
    for sim in simulations:
        if not isinstance(sim, dict):
            result.validation_errors.append("simulation must be an object")
            continue
        task_id_raw = sim.get("task_id")
        trial = sim.get("trial")
        term_reason = sim.get("termination_reason", "unknown")

        if (type(trial) is not int or not 0 <= trial < expected_trials_per_task
                or type(task_id_raw) not in (str, int)):
            result.validation_errors.append("invalid or unexpected task/trial pair")
            continue
        if not isinstance(term_reason, str) or term_reason in ("", "unknown"):
            result.validation_errors.append("missing termination reason")
            continue

        # task_id may be string or int
        try:
            task_id = int(task_id_raw)
        except (ValueError, TypeError):
            result.validation_errors.append("invalid task ID")
            continue

        if task_id not in expected_task_ids:
            result.validation_errors.append(f"unexpected task ID: {task_id}")
            continue

        if task_id not in seen_trials:
            seen_trials[task_id] = {}
        if trial not in seen_trials[task_id]:
            seen_trials[task_id][trial] = []
        seen_trials[task_id][trial].append(term_reason)

    # Build coverage result per task
    for task_id in expected_task_ids:
        trials_for_task = seen_trials.get(task_id, {})
        actual_trials = len(trials_for_task)
        completed = 0
        infra_errors = 0
        duplicates = []

        for trial_num, reasons in trials_for_task.items():
            if len(reasons) > 1:
                duplicates.append(trial_num)
            for reason in reasons:
                if reason == "infrastructure_error":
                    infra_errors += 1
                else:
                    completed += 1

        coverage = TrialCoverage(
            task_id=task_id,
            expected_trials=expected_trials_per_task,
            actual_trials=actual_trials,
            completed_trials=completed,
            infra_error_trials=infra_errors,
            duplicate_trials=duplicates,
        )
        result.coverage_by_task[task_id] = coverage

        missing = expected_trials_per_task - actual_trials
        if missing > 0:
            result.missing_trials_total += missing
        result.infra_errors_total += infra_errors
        result.duplicate_trials_total += len(duplicates)

    return result


def percentile(values: list[float], p: float) -> float:
    """Compute percentile (p in [0, 1]) using linear interpolation.

    Matches NumPy's default 'linear' method: for n values, compute index
    h = (n-1) * p, then interpolate between floor(h) and ceil(h).

    For [10, 100]: median = 55, p95 = 95.5
    """
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    # Linear interpolation: h = (n-1) * p
    h = (n - 1) * p
    lo = int(math.floor(h))
    hi = int(math.ceil(h))
    if lo == hi:
        return sorted_vals[lo]
    # Interpolate: values[lo] + (values[hi] - values[lo]) * (h - lo)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (h - lo)


def weighted_median_bootstrap(
    sessions_by_task: dict[int, list[SessionData]],
    field: str,
    n_replicates: int,
    seed: int,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Bootstrap the request-weighted median across task strata.

    For each replicate:
    1. Sample sessions with replacement within each task stratum
    2. Pool all calls from sampled sessions
    3. Compute the pooled median

    The confidence interval is the (alpha/2, 1-alpha/2) percentiles
    of the replicate medians.

    Session counts per lane reflect sessions actually contributing to that lane,
    not the union of sessions across both lanes.

    Task and session iteration order is sorted for reproducibility with fixed seed.

    Args:
        sessions_by_task: Sessions grouped by task ID
        field: 'sdk_rtts_us' or 'server_totals_us'
        n_replicates: Number of bootstrap replicates
        seed: Random seed for reproducibility
        alpha: Confidence level (default 0.05 for 95% CI)

    Returns:
        BootstrapResult with median, CI bounds, p95/p99, and coverage info
    """
    rng = random.Random(seed)
    warnings: list[str] = []
    errors: list[str] = []

    # Sort tasks for reproducibility
    sorted_task_ids = sorted(sessions_by_task.keys())

    # Flatten to count totals, counting only sessions with data for THIS lane
    all_values: list[float] = []
    n_sessions_with_data = 0
    n_tasks = len(sessions_by_task)
    empty_tasks = []
    zero_call_sessions = []

    for task_id in sorted_task_ids:
        sessions = sessions_by_task[task_id]
        # Sort sessions by session_id for reproducibility
        sorted_sessions = sorted(sessions, key=lambda s: s.session_id)
        task_values = []
        for s in sorted_sessions:
            vals = getattr(s, field)
            if vals:  # Only count sessions that have data for this field
                n_sessions_with_data += 1
                task_values.extend(vals)
                all_values.extend(vals)
            else:
                zero_call_sessions.append((task_id, s.session_id))

        if not task_values:
            empty_tasks.append(task_id)

    if zero_call_sessions:
        warnings.append(f"{len(zero_call_sessions)} session(s) with zero {field}")

    if empty_tasks:
        errors.append(f"tasks with no {field}: {sorted(empty_tasks)}")

    if not all_values:
        return BootstrapResult(
            median=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
            p95=float("nan"), p99=float("nan"),
            n_calls=0, n_sessions=n_sessions_with_data, n_tasks=n_tasks,
            n_replicates=n_replicates,
            coverage_warnings=warnings,
            coverage_errors=errors + ["no data available"],
        )

    # Compute observed statistics
    observed_median = percentile(all_values, 0.5)
    observed_p95 = percentile(all_values, 0.95)
    observed_p99 = percentile(all_values, 0.99)

    # Bootstrap: stratified sampling within tasks (sorted for reproducibility)
    replicate_medians: list[float] = []

    for _ in range(n_replicates):
        pooled: list[float] = []
        for task_id in sorted_task_ids:
            sessions = sessions_by_task[task_id]
            # Sort and filter to sessions that actually have data for this field
            sorted_sessions = sorted(sessions, key=lambda s: s.session_id)
            sessions_with_data = [s for s in sorted_sessions if getattr(s, field)]
            if not sessions_with_data:
                continue
            # Sample same number of sessions with replacement
            sampled = rng.choices(sessions_with_data, k=len(sessions_with_data))
            for s in sampled:
                pooled.extend(getattr(s, field))
        if pooled:
            replicate_medians.append(percentile(pooled, 0.5))

    if not replicate_medians:
        return BootstrapResult(
            median=observed_median, ci_low=float("nan"), ci_high=float("nan"),
            p95=observed_p95, p99=observed_p99,
            n_calls=len(all_values), n_sessions=n_sessions_with_data, n_tasks=n_tasks,
            n_replicates=0,
            coverage_warnings=warnings,
            coverage_errors=errors + ["bootstrap produced no valid replicates"],
        )

    # Compute CI from replicate distribution
    ci_low = percentile(replicate_medians, alpha / 2)
    ci_high = percentile(replicate_medians, 1 - alpha / 2)

    return BootstrapResult(
        median=observed_median,
        ci_low=ci_low,
        ci_high=ci_high,
        p95=observed_p95,
        p99=observed_p99,
        n_calls=len(all_values),
        n_sessions=n_sessions_with_data,
        n_tasks=n_tasks,
        n_replicates=len(replicate_medians),
        coverage_warnings=warnings,
        coverage_errors=errors,
    )


@dataclass
class CellSummary:
    """Summary statistics for one concurrency cell."""
    concurrency: int
    sdk_bootstrap: BootstrapResult
    server_bootstrap: BootstrapResult
    sessions_by_task: dict[int, int]  # task_id -> session count
    unmapped_count: int
    missing_tasks: list[int]
    sdk_stats: LoadStats
    server_stats: LoadStats
    canonical_coverage: CanonicalCoverageResult | None = None
    trials_per_task: int | None = None
    has_errors: bool = False


def summarize_cell(
    cell: dict,
    run_dir: Path,
    expected_task_ids: set[int],
    n_replicates: int,
    seed: int,
) -> CellSummary:
    """Compute bootstrap statistics for one cell.

    Validates both SDK/server latency coverage and canonical trial coverage
    from the tau2 Results JSON. Per-task expected trial counts come from
    cell["trials_per_task"]; missing counts make coverage incomplete.
    """
    concurrency = cell["concurrency"]
    cell_dir = run_dir / cell.get("out_dir", f"c{concurrency}")
    trials_per_task = cell.get("trials_per_task")

    latencies = load_cell_latencies(cell_dir, expected_task_ids)
    latencies.concurrency = concurrency

    # Group sessions by task
    sessions_by_task: dict[int, list[SessionData]] = defaultdict(list)
    for session in latencies.sessions.values():
        if session.task_id is not None:
            sessions_by_task[session.task_id].append(session)

    # Find tasks with no sessions
    missing_tasks = sorted(expected_task_ids - set(sessions_by_task.keys()))

    # Bootstrap SDK and server latencies
    sdk_boot = weighted_median_bootstrap(
        sessions_by_task, "sdk_rtts_us", n_replicates, seed
    )
    server_boot = weighted_median_bootstrap(
        sessions_by_task, "server_totals_us", n_replicates, seed
    )

    # Each trial must contribute timing coverage independently in both lanes.
    if type(trials_per_task) is int and trials_per_task > 0:
        for field, boot in (("sdk_rtts_us", sdk_boot), ("server_totals_us", server_boot)):
            for task_id in sorted(expected_task_ids):
                actual = sum(bool(getattr(session, field))
                             for session in sessions_by_task.get(task_id, []))
                if actual != trials_per_task:
                    boot.coverage_errors.append(
                        f"task {task_id}: expected {trials_per_task} timing sessions, got {actual}"
                    )

    # Validate canonical trial coverage from tau2 Results
    canonical_coverage = validate_canonical_coverage(
        cell, run_dir, expected_task_ids, trials_per_task
    )

    # Detect errors: any rejected records or missing coverage
    has_errors = (
        latencies.sdk_stats.has_rejected_records()
        or latencies.server_stats.has_rejected_records()
        or latencies.sdk_stats.error_records > 0
        or latencies.server_stats.error_records > 0
        or len(latencies.unmapped_sessions) > 0
        or len(missing_tasks) > 0
        or len(sdk_boot.coverage_errors) > 0
        or len(server_boot.coverage_errors) > 0
        # Canonical coverage errors
        or len(canonical_coverage.validation_errors) > 0
        or canonical_coverage.missing_trials_total > 0
        or canonical_coverage.infra_errors_total > 0
        or canonical_coverage.duplicate_trials_total > 0
    )

    return CellSummary(
        concurrency=concurrency,
        sdk_bootstrap=sdk_boot,
        server_bootstrap=server_boot,
        sessions_by_task={tid: len(sessions) for tid, sessions in sessions_by_task.items()},
        unmapped_count=len(latencies.unmapped_sessions),
        missing_tasks=missing_tasks,
        sdk_stats=latencies.sdk_stats,
        server_stats=latencies.server_stats,
        canonical_coverage=canonical_coverage,
        trials_per_task=trials_per_task,
        has_errors=has_errors,
    )


def format_us(x: float) -> str:
    """Format microseconds for display."""
    if math.isnan(x):
        return "—"
    if x < 1000:
        return f"{x:.0f}us"
    return f"{x/1000:.2f}ms"


def format_ci(boot: BootstrapResult) -> str:
    """Format bootstrap result with CI."""
    if math.isnan(boot.median):
        return "—"
    return f"{format_us(boot.median)} [{format_us(boot.ci_low)}, {format_us(boot.ci_high)}]"


def _load_stats_to_dict(stats: LoadStats) -> dict:
    """Convert LoadStats to dictionary for JSON output."""
    return {
        "total_lines": stats.total_lines,
        "valid_records": stats.valid_records,
        "malformed_json": stats.malformed_json,
        "non_object_json": stats.non_object_json,
        "missing_session": stats.missing_session,
        "non_string_session": stats.non_string_session,
        "missing_value": stats.missing_value,
        "invalid_value": stats.invalid_value,
        "error_records": stats.error_records,
    }


def _canonical_coverage_to_dict(cc: CanonicalCoverageResult) -> dict:
    """Convert CanonicalCoverageResult to dictionary for JSON output."""
    coverage_by_task = {}
    for task_id, tc in cc.coverage_by_task.items():
        coverage_by_task[str(task_id)] = {
            "expected_trials": tc.expected_trials,
            "actual_trials": tc.actual_trials,
            "completed_trials": tc.completed_trials,
            "infra_error_trials": tc.infra_error_trials,
            "duplicate_trials": tc.duplicate_trials,
        }
    return {
        "has_child_manifest": cc.has_child_manifest,
        "has_sim_path": cc.has_sim_path,
        "has_results_file": cc.has_results_file,
        "coverage_by_task": coverage_by_task,
        "missing_trials_total": cc.missing_trials_total,
        "infra_errors_total": cc.infra_errors_total,
        "duplicate_trials_total": cc.duplicate_trials_total,
        "validation_errors": cc.validation_errors,
        "identity_limitation": (
            "Canonical results do not include SASY session IDs; exact trial-to-session "
            "identity cannot be recovered. Count deficits are detected but which specific "
            "SASY session corresponds to which canonical trial is not traceable."
        ),
    }


def summarize_sweep(manifest: dict, run_dir: Path, args: argparse.Namespace | None = None) -> int:
    """Summarize a latency sweep with bootstrap statistics.

    Returns nonzero if there are coverage errors, data integrity failures,
    failed cells, or incomplete coverage.

    Always writes JSON output when the manifest is structurally valid.
    """
    if args is None:
        # Create default args
        class DefaultArgs:
            n_replicates = 2000
            seed = 42
        args = DefaultArgs()

    task_ids = set(manifest.get("task_ids", []))
    if not task_ids:
        print("ERROR: no task_ids in manifest")
        return 1

    manifest_state = manifest.get("state", "unknown")
    planned_concurrencies = set(manifest.get("concurrencies", []))
    cells = manifest.get("cells", [])

    # Check for interrupted/running manifests
    if manifest_state not in ("stopped", "completed"):
        print(f"WARNING: manifest state is '{manifest_state}', run may be incomplete")

    print("\nStratified Bootstrap Summary")
    print(f"{'='*70}")
    print(f"Domain: {manifest.get('domain')}, Mode: {manifest.get('mode')}")
    print(f"Tasks: {sorted(task_ids)} (K={len(task_ids)})")
    print(f"Bootstrap: {args.n_replicates} replicates, seed={args.seed}, 95% CI")
    print(f"Manifest state: {manifest_state}")
    print()

    summaries: list[CellSummary] = []
    failed_cells: list[dict] = []
    present_concurrencies: set[int] = set()

    for cell in sorted(cells, key=lambda c: c["concurrency"]):
        conc = cell["concurrency"]
        present_concurrencies.add(conc)
        rc = cell.get("returncode")
        if rc != 0:
            print(f"c={conc}: FAILED (returncode={rc})")
            failed_cells.append({"concurrency": conc, "returncode": rc})
            continue
        summary = summarize_cell(cell, run_dir, task_ids, args.n_replicates, args.seed)
        summaries.append(summary)

    # Check for missing concurrency levels
    missing_concurrencies = sorted(planned_concurrencies - present_concurrencies)

    # Determine overall error status
    any_errors = (
        len(failed_cells) > 0
        or len(missing_concurrencies) > 0
        or manifest_state not in ("stopped", "completed")
    )

    # Print tables even if some cells failed (for partial results)
    if summaries:
        # Print SDK table
        print("SDK RTT (client-side wall time)")
        print("-" * 70)
        print(f"{'conc':>6} | {'median [95% CI]':^35} | {'p95':>10} | {'p99':>10} | {'n':>6}")
        print("-" * 70)
        for s in summaries:
            b = s.sdk_bootstrap
            print(f"{s.concurrency:>6} | {format_ci(b):^35} | {format_us(b.p95):>10} | {format_us(b.p99):>10} | {b.n_calls:>6}")
        print()

        # Print server table
        print("Server total_us (engine processing time)")
        print("-" * 70)
        print(f"{'conc':>6} | {'median [95% CI]':^35} | {'p95':>10} | {'p99':>10} | {'n':>6}")
        print("-" * 70)
        for s in summaries:
            b = s.server_bootstrap
            print(f"{s.concurrency:>6} | {format_ci(b):^35} | {format_us(b.p95):>10} | {format_us(b.p99):>10} | {b.n_calls:>6}")
        print()
    else:
        print("No valid cells to summarize.")
        print()

    # Coverage / warnings / errors
    print("Data Integrity Report")
    print("-" * 70)

    for s in summaries:
        issues = []

        # Load stats - all rejected records are errors
        if s.sdk_stats.malformed_json > 0:
            issues.append(f"sdk: {s.sdk_stats.malformed_json} malformed JSON")
            any_errors = True
        if s.sdk_stats.non_object_json > 0:
            issues.append(f"sdk: {s.sdk_stats.non_object_json} non-object JSON")
            any_errors = True
        if s.sdk_stats.missing_session > 0:
            issues.append(f"sdk: {s.sdk_stats.missing_session} missing session_id")
            any_errors = True
        if s.sdk_stats.non_string_session > 0:
            issues.append(f"sdk: {s.sdk_stats.non_string_session} non-string session_id")
            any_errors = True
        if s.sdk_stats.missing_value > 0:
            issues.append(f"sdk: {s.sdk_stats.missing_value} missing rtt_us")
            any_errors = True
        if s.sdk_stats.invalid_value > 0:
            issues.append(f"sdk: {s.sdk_stats.invalid_value} invalid rtt_us")
            any_errors = True
        if s.sdk_stats.error_records > 0:
            issues.append(f"sdk: {s.sdk_stats.error_records} explicit errors")
            any_errors = True

        if s.server_stats.malformed_json > 0:
            issues.append(f"server: {s.server_stats.malformed_json} malformed JSON")
            any_errors = True
        if s.server_stats.non_object_json > 0:
            issues.append(f"server: {s.server_stats.non_object_json} non-object JSON")
            any_errors = True
        if s.server_stats.missing_session > 0:
            issues.append(f"server: {s.server_stats.missing_session} missing session_id")
            any_errors = True
        if s.server_stats.non_string_session > 0:
            issues.append(f"server: {s.server_stats.non_string_session} non-string session_id")
            any_errors = True
        if s.server_stats.missing_value > 0:
            issues.append(f"server: {s.server_stats.missing_value} missing total_us")
            any_errors = True
        if s.server_stats.invalid_value > 0:
            issues.append(f"server: {s.server_stats.invalid_value} invalid total_us")
            any_errors = True
        if s.server_stats.error_records > 0:
            issues.append(f"server: {s.server_stats.error_records} explicit errors")
            any_errors = True

        # Coverage errors
        for err in s.sdk_bootstrap.coverage_errors + s.server_bootstrap.coverage_errors:
            issues.append(f"ERROR: {err}")
            any_errors = True

        # Coverage warnings
        for warn in s.sdk_bootstrap.coverage_warnings + s.server_bootstrap.coverage_warnings:
            issues.append(f"WARN: {warn}")

        if s.unmapped_count > 0:
            issues.append(f"{s.unmapped_count} unmapped sessions")
            any_errors = True
        if s.missing_tasks:
            issues.append(f"missing tasks: {s.missing_tasks}")
            any_errors = True

        # Canonical coverage errors
        if s.canonical_coverage:
            cc = s.canonical_coverage
            for err in cc.validation_errors:
                issues.append(f"canonical: {err}")
                any_errors = True
            if cc.missing_trials_total > 0:
                issues.append(f"canonical: {cc.missing_trials_total} missing trials")
                any_errors = True
            if cc.infra_errors_total > 0:
                issues.append(f"canonical: {cc.infra_errors_total} infrastructure_error trials")
                any_errors = True
            if cc.duplicate_trials_total > 0:
                issues.append(f"canonical: {cc.duplicate_trials_total} duplicate trial numbers")
                any_errors = True

        # Print summary line with stats
        sdk_stat = f"sdk: {s.sdk_stats.valid_records}/{s.sdk_stats.total_lines}"
        server_stat = f"server: {s.server_stats.valid_records}/{s.server_stats.total_lines}"
        status = "ERROR" if s.has_errors else "OK"
        print(f"c={s.concurrency}: [{status}] {sdk_stat}, {server_stat}")

        if issues:
            for issue in issues:
                print(f"    {issue}")

    # Report failed and missing cells
    if failed_cells:
        print()
        for fc in failed_cells:
            print(f"c={fc['concurrency']}: [FAILED] returncode={fc['returncode']}")

    if missing_concurrencies:
        print()
        for mc in missing_concurrencies:
            print(f"c={mc}: [NOT_RUN] cell not in manifest")

    print()

    # Always write structured output when manifest is structurally valid
    output = {
        "kind": "latency-sweep-summary",
        "version": 3,
        "domain": manifest.get("domain"),
        "mode": manifest.get("mode"),
        "task_ids": sorted(task_ids),
        "planned_concurrencies": sorted(planned_concurrencies),
        "n_replicates": args.n_replicates,
        "seed": args.seed,
        "manifest_state": manifest_state,
        "has_errors": any_errors,
        "failed_cells": failed_cells,
        "missing_concurrencies": missing_concurrencies,
        "cells": [],
    }

    for s in summaries:
        output["cells"].append({
            "concurrency": s.concurrency,
            "has_errors": s.has_errors,
            "sdk": {
                "median_us": s.sdk_bootstrap.median if not math.isnan(s.sdk_bootstrap.median) else None,
                "ci_low_us": s.sdk_bootstrap.ci_low if not math.isnan(s.sdk_bootstrap.ci_low) else None,
                "ci_high_us": s.sdk_bootstrap.ci_high if not math.isnan(s.sdk_bootstrap.ci_high) else None,
                "p95_us": s.sdk_bootstrap.p95 if not math.isnan(s.sdk_bootstrap.p95) else None,
                "p99_us": s.sdk_bootstrap.p99 if not math.isnan(s.sdk_bootstrap.p99) else None,
                "n_calls": s.sdk_bootstrap.n_calls,
                "n_sessions": s.sdk_bootstrap.n_sessions,
                "load_stats": _load_stats_to_dict(s.sdk_stats),
                "coverage_warnings": s.sdk_bootstrap.coverage_warnings,
                "coverage_errors": s.sdk_bootstrap.coverage_errors,
            },
            "server": {
                "median_us": s.server_bootstrap.median if not math.isnan(s.server_bootstrap.median) else None,
                "ci_low_us": s.server_bootstrap.ci_low if not math.isnan(s.server_bootstrap.ci_low) else None,
                "ci_high_us": s.server_bootstrap.ci_high if not math.isnan(s.server_bootstrap.ci_high) else None,
                "p95_us": s.server_bootstrap.p95 if not math.isnan(s.server_bootstrap.p95) else None,
                "p99_us": s.server_bootstrap.p99 if not math.isnan(s.server_bootstrap.p99) else None,
                "n_calls": s.server_bootstrap.n_calls,
                "n_sessions": s.server_bootstrap.n_sessions,
                "load_stats": _load_stats_to_dict(s.server_stats),
                "coverage_warnings": s.server_bootstrap.coverage_warnings,
                "coverage_errors": s.server_bootstrap.coverage_errors,
            },
            "sessions_by_task": s.sessions_by_task,
            "unmapped_count": s.unmapped_count,
            "missing_tasks": s.missing_tasks,
            "trials_per_task": s.trials_per_task,
            "canonical_coverage": _canonical_coverage_to_dict(s.canonical_coverage) if s.canonical_coverage else None,
        })

    summary_path = run_dir / "bootstrap_summary.json"
    summary_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Summary: {summary_path}")

    if any_errors:
        print("\nERROR: incomplete coverage or data integrity issues (see report above)")
        return 1

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="Latency sweep run directory containing manifest.json",
    )
    parser.add_argument(
        "--n-replicates", type=int, default=2000,
        help="Number of bootstrap replicates (default: 2000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for bootstrap (default: 42).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: no manifest.json in {run_dir}")
        return 1

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON in manifest: {e}")
        return 1

    if manifest.get("kind") != "latency-sweep":
        print(f"ERROR: not a latency-sweep manifest (kind={manifest.get('kind')})")
        return 1

    return summarize_sweep(manifest, run_dir, args)


if __name__ == "__main__":
    sys.exit(main())
