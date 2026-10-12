"""Summarize AgentDojo benchmark runs into paper-ready tables.

Reads per-run JSON results from an AgentDojo runs directory and computes
metrics matching the paper's Table format:

- Clean Utility: % of no-attack runs with utility=True
- Utility Under Attack: % of attacked runs with utility=True
- ASR (AgentDojo): % of attacked runs with security=True (attack success)
- ASR (Effect): Same population, correcting fully policy-refused attacker-goal attempts

Metrics are computed over runs matched across all arms and sweeps, using
population standard deviation (matching make_table.py). Single sweeps are
marked, as one number is not a mean.

Usage::

    uv run python scripts/evaluation/summarize_agentdojo.py \\
        results/<ts>/manifest.json

    # From a run directory containing manifest.json
    uv run python scripts/evaluation/summarize_agentdojo.py \\
        results/<ts>

    # Or directly from a runs directory
    uv run python scripts/evaluation/summarize_agentdojo.py \\
        --logdir results/<ts>/agentdojo/runs

    # Output to files
    uv run python scripts/evaluation/summarize_agentdojo.py <manifest> \\
        --md results/<ts>/agentdojo.md \\
        --json results/<ts>/agentdojo.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentdojo_effect import METHOD, effect_success

# Row definitions: (label, tag without repeat, tag with repeat)
# Tags must exactly match pipeline directory naming.
ROWS = [
    ("No runtime defense", "N-none", "N-none-rep"),
    ("Tool-call filter", "N-tf", "N-tf-rep"),
    ("FORGE (FIDES IFC Policy)", "P-fides", "P-fides-rep"),
    ("FORGE (provenance policy)", "C-w2-norep", "C-w2"),
]

# Refusal prefix in tool error messages
REFUSAL_PREFIX = "This call was refused by the security policy."


@dataclass
class ArmAggregate:
    """Aggregated metrics for one arm (tag) across its sweeps."""

    label: str
    tag: str
    n_sweeps: int = 0
    # Per-sweep metrics (for computing mean/sd); None if subgroup absent
    clean_utility: list[float | None] = field(default_factory=list)
    utility_attack: list[float | None] = field(default_factory=list)
    asr_agentdojo: list[float | None] = field(default_factory=list)
    asr_effect: list[float | None] = field(default_factory=list)
    # Raw counts (summed across sweeps)
    n_clean: int = 0
    n_attack: int = 0
    n_effect: int = 0
    n_errors: int = 0
    n_refusals: int = 0  # runs with at least one policy refusal
    n_refusal_calls: int = 0  # total refused tool calls
    # Runtime samples (pooled from all matched runs)
    durations: list[float] = field(default_factory=list)
    # Duration coverage
    n_total_runs: int = 0
    n_invalid_durations: int = 0


@dataclass
class SweepData:
    """A single sweep: {run_key: record}."""

    name: str
    runs: dict[tuple, dict] = field(default_factory=dict)
    unreadable: int = 0
    duplicates: int = 0
    # Schema validation failures
    invalid_schema: int = 0
    # Conflicting duplicates (same key, different scores)
    conflicting_duplicates: int = 0
    # Versions found in this sweep
    versions: set[str] = field(default_factory=set)


def _run_key(rec: dict) -> tuple:
    """Unique key for a run: (suite, user_task, attack_type, injection_task)."""
    return (
        rec["suite_name"],
        rec["user_task_id"],
        rec.get("attack_type"),
        rec.get("injection_task_id"),
    )


def _is_solvability_probe(rec: dict) -> bool:
    """AgentDojo's injection-solvability probes are not user tasks."""
    return (rec.get("user_task_id") or "").startswith("injection_task_")


def _validate_record(rec: Any) -> tuple[bool, str]:
    """Validate a record has required schema for SCORING.

    Returns (valid, reason). A scorable record must:
    - Be a dict (not null, scalar, array)
    - Have string suite_name and user_task_id
    - Have actual boolean utility and security scores (both required)
    - Have valid messages array if present

    Error records with boolean scores are valid (the benchmark may have
    logged an error AND scored the run). Error records with missing/null
    scores are invalid for scoring - they are noise, not evidence.
    """
    if not isinstance(rec, dict):
        return False, "not a dict"

    # Required identity fields must be strings
    suite = rec.get("suite_name")
    user_task = rec.get("user_task_id")
    if not isinstance(suite, str) or not suite:
        return False, "missing/invalid suite_name"
    if not isinstance(user_task, str) or not user_task:
        return False, "missing/invalid user_task_id"

    # Score fields MUST be actual booleans for scoring.
    # Missing/null scores are invalid - they are noise, not 0% evidence.
    utility = rec.get("utility")
    security = rec.get("security")

    if not isinstance(utility, bool):
        return False, f"utility not boolean (got {type(utility).__name__})"
    if not isinstance(security, bool):
        return False, f"security not boolean (got {type(security).__name__})"

    # Optional fields: attack_type, injection_task_id must be strings or null
    attack = rec.get("attack_type")
    if attack is not None and not isinstance(attack, str):
        return False, "attack_type not string"
    injection = rec.get("injection_task_id")
    if injection is not None and not isinstance(injection, str):
        return False, "injection_task_id not string"

    # Validate messages array if present
    messages = rec.get("messages")
    if messages is not None:
        if not isinstance(messages, list):
            return False, "messages not array"
        for i, m in enumerate(messages):
            if not isinstance(m, dict):
                return False, f"message[{i}] not dict"

    return True, ""


def _records_conflict(a: dict, b: dict) -> bool:
    """Check if two records with the same key have conflicting scores."""
    return (
        a.get("utility") != b.get("utility") or
        a.get("security") != b.get("security")
    )


def load_sweeps(
    logdir: Path, tag: str, model: str
) -> list[SweepData]:
    """Load all sweeps for one arm from a runs directory.

    A sweep directory name matches: {model}-{tag}-r{N}[-description]
    Only loads directories matching the exact model and tag.
    The -description suffix is optional (accepts -rN at end).

    Validates record schema strictly:
    - Records must be dicts with string identity fields
    - Score booleans must be actual booleans
    - Messages must be valid arrays of dicts
    - Conflicting duplicates are warned about

    Returns ALL matched directories including empty/invalid ones so
    diagnostics can be reported. Callers must check sweep.runs.
    """
    out = []
    # Pattern: model-tag-rN or model-tag-rN-description
    # (?:-|$) accepts -rN at end (no description) or -rN-description
    pattern = re.compile(
        rf"^{re.escape(model)}-{re.escape(tag)}-r(\d+)(?:-|$)"
    )

    for d in sorted(logdir.iterdir()):
        if not d.is_dir():
            continue
        if not pattern.match(d.name):
            continue

        sweep = SweepData(name=d.name)
        # Sort paths for deterministic "kept first" behavior
        for p in sorted(d.rglob("*.json")):
            try:
                text = p.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                sweep.unreadable += 1
                continue

            try:
                rec = json.loads(text)
            except json.JSONDecodeError:
                sweep.unreadable += 1
                continue

            # Validate schema
            valid, reason = _validate_record(rec)
            if not valid:
                sweep.invalid_schema += 1
                continue

            if _is_solvability_probe(rec):
                continue  # Skip solvability probes

            key = _run_key(rec)
            if key in sweep.runs:
                existing = sweep.runs[key]
                if _records_conflict(existing, rec):
                    sweep.conflicting_duplicates += 1
                else:
                    sweep.duplicates += 1
                continue  # Keep first occurrence (deterministic via sorted paths)

            # Track version
            v = rec.get("benchmark_version")
            if v and isinstance(v, str):
                sweep.versions.add(v)

            sweep.runs[key] = rec

        # Always include the sweep - even empty ones - so diagnostics
        # can be reported. Callers check sweep.runs for actual data.
        out.append(sweep)

    return out


def matched_keys(all_arms: dict[str, list[SweepData]], attack: str) -> set[tuple]:
    """Find runs present in EVERY sweep of every arm.

    Filters to only clean runs (no injection) and runs with the selected attack.
    This ensures metrics compare the same population across arms.
    """
    keys = None
    for sweeps in all_arms.values():
        for sweep in sweeps:
            # Filter to relevant runs
            relevant = {
                k for k, v in sweep.runs.items()
                if not v.get("injection_task_id") or v.get("attack_type") == attack
            }
            keys = relevant if keys is None else (keys & relevant)
    return keys or set()


@dataclass
class MetricsResult:
    """Metrics for one sweep. None values indicate unavailable subgroups."""
    clean_utility: float | None
    utility_attack: float | None
    asr_agentdojo: float | None
    asr_effect: float | None
    n_clean: int
    n_attack: int
    n_effect: int


def compute_metrics(
    runs: dict[tuple, dict], attack: str
) -> MetricsResult:
    """Compute metrics for a single sweep's runs.

    Returns MetricsResult with None for metrics where the denominator is zero.
    This ensures we don't fabricate 0% for absent data.
    """
    clean = [r for r in runs.values() if not r.get("injection_task_id")]
    atk = [
        r for r in runs.values()
        if r.get("injection_task_id") and r.get("attack_type") == attack
    ]

    n_clean = len(clean)
    n_atk = len(atk)
    n_effect = n_atk

    # Return None when denominator is zero - don't fabricate 0%
    clean_util = (100 * sum(bool(r.get("utility")) for r in clean) / n_clean
                  if n_clean > 0 else None)
    util_atk = (100 * sum(bool(r.get("utility")) for r in atk) / n_atk
                if n_atk > 0 else None)
    asr_dojo = (100 * sum(bool(r.get("security")) for r in atk) / n_atk
                if n_atk > 0 else None)
    asr_effect = (100 * sum(effect_success(r) for r in atk) / n_effect
                  if n_effect > 0 else None)

    return MetricsResult(
        clean_utility=clean_util,
        utility_attack=util_atk,
        asr_agentdojo=asr_dojo,
        asr_effect=asr_effect,
        n_clean=n_clean,
        n_attack=n_atk,
        n_effect=n_effect,
    )


def count_refusals(runs: dict[tuple, dict]) -> tuple[int, int]:
    """Count runs with refusals and total refused calls."""
    n_runs_with_refusal = 0
    n_refusal_calls = 0

    for r in runs.values():
        messages = r.get("messages") or []
        run_refusals = 0
        for m in messages:
            if m.get("role") == "tool":
                error = m.get("error")
                if error is not None and isinstance(error, str):
                    if error.startswith(REFUSAL_PREFIX):
                        run_refusals += 1
        if run_refusals > 0:
            n_runs_with_refusal += 1
            n_refusal_calls += run_refusals

    return n_runs_with_refusal, n_refusal_calls


def count_errors(runs: dict[tuple, dict]) -> int:
    """Count runs with non-null error field."""
    return sum(1 for r in runs.values() if r.get("error") is not None)


@dataclass
class DurationResult:
    """Duration collection result with coverage tracking."""
    valid: list[float]
    n_total: int
    n_missing: int
    n_invalid: int


def collect_durations(runs: dict[tuple, dict]) -> DurationResult:
    """Collect valid duration values from runs with coverage tracking.

    Invalid durations include: non-numeric types, booleans, non-finite values,
    negative values.
    """
    valid = []
    n_missing = 0
    n_invalid = 0

    for r in runs.values():
        d = r.get("duration")
        if d is None:
            n_missing += 1
            continue

        # Reject booleans explicitly (bool is a subclass of int in Python)
        if isinstance(d, bool):
            n_invalid += 1
            continue

        try:
            d_float = float(d)
            if math.isfinite(d_float) and d_float >= 0:
                valid.append(d_float)
            else:
                n_invalid += 1
        except (TypeError, ValueError):
            n_invalid += 1

    return DurationResult(
        valid=valid,
        n_total=len(runs),
        n_missing=n_missing,
        n_invalid=n_invalid,
    )


def aggregate_arm(
    label: str,
    tag: str,
    sweeps: list[SweepData],
    keys: set[tuple],
    attack: str,
) -> ArmAggregate:
    """Aggregate metrics for one arm across its sweeps.

    Preserves None values for unavailable metrics (zero denominator).
    """
    agg = ArmAggregate(label=label, tag=tag, n_sweeps=len(sweeps))

    for sweep in sweeps:
        # Filter to matched keys
        matched = {k: v for k, v in sweep.runs.items() if k in keys}
        if not matched:
            continue

        metrics = compute_metrics(matched, attack)
        agg.clean_utility.append(metrics.clean_utility)
        agg.utility_attack.append(metrics.utility_attack)
        agg.asr_agentdojo.append(metrics.asr_agentdojo)
        agg.asr_effect.append(metrics.asr_effect)
        agg.n_clean += metrics.n_clean
        agg.n_attack += metrics.n_attack
        agg.n_effect += metrics.n_effect

        n_ref, n_ref_calls = count_refusals(matched)
        agg.n_refusals += n_ref
        agg.n_refusal_calls += n_ref_calls
        agg.n_errors += count_errors(matched)

        dur_result = collect_durations(matched)
        agg.durations.extend(dur_result.valid)
        agg.n_total_runs += dur_result.n_total
        agg.n_invalid_durations += dur_result.n_invalid + dur_result.n_missing

    return agg


def _median(samples: list[float]) -> float | None:
    """Median of samples, or None if empty."""
    if not samples:
        return None
    s = sorted(samples)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _pstdev(values: list[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def _valid_values(values: list[float | None]) -> list[float]:
    """Filter to non-None values."""
    return [v for v in values if v is not None]


def _fmt_cell(values: list[float | None], single_marker: str = "†") -> str:
    """Format metric cell: mean ± sd, or mark single sweep.

    Filters out None values. Returns --- if all values are unavailable.
    """
    valid = _valid_values(values)
    if not valid:
        return "---"
    if len(valid) == 1:
        return f"{valid[0]:.1f}{single_marker}"
    return f"{statistics.mean(valid):.1f} ± {_pstdev(valid):.2f}"


def format_markdown(
    arms_no_rep: list[ArmAggregate],
    arms_with_rep: list[ArmAggregate],
    attack: str,
    model: str,
    n_sweeps: int,
    sweep_counts: dict[str, int],
    keys: set[tuple],
    warnings: list[str],
    domains: set[str],
) -> str:
    """Generate markdown table output with correct column counts."""
    lines = []

    lines.append(f"## AgentDojo Results: {model} / {attack}")
    lines.append("")

    if warnings:
        for w in warnings:
            lines.append(f"**Warning:** {w}")
        lines.append("")

    # Count denominators
    n_clean = len({k for k in keys if k[3] is None})
    n_atk = len({k for k in keys if k[3] is not None})
    domains_str = ", ".join(sorted(domains)) if domains else "none"
    lines.append(f"Domains: {domains_str}")
    lines.append(f"Matched runs: {n_clean} clean + {n_atk} attacked per sweep")
    lines.append(f"Sweeps: {n_sweeps}")
    lines.append(METHOD)
    lines.append("")

    # Two separate tables for clarity: Without Repeat, With Repeat
    lines.append("### Without Repeat User Prompt")
    lines.append("")
    lines.append("| Configuration | Clean Utility | Util Under Attack | ASR (AgentDojo) | ASR (Effect) |")
    lines.append("|---------------|---------------|-------------------|-----------------|-------------|")

    for no_rep in arms_no_rep:
        if not no_rep.n_total_runs:
            continue
        cells = [
            no_rep.label,
            _fmt_cell(no_rep.clean_utility),
            _fmt_cell(no_rep.utility_attack),
            _fmt_cell(no_rep.asr_agentdojo),
            _fmt_cell(no_rep.asr_effect),
        ]
        sweep_n = sweep_counts.get(no_rep.tag, 0)
        if sweep_n != n_sweeps:
            cells[0] += f" ({sweep_n} sweeps)"
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("### With Repeat User Prompt")
    lines.append("")
    lines.append("| Configuration | Clean Utility | Util Under Attack | ASR (AgentDojo) | ASR (Effect) |")
    lines.append("|---------------|---------------|-------------------|-----------------|-------------|")

    for with_rep in arms_with_rep:
        if not with_rep.n_total_runs:
            continue
        cells = [
            with_rep.label,
            _fmt_cell(with_rep.clean_utility),
            _fmt_cell(with_rep.utility_attack),
            _fmt_cell(with_rep.asr_agentdojo),
            _fmt_cell(with_rep.asr_effect),
        ]
        sweep_n = sweep_counts.get(with_rep.tag, 0)
        if sweep_n != n_sweeps:
            cells[0] += f" ({sweep_n} sweeps)"
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("† Single sweep; no spread available.")
    lines.append("")

    # Supplementary stats with runtime coverage
    lines.append("### Supplementary Statistics")
    lines.append("")
    lines.append("| Arm | Sweeps | Errors | Runs w/Refusal | Refusal Calls | "
                 "Runtime Coverage | Runtime Mean | Runtime Median | Runtime Min | Runtime Max |")
    lines.append("|-----|--------|--------|----------------|---------------|"
                 "-----------------|----------------|--------------|-------------|-------------|")

    for agg in arms_no_rep + arms_with_rep:
        if not agg.n_total_runs:
            continue
        med = _median(agg.durations)
        mean = statistics.mean(agg.durations) if agg.durations else None
        mi = min(agg.durations) if agg.durations else None
        ma = max(agg.durations) if agg.durations else None
        med_str = f"{med:.2f}s" if med is not None else "---"
        mean_str = f"{mean:.2f}s" if mean is not None else "---"
        mi_str = f"{mi:.2f}s" if mi is not None else "---"
        ma_str = f"{ma:.2f}s" if ma is not None else "---"
        # Runtime coverage: valid / total
        coverage_str = f"{len(agg.durations)}/{agg.n_total_runs}"
        if agg.n_invalid_durations > 0:
            coverage_str += f" ({agg.n_invalid_durations} invalid)"
        lines.append(
            f"| {agg.tag} | {agg.n_sweeps} | {agg.n_errors} | {agg.n_refusals} | "
            f"{agg.n_refusal_calls} | {coverage_str} | "
            f"{mean_str} | {med_str} | {mi_str} | {ma_str} |"
        )

    lines.append("")
    lines.append("**Note:** Cost data is not recorded in run results; cost is unavailable.")

    return "\n".join(lines)


def format_json(
    arms_no_rep: list[ArmAggregate],
    arms_with_rep: list[ArmAggregate],
    attack: str,
    model: str,
    n_sweeps: int,
    sweep_counts: dict[str, int],
    keys: set[tuple],
    warnings: list[str],
    versions: set[str],
    domains: set[str],
) -> dict[str, Any]:
    """Generate machine-readable JSON output."""
    n_clean = len({k for k in keys if k[3] is None})
    n_atk = len({k for k in keys if k[3] is not None})
    n_effect = n_atk

    def metric_summary(values: list[float | None]) -> dict[str, Any]:
        """Generate metric summary handling None values."""
        valid = _valid_values(values)
        return {
            "values": values,  # Preserve None entries for transparency
            "valid_count": len(valid),
            "mean": statistics.mean(valid) if valid else None,
            "pstdev": _pstdev(valid) if len(valid) > 1 else None,
        }

    def arm_summary(agg: ArmAggregate) -> dict[str, Any]:
        durations = agg.durations
        return {
            "label": agg.label,
            "tag": agg.tag,
            "n_sweeps": agg.n_sweeps,
            "metrics": {
                "clean_utility": metric_summary(agg.clean_utility),
                "utility_attack": metric_summary(agg.utility_attack),
                "asr_agentdojo": metric_summary(agg.asr_agentdojo),
                "asr_effect": metric_summary(agg.asr_effect),
            },
            "counts": {
                "n_clean": agg.n_clean,
                "n_attack": agg.n_attack,
                "n_effect": agg.n_effect,
                "n_errors": agg.n_errors,
                "n_refusals": agg.n_refusals,
                "n_refusal_calls": agg.n_refusal_calls,
            },
            "runtime": {
                "valid_count": len(durations),
                "total_runs": agg.n_total_runs,
                "invalid_count": agg.n_invalid_durations,
                "median": _median(durations),
                "mean": statistics.mean(durations) if durations else None,
                "min": min(durations) if durations else None,
                "max": max(durations) if durations else None,
            },
            "cost": {
                "available": False,
                "note": "Cost data is not recorded in run results",
            },
        }

    return {
        "model": model,
        "attack": attack,
        "versions": sorted(versions) if versions else None,
        "domains": sorted(domains),
        "n_sweeps": n_sweeps,
        "sweep_counts_by_arm": sweep_counts,
        "effect_asr_method": METHOD,
        "matched_runs": {
            "n_clean": n_clean,
            "n_attack": n_atk,
            "n_effect": n_effect,
        },
        "warnings": warnings,
        "arms_without_repeat": [arm_summary(a) for a in arms_no_rep],
        "arms_with_repeat": [arm_summary(a) for a in arms_with_rep],
    }


def resolve_logdir(
    args: argparse.Namespace,
    manifest: dict | None,
    manifest_path: Path | None,
) -> Path | None:
    """Resolve the runs directory from args or manifest.

    Priority:
    1. Explicit --logdir flag
    2. Manifest-local agentdojo/runs (conventional relocated path)
    3. Saved path from manifest (absolute or relative to manifest.parent)
    """
    # Explicit --logdir takes precedence
    if args.logdir:
        return args.logdir

    # If we have a manifest path, try local conventional path first
    if manifest_path:
        # Prefer manifest-local agentdojo/runs
        local_runs = manifest_path.parent / "agentdojo" / "runs"
        if local_runs.is_dir():
            return local_runs

    # Then try manifest's saved path
    if manifest:
        agentdojo = manifest.get("agentdojo")
        if isinstance(agentdojo, dict):
            logdir_str = agentdojo.get("logdir")
            if logdir_str and isinstance(logdir_str, str):
                logdir = Path(logdir_str)
                # Try absolute path first
                if logdir.is_absolute() and logdir.is_dir():
                    return logdir
                # Resolve relative path against manifest.parent
                if manifest_path and not logdir.is_absolute():
                    resolved = manifest_path.parent / logdir
                    if resolved.is_dir():
                        return resolved

    return None


def collect_versions_from_keys(
    sweeps_by_arm: dict[str, list[SweepData]],
    keys: set[tuple],
) -> set[str]:
    """Collect benchmark versions from SELECTED runs only.

    Only collects versions from runs that are:
    - In the matched key set (clean + selected attack)
    - Actually used for scoring

    This prevents unrelated attacks from contaminating version detection.
    """
    versions: set[str] = set()
    for sweeps in sweeps_by_arm.values():
        for sweep in sweeps:
            for key, rec in sweep.runs.items():
                if key in keys:
                    v = rec.get("benchmark_version")
                    if v and isinstance(v, str):
                        versions.add(v)
    return versions


@dataclass
class VersionCheck:
    """Result of version validation."""
    versions: set[str]
    warnings: list[str]
    errors: list[str]


def check_version_issues(
    sweeps_by_arm: dict[str, list[SweepData]],
    keys: set[tuple],
    manifest_version: str | None,
) -> VersionCheck:
    """Check version consistency of selected runs.

    Returns errors (not warnings) for:
    - Multiple known benchmark versions mixed -> nonzero exit
    - Manifest declares version that disagrees with selected -> nonzero exit

    Absent versions are not treated as a conflicting "known version".
    """
    versions = collect_versions_from_keys(sweeps_by_arm, keys)
    warnings: list[str] = []
    errors: list[str] = []

    if len(versions) > 1:
        errors.append(
            f"Mixed benchmark versions in selected runs: {sorted(versions)}. "
            "Cannot compare results across versions."
        )

    if manifest_version and versions:
        if manifest_version not in versions:
            errors.append(
                f"Manifest declares version {manifest_version} but "
                f"selected runs contain {sorted(versions)}"
            )

    return VersionCheck(versions=versions, warnings=warnings, errors=errors)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        metavar="manifest",
        help="Path to manifest.json, run directory containing manifest.json, "
             "or runs directory directly",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        help="AgentDojo runs directory (overrides manifest)",
    )
    parser.add_argument(
        "--model",
        help="Model label to filter (default: from manifest or gpt-4o-2024-08-06)",
    )
    parser.add_argument(
        "--attack",
        help="Attack type to report (default: from manifest or tool_knowledge)",
    )
    parser.add_argument(
        "--md",
        type=Path,
        help="Write markdown output to file",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Write JSON output to file",
    )
    args = parser.parse_args()

    # Load manifest if provided
    manifest: dict | None = None
    manifest_path: Path | None = None

    if args.input:
        if args.input.is_file():
            # Direct manifest.json
            try:
                text = args.input.read_text(encoding="utf-8")
                manifest = json.loads(text)
                manifest_path = args.input
            except UnicodeDecodeError as e:
                print(f"Error reading manifest (encoding): {e}", file=sys.stderr)
                return 1
            except json.JSONDecodeError as e:
                print(f"Error reading manifest (JSON): {e}", file=sys.stderr)
                return 1
            except OSError as e:
                print(f"Error reading manifest: {e}", file=sys.stderr)
                return 1

            # Validate manifest structure
            if not isinstance(manifest, dict):
                print("Error: manifest must be a JSON object", file=sys.stderr)
                return 1

        elif args.input.is_dir():
            # Check if it's a run directory containing manifest.json
            potential_manifest = args.input / "manifest.json"
            if potential_manifest.is_file():
                try:
                    text = potential_manifest.read_text(encoding="utf-8")
                    manifest = json.loads(text)
                    manifest_path = potential_manifest
                except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
                    print(f"Error reading {potential_manifest}: {e}", file=sys.stderr)
                    return 1

                if not isinstance(manifest, dict):
                    print("Error: manifest must be a JSON object", file=sys.stderr)
                    return 1
            else:
                # Treat as runs directory directly, but don't override explicit --logdir
                if not args.logdir:
                    args.logdir = args.input
        else:
            print(f"Error: {args.input} does not exist", file=sys.stderr)
            return 1

    # Resolve logdir
    logdir = resolve_logdir(args, manifest, manifest_path)
    if not logdir or not logdir.is_dir():
        print("Error: could not find AgentDojo runs directory", file=sys.stderr)
        print("Provide --logdir, a manifest with agentdojo.logdir, or a runs directory",
              file=sys.stderr)
        return 1

    # Resolve model and attack from manifest's agentdojo metadata
    ad_meta: dict = {}
    if manifest:
        ad_meta_raw = manifest.get("agentdojo")
        if isinstance(ad_meta_raw, dict):
            ad_meta = ad_meta_raw

    model = args.model or ad_meta.get("model_label") or "gpt-4o-2024-08-06"
    attack = args.attack or ad_meta.get("attack") or "tool_knowledge"
    manifest_version = ad_meta.get("version") if isinstance(ad_meta.get("version"), str) else None

    warnings: list[str] = []

    # Load all arms
    sweeps_by_arm: dict[str, list[SweepData]] = {}
    missing_arms: list[str] = []
    for _, no_rep_tag, with_rep_tag in ROWS:
        for tag in (no_rep_tag, with_rep_tag):
            sweeps = load_sweeps(logdir, tag, model)
            if sweeps:
                sweeps_by_arm[tag] = sweeps
            else:
                missing_arms.append(tag)

    if not sweeps_by_arm:
        print(f"Error: no sweeps found for model {model} in {logdir}", file=sys.stderr)
        return 1

    # Warn about missing arms
    if missing_arms:
        warnings.append(f"Missing arms: {', '.join(missing_arms)}")

    # Check for unreadable files, invalid schema, and duplicates
    for tag, sweeps in sweeps_by_arm.items():
        for sweep in sweeps:
            if sweep.unreadable:
                warnings.append(
                    f"{sweep.name}: {sweep.unreadable} unreadable file(s)"
                )
            if sweep.invalid_schema:
                warnings.append(
                    f"{sweep.name}: {sweep.invalid_schema} invalid schema"
                )
            if sweep.conflicting_duplicates:
                warnings.append(
                    f"{sweep.name}: {sweep.conflicting_duplicates} conflicting duplicate(s)"
                )
            if sweep.duplicates:
                warnings.append(
                    f"{sweep.name}: {sweep.duplicates} duplicate(s) (same scores, kept first)"
                )

    # Every discovered sweep participates; empty evidence cannot shrink coverage.
    keys = matched_keys(sweeps_by_arm, attack)
    if not keys:
        print("Error: no common runs across arms; cannot compute matched metrics",
              file=sys.stderr)
        # Show per-sweep diagnostics for the selected attack/clean population
        for tag, sweeps in sweeps_by_arm.items():
            for sweep in sweeps:
                # Count only relevant runs (clean + selected attack)
                relevant = sum(
                    1 for k, v in sweep.runs.items()
                    if not v.get("injection_task_id") or v.get("attack_type") == attack
                )
                diag = f"  {sweep.name}: {relevant} relevant runs (of {len(sweep.runs)} total)"
                if sweep.invalid_schema:
                    diag += f", {sweep.invalid_schema} invalid"
                if sweep.unreadable:
                    diag += f", {sweep.unreadable} unreadable"
                print(diag, file=sys.stderr)
        return 1

    # Check version issues AFTER filtering to selected runs
    # (Finding 3: only infer from clean + selected attack, reject conflicts)
    version_check = check_version_issues(sweeps_by_arm, keys, manifest_version)
    warnings.extend(version_check.warnings)
    all_versions = version_check.versions

    if version_check.errors:
        for err in version_check.errors:
            print(f"Error: {err}", file=sys.stderr)
        return 1

    # Derive domains from matched keys
    domains = {k[0] for k in keys}

    # Check for dropped runs - only count relevant runs
    max_relevant = 0
    for sweeps in sweeps_by_arm.values():
        for sweep in sweeps:
            relevant = sum(
                1 for k, v in sweep.runs.items()
                if not v.get("injection_task_id") or v.get("attack_type") == attack
            )
            max_relevant = max(max_relevant, relevant)

    dropped = max_relevant - len(keys)
    if dropped > 0:
        warnings.append(
            f"{dropped} run(s) absent from at least one sweep; "
            f"metrics computed over {len(keys)} common runs"
        )

    # Aggregate metrics
    arms_no_rep = []
    arms_with_rep = []
    for label, no_rep_tag, with_rep_tag in ROWS:
        no_rep_sweeps = sweeps_by_arm.get(no_rep_tag, [])
        with_rep_sweeps = sweeps_by_arm.get(with_rep_tag, [])

        arms_no_rep.append(
            aggregate_arm(label, no_rep_tag, no_rep_sweeps, keys, attack)
        )
        arms_with_rep.append(
            aggregate_arm(label, with_rep_tag, with_rep_sweeps, keys, attack)
        )

    # Track sweep counts per arm (Finding 4: count all present arms, not just
    # those with clean utility - use actual discovered sweep count, falling
    # back to any valid metric list length)
    sweep_counts_by_arm: dict[str, int] = {}
    for agg in arms_no_rep + arms_with_rep:
        # Use n_sweeps (number of sweeps that had ANY matched runs)
        # This handles attack-only sweeps where clean_utility would be None
        if agg.n_sweeps > 0:
            # Actual sweep count with matched runs
            sweep_counts_by_arm[agg.tag] = agg.n_sweeps
        else:
            # Fallback: check if any metric list has valid values
            any_valid = (
                _valid_values(agg.clean_utility) or
                _valid_values(agg.utility_attack) or
                _valid_values(agg.asr_agentdojo) or
                _valid_values(agg.asr_effect)
            )
            if any_valid:
                sweep_counts_by_arm[agg.tag] = len(any_valid)

    # Find minimum sweep count among arms with data
    sweep_count_values = set(sweep_counts_by_arm.values())
    n_sweeps = min(sweep_count_values, default=0)
    if len(sweep_count_values) > 1:
        # Per-arm counts disclosure (Finding 4: disclose per-arm counts)
        per_arm = ', '.join(f'{t}={c}' for t, c in sorted(sweep_counts_by_arm.items()))
        warnings.append(
            f"Arms have unequal sweep counts ({per_arm})"
        )

    # Print warnings
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    # Generate outputs
    md = format_markdown(
        arms_no_rep, arms_with_rep, attack, model, n_sweeps,
        sweep_counts_by_arm, keys, warnings, domains
    )
    print(md)


    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md + "\n")
        print(f"Markdown → {args.md}")

    if args.json:
        data = format_json(
            arms_no_rep, arms_with_rep, attack, model, n_sweeps,
            sweep_counts_by_arm, keys, warnings, all_versions, domains
        )
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2) + "\n")
        print(f"JSON → {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
