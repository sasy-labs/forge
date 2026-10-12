"""Bounded live airline concurrency sweep with explicit task selection.

Drives ``run_benchmark.py`` at fixed concurrency levels with a controlled,
deliberately chosen task set, capturing per-call enforcement latencies. Each
concurrency level runs in a fresh subprocess with its own managed engine
(started by run_benchmark.py), isolating process-level state (compile cache,
channel pools) while reusing the same checkout binary.

The managed engine's built-in warmup primes the policy compile cache before
measured trials begin. Each concurrency level gets exactly one warmup trial
(excluded from measurements) on its own fresh server.

The fixed task set (default: 1 7 8 12 17 19 43 47) exercises lookup, booking,
changes, cancellation, and longer read/denial paths. This is a deliberately
chosen workload, not a random representative statistical sample.

Concurrencies default to 1 2 4 8 16 32. Trials scale as ceil(max(24, 4*c)/K)
where K is the number of tasks: 3,3,3,4,8,16 for 8 tasks = 296 measured
executions total (plus exactly 6 excluded warmup trials, one per cell).
These are pilot budgets, not stable-p99 guarantees.

Usage::

    # Full sweep with defaults (airline, gpt agent, opus user, enforce-on)
    uv run python scripts/evaluation/run_latency_sweep.py

    # Dry run — print plan, run nothing
    uv run python scripts/evaluation/run_latency_sweep.py --dry-run

    # Custom task IDs and concurrencies
    uv run python scripts/evaluation/run_latency_sweep.py \\
        --task-ids 8 43 47 --concurrencies 1 4 16

    # Override trials per task (applies to all concurrency levels)
    uv run python scripts/evaluation/run_latency_sweep.py \\
        --concurrencies 1 --trials-override 1

    # Reduce a completed sweep without re-running
    uv run python scripts/evaluation/run_latency_sweep.py \\
        --reduce-only results/latency-sweep/2026-10-05T14-00-00
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_BENCH = REPO_ROOT / "scripts" / "evaluation" / "run_benchmark.py"
SERVER_BIN_DEFAULT = REPO_ROOT / "sasy" / "target" / "debug" / "sasy"
TAU2_DOMAINS_DIR = REPO_ROOT / "experiments" / "tau2-bench" / "data" / "tau2" / "domains"

# The deliberately chosen task set covering lookup, booking, changes,
# cancellation, and longer read/denial paths.
DEFAULT_TASK_IDS = (1, 7, 8, 12, 17, 19, 43, 47)
DEFAULT_CONCURRENCIES = (1, 2, 4, 8, 16, 32)
DEFAULT_DOMAIN = "airline"
DEFAULT_MODE = "enforce-on"
DEFAULT_AGENT_MODEL = "gpt"
DEFAULT_USER_MODEL = "opus"


def _load_dotenv(target: dict[str, str], path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            target.setdefault(k, v)


_load_dotenv(os.environ, REPO_ROOT / ".env")
_load_dotenv(os.environ, REPO_ROOT / ".env-credentials")


def tau2_base_task_ids(domain: str) -> set[str]:
    """All task IDs in the domain's base split."""
    split_file = TAU2_DOMAINS_DIR / domain / "split_tasks.json"
    if not split_file.exists():
        raise FileNotFoundError(f"tau2 domain not found: {domain}")
    data = json.loads(split_file.read_text())
    return {str(t) for t in data.get("base", [])}


def trials_for_concurrency(concurrency: int, num_tasks: int) -> int:
    """Compute trials per task for a given concurrency level.

    Formula: ceil(max(24, 4 * concurrency) / num_tasks).
    For K=8 tasks: c=1→3, c=2→3, c=4→3, c=8→4, c=16→8, c=32→16.
    """
    return math.ceil(max(24, 4 * concurrency) / num_tasks)


def validate_task_ids(task_ids: list[int], domain: str) -> None:
    """Validate that all task IDs are positive, unique, and in the domain's base split."""
    if not task_ids:
        raise ValueError("task_ids cannot be empty")
    if any(t < 0 for t in task_ids):
        raise ValueError(f"task_ids must be non-negative: {task_ids}")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"task_ids must be unique: {task_ids}")
    valid = tau2_base_task_ids(domain)
    invalid = {str(t) for t in task_ids} - valid
    if invalid:
        raise ValueError(
            f"task_ids not in {domain} base split: {sorted(invalid)}. "
            f"Valid: {sorted(valid, key=int)}"
        )


def validate_concurrencies(concurrencies: list[int]) -> None:
    """Validate that all concurrency levels are positive."""
    if not concurrencies:
        raise ValueError("concurrencies cannot be empty")
    if any(c < 1 for c in concurrencies):
        raise ValueError(f"concurrencies must be positive: {concurrencies}")
    if len(concurrencies) != len(set(concurrencies)):
        raise ValueError(f"concurrencies must be unique: {concurrencies}")


@dataclass
class CellPlan:
    """One concurrency level in the sweep."""
    concurrency: int
    trials_per_task: int
    task_ids: list[int]
    out_dir: Path
    port: int

    @property
    def total_trials(self) -> int:
        return len(self.task_ids) * self.trials_per_task


@dataclass
class SweepResult:
    """Result of one concurrency cell."""
    concurrency: int
    returncode: int
    wall_s: float
    out_dir: Path
    child_manifest_path: Path | None = None
    sdk_log: Path | None = None
    server_replay: Path | None = None
    num_auth_calls: int = 0
    num_tasks: int = 0
    warmup_excluded: bool = True
    warmup_failed: bool = False
    validation_errors: list[str] = field(default_factory=list)


def build_plan(
    run_root: Path,
    task_ids: list[int],
    concurrencies: list[int],
    base_port: int,
    trials_override: int | None = None,
) -> list[CellPlan]:
    """Build the plan for all concurrency levels.

    If trials_override is provided, use that for all cells instead of the formula.
    """
    K = len(task_ids)
    plan = []
    for i, c in enumerate(concurrencies):
        trials = trials_override if trials_override is not None else trials_for_concurrency(c, K)
        plan.append(CellPlan(
            concurrency=c,
            trials_per_task=trials,
            task_ids=task_ids,
            out_dir=run_root / f"c{c}",
            port=base_port + i,
        ))
    return plan


@dataclass
class ChildManifestInfo:
    """Parsed information from a child benchmark manifest."""
    path: Path
    warmups: list[dict]
    cells: list[dict]
    sdk_log: Path | None
    server_replay: Path | None
    warmup_ok: bool
    measured_ok: bool
    errors: list[str]


def _parse_child_manifest(child_manifest_path: Path) -> ChildManifestInfo:
    """Parse and validate a child benchmark manifest.

    Validates:
    - Exactly one warmup with rc=0 (warmup_ok)
    - Exactly one measured cell (measured_ok)
    - SDK log exists in the measured cell's latency_capture
    - Server replay exists in the measured cell's latency_capture
    """
    errors: list[str] = []

    if not child_manifest_path.exists():
        return ChildManifestInfo(
            path=child_manifest_path, warmups=[], cells=[], sdk_log=None,
            server_replay=None, warmup_ok=False, measured_ok=False,
            errors=["child manifest not found"],
        )

    try:
        manifest = json.loads(child_manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return ChildManifestInfo(
            path=child_manifest_path, warmups=[], cells=[], sdk_log=None,
            server_replay=None, warmup_ok=False, measured_ok=False,
            errors=[f"child manifest parse error: {e}"],
        )

    manifest_dir = child_manifest_path.parent
    warmups = manifest.get("warmups", [])
    cells = manifest.get("cells", [])

    # Validate exactly one warmup with rc=0
    warmup_ok = False
    if len(warmups) == 0:
        errors.append("no warmup in child manifest")
    elif len(warmups) > 1:
        errors.append(f"expected 1 warmup, got {len(warmups)}")
    else:
        warmup_rc = warmups[0].get("returncode")
        if warmup_rc == 0:
            warmup_ok = True
        else:
            errors.append(f"warmup failed with rc={warmup_rc}")

    # Validate exactly one measured cell
    measured_ok = False
    if len(cells) == 0:
        errors.append("no measured cell in child manifest")
    elif len(cells) > 1:
        errors.append(f"expected 1 measured cell, got {len(cells)}")
    else:
        cell_rc = cells[0].get("returncode")
        if cell_rc == 0:
            measured_ok = True
        else:
            errors.append(f"measured cell failed with rc={cell_rc}")

    # Resolve SDK log from measured cell's latency_capture (not fallback)
    sdk_log = None
    server_replay = None

    if cells:
        capture = cells[0].get("latency_capture", {})
        if capture.get("sdk_log"):
            sdk_path = manifest_dir / capture["sdk_log"]
            if sdk_path.exists():
                sdk_log = sdk_path
            else:
                errors.append(f"sdk_log not found: {capture['sdk_log']}")
        else:
            errors.append("no sdk_log in measured cell latency_capture")

        if capture.get("server_replay"):
            replay_path = manifest_dir / capture["server_replay"]
            if replay_path.exists():
                server_replay = replay_path
            else:
                errors.append(f"server_replay not found: {capture['server_replay']}")
        else:
            errors.append("no server_replay in measured cell latency_capture")

    return ChildManifestInfo(
        path=child_manifest_path,
        warmups=warmups,
        cells=cells,
        sdk_log=sdk_log,
        server_replay=server_replay,
        warmup_ok=warmup_ok,
        measured_ok=measured_ok,
        errors=errors,
    )


def _count_auth_calls(sdk_log: Path | None) -> int:
    """Count valid authorization calls from SDK log."""
    if sdk_log is None or not sdk_log.exists():
        return 0
    count = 0
    for line in sdk_log.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if not isinstance(rec, dict):
                continue
            rtt = rec.get("rtt_us")
            # Validate: must be numeric int/float, finite, non-negative
            if rtt is not None and not isinstance(rtt, bool) and isinstance(rtt, (int, float)):
                try:
                    val = float(rtt)
                    if math.isfinite(val) and val >= 0:
                        count += 1
                except (ValueError, TypeError):
                    pass
        except json.JSONDecodeError:
            pass
    return count


def run_cell(
    cell: CellPlan,
    args: argparse.Namespace,
    server_bin: Path,
    policy_path: Path,
) -> SweepResult:
    """Run one concurrency level using run_benchmark.py's managed server.

    Each cell gets its own fresh server with built-in warmup. The SDK log
    and server replay are resolved from the child benchmark manifest.

    Fails the cell (nonzero returncode) if:
    - run_benchmark.py returns nonzero
    - Child manifest is missing or invalid
    - Warmup did not complete successfully (rc=0)
    - SDK log is missing or empty
    - Server replay is missing
    """
    print(f"\n{'='*70}")
    print(f"Concurrency {cell.concurrency}: {len(cell.task_ids)} tasks × "
          f"{cell.trials_per_task} trials = {cell.total_trials} measured + 1 warmup")
    print(f"  output: {cell.out_dir}")
    print(f"{'='*70}")

    if args.dry_run:
        return SweepResult(
            concurrency=cell.concurrency, returncode=0, wall_s=0.0,
            out_dir=cell.out_dir,
        )

    cell.out_dir.mkdir(parents=True, exist_ok=True)

    # Build the benchmark command - let run_benchmark manage the server
    cmd = [
        sys.executable, str(RUN_BENCH),
        "--targets", "tau2",
        "--domain", args.domain,
        "--mode", args.mode,
        "--model", args.model,
        "--user-model", args.user_model,
        "--task-ids", *[str(t) for t in cell.task_ids],
        "--num-trials", str(cell.trials_per_task),
        "--max-concurrency", str(cell.concurrency),
        "--output-root", str(cell.out_dir),
        "--start-server",  # run_benchmark manages its own server
        "--server-port", str(cell.port),
        "--server-bin", str(server_bin),
        "--server-policy", str(policy_path),
        "--warmup",  # built-in warmup primes this cell's engine
        "--capture-latency",
        "--cell-timeout", str(args.cell_timeout),
    ]

    env = os.environ.copy()
    # Enable readable session IDs for task-to-session mapping
    env["SASY_SESSION_READABLE"] = "1"

    print(f"  [cell] {' '.join(shlex.quote(c) for c in cmd[:10])}...")
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd, env=env, cwd=REPO_ROOT,
        start_new_session=True,
    )
    try:
        bench_rc = proc.wait()
    except KeyboardInterrupt:
        print(f"  ! interrupted; killing benchmark (pid {proc.pid})")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            # The child tears down its engine and benchmark process groups
            # sequentially (up to 5 seconds each); keep their owner alive.
            proc.wait(timeout=30)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise
    wall_s = time.monotonic() - t0

    # Find child manifest (run_benchmark creates <output_root>/<timestamp>/manifest.json)
    child_manifests = list(cell.out_dir.glob("*/manifest.json"))
    validation_errors: list[str] = []

    if len(child_manifests) == 0:
        validation_errors.append("no child manifest found")
        child_info = None
    elif len(child_manifests) > 1:
        validation_errors.append(f"expected 1 child manifest, got {len(child_manifests)}")
        child_info = None
    else:
        child_info = _parse_child_manifest(child_manifests[0])
        validation_errors.extend(child_info.errors)

    # Determine final returncode
    # Fail if: bench_rc nonzero, warmup failed, or validation errors
    warmup_failed = child_info is not None and not child_info.warmup_ok
    has_validation_errors = len(validation_errors) > 0

    if bench_rc != 0:
        final_rc = bench_rc
    elif warmup_failed:
        final_rc = 101  # Warmup failure
    elif has_validation_errors:
        final_rc = 102  # Validation failure
    else:
        final_rc = 0

    # Copy resolved artifacts to canonical cell locations for summarizer
    cell_sdk_log = cell.out_dir / "latency.jsonl"
    cell_server_replay = cell.out_dir / "server_replay.jsonl"

    sdk_log = child_info.sdk_log if child_info else None
    server_replay = child_info.server_replay if child_info else None

    if sdk_log and sdk_log.exists():
        shutil.copy(sdk_log, cell_sdk_log)
    if server_replay and server_replay.exists():
        shutil.copy(server_replay, cell_server_replay)

    num_auth_calls = _count_auth_calls(cell_sdk_log if cell_sdk_log.exists() else None)

    # Fail on missing/empty SDK log (not just warn)
    if final_rc == 0 and num_auth_calls == 0:
        validation_errors.append("cell succeeded but no SDK auth calls captured")
        final_rc = 103

    # Print status
    status = "OK" if final_rc == 0 else f"FAILED (rc={final_rc})"
    print(f"  [cell] {status}, {num_auth_calls} auth calls, {wall_s:.1f}s")
    if validation_errors:
        for err in validation_errors:
            print(f"    ERROR: {err}")

    return SweepResult(
        concurrency=cell.concurrency, returncode=final_rc, wall_s=wall_s,
        out_dir=cell.out_dir,
        child_manifest_path=child_manifests[0] if len(child_manifests) == 1 else None,
        sdk_log=cell_sdk_log if cell_sdk_log.exists() else None,
        server_replay=cell_server_replay if cell_server_replay.exists() else None,
        num_auth_calls=num_auth_calls,
        num_tasks=len(cell.task_ids),
        warmup_excluded=True,
        warmup_failed=warmup_failed,
        validation_errors=validation_errors,
    )


def compute_binary_hash(path: Path) -> str:
    """Compute SHA256 hash of the binary."""
    import hashlib
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def portable_path(path: Path, manifest_dir: Path) -> str:
    """Store path relative to manifest_dir if inside, else absolute."""
    try:
        return str(path.relative_to(manifest_dir))
    except ValueError:
        return str(path)


def _write_manifest(path: Path, manifest: dict) -> None:
    """Write manifest atomically."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(path)


def run_sweep(args: argparse.Namespace) -> int:
    """Run the complete concurrency sweep."""
    # Validate inputs
    task_ids = list(args.task_ids)
    validate_task_ids(task_ids, args.domain)
    validate_concurrencies(args.concurrencies)

    # Validate trials_override if provided
    if args.trials_override is not None and args.trials_override < 1:
        print(f"ERROR: --trials-override must be at least 1, got {args.trials_override}")
        return 2

    server_bin = args.server_bin.expanduser().resolve()
    if not server_bin.exists() and not args.dry_run:
        print(f"ERROR: server binary not found: {server_bin}")
        print("       Build with: make build-rust-dev")
        return 2

    policy_path = REPO_ROOT / "experiments" / "tau2" / f"{args.domain}_policy.dl"
    if not policy_path.exists() and not args.dry_run:
        print(f"ERROR: policy not found: {policy_path}")
        return 2

    # Create run directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_root = (args.output_root / timestamp).resolve()

    # Fail on existing directory - no resume support
    if run_root.exists():
        print(f"\nERROR: run directory exists: {run_root}")
        print("       Remove manually or use a different --output-root.")
        return 1

    plan = build_plan(
        run_root, task_ids, args.concurrencies, args.base_port,
        trials_override=args.trials_override,
    )

    # Print plan
    total_executions = sum(c.total_trials for c in plan)
    warmup_count = len(plan)  # one warmup per cell
    print(f"\nLatency sweep: {len(plan)} concurrency level(s), "
          f"{len(task_ids)} tasks, {total_executions} measured + {warmup_count} warmup")
    print(f"Run directory: {run_root}")
    if args.trials_override is not None:
        print(f"Trials override: {args.trials_override} per task (formula disabled)")
    print()
    print("Plan:")
    for cell in plan:
        print(f"  c={cell.concurrency:2d}: {len(cell.task_ids)} tasks × "
              f"{cell.trials_per_task} trials = {cell.total_trials} measured + 1 warmup")

    if args.dry_run:
        print("\n[dry-run] Nothing executed.")
        return 0

    run_root.mkdir(parents=True, exist_ok=True)

    # Initialize manifest
    manifest: dict = {
        "kind": "latency-sweep",
        "version": 2,
        "timestamp": timestamp,
        "run_dir": ".",
        "domain": args.domain,
        "mode": args.mode,
        "agent_model": args.model,
        "user_model": args.user_model,
        "task_ids": task_ids,
        "concurrencies": args.concurrencies,
        "trials_override": args.trials_override,
        "binary": str(server_bin),
        "binary_sha256": compute_binary_hash(server_bin) if server_bin.exists() else None,
        "policy": str(policy_path),
        "warmup_per_cell": True,
        "cells": [],
        "state": "running",
    }
    manifest_path = run_root / "manifest.json"
    _write_manifest(manifest_path, manifest)

    # Run each concurrency level (each manages its own server + warmup)
    results: list[SweepResult] = []
    failures: list[tuple[int, int]] = []

    try:
        for cell in plan:
            result = run_cell(cell, args, server_bin, policy_path)
            results.append(result)

            # Update manifest
            cell_entry = {
                "concurrency": result.concurrency,
                "returncode": result.returncode,
                "wall_s": round(result.wall_s, 2),
                "num_auth_calls": result.num_auth_calls,
                "num_tasks": result.num_tasks,
                "trials_per_task": cell.trials_per_task,
                "warmup_excluded": result.warmup_excluded,
                "warmup_failed": result.warmup_failed,
                "out_dir": portable_path(result.out_dir, run_root),
            }
            if result.child_manifest_path:
                cell_entry["child_manifest"] = portable_path(result.child_manifest_path, run_root)
            if result.sdk_log:
                cell_entry["sdk_log"] = portable_path(result.sdk_log, run_root)
            if result.server_replay:
                cell_entry["server_replay"] = portable_path(result.server_replay, run_root)
            if result.validation_errors:
                cell_entry["validation_errors"] = result.validation_errors

            manifest["cells"] = [
                c for c in manifest["cells"]
                if c.get("concurrency") != cell.concurrency
            ] + [cell_entry]
            _write_manifest(manifest_path, manifest)

            if result.returncode != 0:
                failures.append((result.concurrency, result.returncode))
                if not args.continue_on_error:
                    break

    except KeyboardInterrupt:
        print("\n! interrupted; writing partial manifest")
        manifest["state"] = "interrupted"
        _write_manifest(manifest_path, manifest)
        return 130

    manifest["state"] = "stopped"
    _write_manifest(manifest_path, manifest)

    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    total_auth = sum(r.num_auth_calls for r in results)
    total_wall = sum(r.wall_s for r in results)
    print(f"Cells: {len(results)}, Auth calls: {total_auth}, Wall: {total_wall:.1f}s")
    if failures:
        print(f"Failures: {failures}")

    print(f"\nManifest: {manifest_path}")
    print(f"Reduce with: uv run python {__file__} --reduce-only {run_root}")

    return 1 if failures else 0


def reduce_sweep(run_dir: Path) -> int:
    """Reduce a completed sweep to statistics."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: no manifest.json in {run_dir}")
        return 1

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "latency-sweep":
        print(f"ERROR: not a latency-sweep manifest (kind={manifest.get('kind')})")
        return 1

    print(f"Sweep manifest: {manifest_path}")
    print(f"Domain: {manifest.get('domain')}, Mode: {manifest.get('mode')}")
    print(f"Tasks: {manifest.get('task_ids')}")
    print(f"Cells: {len(manifest.get('cells', []))}")

    # Import bootstrap summarizer
    try:
        from summarize_latency_sessions import summarize_sweep
        return summarize_sweep(manifest, run_dir)
    except ImportError:
        print("\n(summarize_latency_sessions not yet available)")
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    task = parser.add_argument_group("task selection")
    task.add_argument(
        "--task-ids", type=int, nargs="+", default=list(DEFAULT_TASK_IDS),
        help=f"Task IDs to run (default: {' '.join(map(str, DEFAULT_TASK_IDS))}). "
        "Must be in the domain's base split.",
    )
    task.add_argument(
        "--domain", default=DEFAULT_DOMAIN, choices=("airline", "retail"),
        help=f"tau2 domain (default: {DEFAULT_DOMAIN}).",
    )

    conc = parser.add_argument_group("concurrency")
    conc.add_argument(
        "--concurrencies", type=int, nargs="+", default=list(DEFAULT_CONCURRENCIES),
        help=f"Concurrency levels (default: {' '.join(map(str, DEFAULT_CONCURRENCIES))}).",
    )
    conc.add_argument(
        "--trials-override", type=int, default=None,
        help="Override trials per task for all cells (default: ceil(max(24, 4*c)/K)).",
    )

    model = parser.add_argument_group("models")
    model.add_argument(
        "--model", default=DEFAULT_AGENT_MODEL,
        help=f"Agent model (default: {DEFAULT_AGENT_MODEL}).",
    )
    model.add_argument(
        "--user-model", default=DEFAULT_USER_MODEL,
        help=f"Simulated user model (default: {DEFAULT_USER_MODEL}).",
    )
    model.add_argument(
        "--mode", default=DEFAULT_MODE, choices=("enforce-on", "enforce-off"),
        help=f"Enforcement mode (default: {DEFAULT_MODE}).",
    )

    run = parser.add_argument_group("execution")
    run.add_argument(
        "--output-root", type=Path,
        default=REPO_ROOT / "results" / "latency-sweep",
        help="Output directory (default: results/latency-sweep).",
    )
    run.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without executing.",
    )
    run.add_argument(
        "--continue-on-error", action="store_true",
        help="Continue to next concurrency level after a failure.",
    )
    run.add_argument(
        "--cell-timeout", type=int, default=7200,
        help="Per-cell timeout in seconds (default: 7200 = 2h).",
    )
    run.add_argument(
        "--reduce-only", type=Path, default=None,
        help="Reduce an existing sweep without re-running.",
    )

    srv = parser.add_argument_group("server")
    srv.add_argument(
        "--server-bin", type=Path, default=SERVER_BIN_DEFAULT,
        help=f"Path to sasy binary (default: {SERVER_BIN_DEFAULT}).",
    )
    srv.add_argument(
        "--base-port", type=int, default=12089,
        help="Base port for managed servers (default: 12089). "
        "Each concurrency uses base_port + index.",
    )

    return parser


def main() -> int:
    signal.signal(signal.SIGTERM, lambda s, f: (_ for _ in ()).throw(KeyboardInterrupt()))

    parser = build_parser()
    args = parser.parse_args()

    if args.reduce_only:
        return reduce_sweep(args.reduce_only.resolve())

    return run_sweep(args)


if __name__ == "__main__":
    sys.exit(main())
