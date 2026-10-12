"""Requeue tau2 infrastructure failures without losing their original evidence."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from manifest_paths import normalize_cell_entry
from tau2_results import cell_results_file


def infrastructure_failures(path: Path) -> list[dict]:
    """Read the monolithic text checkpoint the benchmark runner produces."""
    data = json.loads(path.read_text())
    simulations = data.get("simulations")
    if not isinstance(simulations, list) or any(not isinstance(s, dict) for s in simulations):
        raise ValueError(f"invalid tau2 simulations in {path}")
    return [s for s in simulations if s.get("termination_reason") == "infrastructure_error"]


def finished_cells(
    previous: dict, run_root: Path, retry_infrastructure_errors: bool,
    selected_ids: set[str] | None = None,
    checkpoint_paths: dict[str, Path] | None = None,
) -> dict:
    """An rc=0 tau2 cell with infrastructure failures is pending when requested."""
    manifest_run_dir = previous.get("run_dir")
    finished = {}
    for cell in previous.get("cells", []):
        if (not cell.get("id") or cell.get("returncode") != 0
                or cell.get("state") in {"pending", "running"}):
            continue
        selected = selected_ids is None or cell["id"] in selected_ids
        if retry_infrastructure_errors and selected and cell.get("target") == "tau2":
            path = cell_results_file(cell, run_root, run_dir=manifest_run_dir)
            if path is None:
                raise ValueError(f"no tau2 results recorded for {cell['id']}")
            if infrastructure_failures(path):
                if checkpoint_paths is not None:
                    expected = checkpoint_paths.get(cell["id"])
                    if expected is None or expected.resolve() != path.resolve():
                        raise ValueError(
                            f"retry checkpoint for {cell['id']} is outside its current cell directory;"
                            " copy the original checkpoint into this run before recovery"
                        )
                continue
        finished[cell["id"]] = cell
    return finished



def pending_cell_entries(
    previous: dict,
    pending_ids: set[str],
    *,
    new_manifest_dir: Path | None = None,
) -> list[dict]:
    """Keep prior evidence, but persist requeued work as pending before startup.

    A crash after tau2 removes failure placeholders must not leave an rc=0
    entry that ordinary resume mistakes for a completed checkpoint.

    When new_manifest_dir is provided, legacy absolute paths in retained entries
    are normalized to be relative to the new directory. This ensures moved runs
    resolve artifacts correctly when run_dir is set to '.'.
    """
    previous_run_dir = previous.get("run_dir")
    entries = []
    for cell in previous.get("cells", []):
        entry = dict(cell)
        # Deep-copy latency_capture to avoid mutating the input
        if entry.get("latency_capture"):
            entry["latency_capture"] = dict(entry["latency_capture"])
            if entry["latency_capture"].get("attempts"):
                entry["latency_capture"]["attempts"] = list(entry["latency_capture"]["attempts"])

        if entry.get("id") in pending_ids:
            if entry.get("state") not in {"pending", "running"}:
                entry["previous_attempt"] = {k: v for k, v in cell.items() if k != "previous_attempt"}
            entry["state"] = "pending"
            entry["returncode"] = None

        # Normalize paths for moved runs
        if new_manifest_dir is not None:
            normalize_cell_entry(entry, previous_run_dir, new_manifest_dir)

        entries.append(entry)
    return entries


def archive_infrastructure_failures(path: Path) -> dict | None:
    """Snapshot the complete checkpoint before tau2 removes failed placeholders.

    This also runs for interrupted cells: tau2's own resume always retries
    infrastructure failures, even without the benchmark's requeue option.
    """
    if not path.exists():
        return None
    failures = infrastructure_failures(path)
    if not failures:
        return None
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    archive_dir = path.parent / "infrastructure-retries"
    archive_dir.mkdir(exist_ok=True)
    snapshot = archive_dir / f"results-{digest}.json"
    if snapshot.exists():
        if snapshot.read_bytes() != raw:
            raise ValueError(f"infrastructure retry snapshot changed: {snapshot}")
    else:
        with tempfile.NamedTemporaryFile(dir=archive_dir, delete=False) as stream:
            tmp = Path(stream.name)
            try:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        try:
            tmp.replace(snapshot)
        finally:
            tmp.unlink(missing_ok=True)
    return {
        "failure_count": len(failures),
        "snapshot": str(snapshot),
        "snapshot_sha256": digest,
        "trial_keys": [[s.get("trial"), s.get("task_id"), s.get("seed")] for s in failures],
    }
