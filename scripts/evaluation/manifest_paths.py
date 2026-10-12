"""Portable path resolution for evaluation manifests.

Manifests store artifact paths as:
- **Relative paths** for internal artifacts (within the run directory): resolved
  against the manifest file's containing directory.
- **Absolute paths** for external resources (policies in the repo, server
  binaries): retained verbatim; environment and argv values are provenance.

New manifests use ONE field with a relative value for internal artifacts (e.g.,
``policy="replay-policy/policy.dl"``). Legacy manifests may have absolute paths
in the primary field and a ``*_relpath`` sibling for portability after moves;
readers try both, preferring the relative form.

Resolution order:
1. If the value is relative, resolve against ``manifest_dir``.
2. If absolute and exists, use it directly (unmoved run or external resource).
3. If absolute and a ``run_dir`` fallback is available, rebase the relative
   portion of the original path under the current manifest directory.

Hash verification is mandatory for capture artifacts. A local file that matches
the expected path but differs from the recorded hash MUST fail, not fall back
to another candidate.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


def sha256(path: Path) -> str:
    """Compute the SHA256 digest of a file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def portable_path(path: Path, manifest_dir: Path, *, run_dir: Path | None = None) -> str:
    """Store a path for the manifest: internal artifacts relative, external absolute.

    Args:
        path: The path to store (absolute).
        manifest_dir: Directory where the manifest file lives.
        run_dir: Run directory to check containment (defaults to manifest_dir).

    Returns:
        A relative string if the path is inside run_dir, else the absolute string.
    """
    container = run_dir if run_dir is not None else manifest_dir
    try:
        return str(path.resolve().relative_to(manifest_dir.resolve()))
    except ValueError:
        pass
    try:
        path.resolve().relative_to(container.resolve())
        return str(path.resolve().relative_to(manifest_dir.resolve()))
    except ValueError:
        return str(path)


def rebase_capture_paths(
    capture: dict[str, Any],
    cell_dir: Path,
    manifest_dir: Path,
) -> dict[str, Any]:
    """Rebase per-cell capture paths to be relative to the run manifest.

    Per-cell capture files store paths relative to the cell directory (e.g.,
    ``"policy": "replay-policy/policy.dl"``). When embedding a capture in the
    main run manifest, paths must be rebased to be relative to the manifest
    directory (e.g., ``"policy": "airline_enforce-on_gpt/replay-policy/policy.dl"``).

    Args:
        capture: The latency_capture dict from a cell's capture file.
        cell_dir: Directory containing the per-cell capture file.
        manifest_dir: Directory where the run manifest lives.

    Returns:
        A new dict with paths rebased to manifest_dir.
    """
    ARTIFACT_KEYS = ("policy", "functors", "sdk_log", "server_replay")
    result = dict(capture)
    for key in ARTIFACT_KEYS:
        value = result.get(key)
        if value and is_relative_path(value):
            absolute = (cell_dir / value).resolve()
            result[key] = portable_path(absolute, manifest_dir)
    return result


def is_relative_path(value: str) -> bool:
    """True if the path string is relative (not absolute)."""
    return not value.startswith("/")


def resolve_path(
    value: str | None,
    manifest_dir: Path,
    *,
    legacy_run_dir: str | None = None,
) -> Path | None:
    """Resolve a manifest path value to an actual file path.

    For moved runs, local copies (rebased paths) are preferred over
    the original absolute path even when the original still exists.
    This ensures operations use the selected run's artifacts.

    Args:
        value: The path string from the manifest.
        manifest_dir: Directory containing the manifest file.
        legacy_run_dir: Original ``run_dir`` from a legacy manifest, for rebasing
            absolute paths in moved runs.

    Returns:
        The resolved Path, or None if value is None/empty.
    """
    if not value:
        return None

    path = Path(value)

    if is_relative_path(value):
        return (manifest_dir / path).resolve()

    if legacy_run_dir:
        original_run = Path(legacy_run_dir)
        try:
            rel = path.relative_to(original_run)
            rebased = manifest_dir / rel
            if rebased.exists():
                return rebased.resolve()
        except ValueError:
            pass

    if path.exists():
        return path

    return path


def resolve_artifact_path(
    info: dict[str, Any],
    key: str,
    manifest_dir: Path,
    *,
    run_dir: str | None = None,
    verify_hash: bool = False,
) -> Path | None:
    """Resolve an artifact path from a manifest entry.

    For moved runs, local copies (relative paths, *_relpath siblings, rebased
    absolute paths) are preferred over the original absolute path even when
    the original still exists. This ensures verification uses the selected
    run's artifacts.

    Args:
        info: The manifest dict (cell or latency_capture entry).
        key: The base key name (e.g., "policy", "sdk_log", "sim_path").
        manifest_dir: Directory containing the manifest file.
        run_dir: Original run_dir from manifest for layout-based fallback.
        verify_hash: If True and ``{key}_sha256`` exists, verify the file hash.

    Returns:
        The resolved Path, or None if the key is absent.

    Raises:
        ValueError: If verify_hash is True and the hash doesn't match.
    """
    hash_key = key + "_sha256"
    if key.endswith("_path"):
        hash_key = key.replace("_path", "_sha256")
    elif key == "sim_path":
        hash_key = None

    value = info.get(key)
    relpath_key = key + "_relpath" if not key.endswith("_relpath") else key
    relpath = info.get(relpath_key)

    def _verify(candidate: Path) -> Path:
        if verify_hash and hash_key and hash_key in info:
            expected = info[hash_key]
            if expected is not None and sha256(candidate) != expected:
                raise ValueError(
                    f"hash mismatch for {key}: file {candidate} "
                    f"differs from recorded hash"
                )
        return candidate

    if value and is_relative_path(value):
        candidate = manifest_dir / value
        if candidate.is_file():
            return _verify(candidate)

    if relpath:
        candidate = manifest_dir / relpath
        if candidate.is_file():
            return _verify(candidate)

    if value and not is_relative_path(value) and run_dir:
        path = Path(value)
        try:
            rel = path.relative_to(Path(run_dir))
            candidate = manifest_dir / rel
            if candidate.is_file():
                return _verify(candidate)
        except ValueError:
            pass

    if value and not is_relative_path(value):
        path = Path(value)
        if path.is_file():
            return _verify(path)

    if value:
        return Path(value) if not is_relative_path(value) else manifest_dir / value
    return None


def resolve_verified_artifact(
    info: dict[str, Any],
    key: str,
    manifest_dir: Path,
    *,
    run_dir: str | None = None,
) -> Path:
    """Resolve and verify an artifact path, raising on missing or mismatched files.

    This is the strict version for replay capture artifacts where hash
    verification is mandatory.

    Raises:
        ValueError: If the artifact is missing, changed, or hash mismatches.
    """
    hash_key = key + "_sha256"
    if key.endswith("_path"):
        hash_key = key.replace("_path", "_sha256")

    digest = info.get(hash_key)
    if not digest:
        raise ValueError(f"no hash recorded for capture artifact: {key}")

    path = resolve_artifact_path(info, key, manifest_dir, run_dir=run_dir, verify_hash=False)
    if path is None or not path.is_file():
        raise ValueError(f"missing capture artifact: {key} (resolved to {path})")

    if sha256(path) != digest:
        raise ValueError(f"capture artifact changed: {key} ({path})")

    return path


def resolve_graph_dir(
    manifest: dict[str, Any],
    manifest_dir: Path,
) -> Path | None:
    """Resolve the backend graph directory from a manifest."""
    backend = manifest.get("backend") or {}
    data_dir = backend.get("data_dir")
    if not data_dir:
        return None

    run_dir = manifest.get("run_dir")
    relpath = backend.get("data_dir_relpath")
    if relpath:
        candidate = manifest_dir / relpath
        if candidate.is_dir() and (candidate / "CURRENT").is_file():
            return candidate

    path = resolve_path(data_dir, manifest_dir, legacy_run_dir=run_dir)
    if path and path.is_dir() and (path / "CURRENT").is_file():
        return path

    if run_dir and not is_relative_path(data_dir):
        original = Path(data_dir)
        try:
            rel = original.relative_to(Path(run_dir))
            candidate = manifest_dir / rel
            if candidate.is_dir() and (candidate / "CURRENT").is_file():
                return candidate
        except ValueError:
            pass

    return Path(data_dir) if data_dir else None


def normalize_cell_entry(
    entry: dict,
    previous_run_dir: str | None,
    new_manifest_dir: Path,
) -> dict:
    """Normalize a retained cell entry's paths from a previous run to the current directory.

    When resuming a moved run, legacy cells may have absolute paths that are now invalid.
    This function rebases them to be relative to the new manifest directory.

    Args:
        entry: The cell entry dict (mutated and returned).
        previous_run_dir: The run_dir from the previous manifest (the original absolute path
            or '.' if already relative).
        new_manifest_dir: The directory where the new manifest will be written.

    Returns:
        The entry with normalized paths.
    """
    ARTIFACT_KEYS = ("policy", "functors", "sdk_log", "server_replay")

    # Determine the base for rebasing absolute paths
    if previous_run_dir and previous_run_dir != ".":
        original_root = Path(previous_run_dir)
    else:
        original_root = None

    # Normalize sim_path
    sim_path = entry.get("sim_path")
    if sim_path and sim_path.startswith("/"):
        path = Path(sim_path)
        # Try to rebase against original run_dir
        if original_root:
            try:
                rel = path.relative_to(original_root)
                if (new_manifest_dir / rel).is_file():
                    entry["sim_path"] = str(rel)
            except ValueError:
                pass
        # Fallback: if file exists under new manifest_dir with same basename structure
        if entry.get("sim_path", "").startswith("/"):
            # Try to detect cell directory pattern
            sim_name = path.name  # results.json
            cell_name = path.parent.name  # airline_enforce-on_gpt
            if (new_manifest_dir / cell_name / sim_name).exists():
                entry["sim_path"] = f"{cell_name}/{sim_name}"

    # Remove redundant sim_relpath if we've made sim_path relative
    if entry.get("sim_path") and not entry["sim_path"].startswith("/"):
        entry.pop("sim_relpath", None)

    # Normalize latency_capture paths
    capture = entry.get("latency_capture")
    if capture:
        # Determine cell directory for capture artifacts
        cell_dir: Path | None = None
        sim_path_val = entry.get("sim_path")
        if sim_path_val:
            if not sim_path_val.startswith("/"):
                cell_dir = (new_manifest_dir / sim_path_val).parent
            elif original_root:
                try:
                    rel = Path(sim_path_val).relative_to(original_root)
                    cell_dir = (new_manifest_dir / rel).parent
                except ValueError:
                    pass

        if cell_dir is None:
            # Fallback: derive from cell id
            cell_id = entry.get("id", "")
            if cell_id.startswith("tau2/"):
                cell_name = cell_id.replace("tau2/", "").replace("/", "_")
                cell_dir = new_manifest_dir / cell_name

        for key in ARTIFACT_KEYS:
            value = capture.get(key)
            if not value:
                continue

            if value.startswith("/"):
                path = Path(value)
                # Try to rebase via original run_dir
                if original_root:
                    try:
                        rel = path.relative_to(original_root)
                        # Make relative to run manifest (cell_dir/artifact -> cell/artifact)
                        if (new_manifest_dir / rel).is_file():
                            capture[key] = str(rel)
                            continue
                    except ValueError:
                        pass

                # Try to find the file under the cell_dir
                if cell_dir:
                    artifact_name = path.name
                    parent_name = path.parent.name
                    # Check if it's a replay-policy/artifact pattern
                    if parent_name in ("replay-policy",):
                        candidate = cell_dir / parent_name / artifact_name
                        if candidate.exists():
                            try:
                                capture[key] = str(candidate.relative_to(new_manifest_dir))
                            except ValueError:
                                pass
                            continue
                    # Direct cell child (latency.jsonl, server_replay.jsonl)
                    candidate = cell_dir / artifact_name
                    if candidate.exists():
                        try:
                            capture[key] = str(candidate.relative_to(new_manifest_dir))
                        except ValueError:
                            pass

            # Remove redundant _relpath fields
            capture.pop(key + "_relpath", None)

    # Recursively normalize previous_attempt if present (outside capture block)
    prev = entry.get("previous_attempt")
    if prev and isinstance(prev, dict):
        normalize_cell_entry(prev, previous_run_dir, new_manifest_dir)

    return entry


def resolve_cell_dir(cell: dict, manifest_dir: Path, *, run_dir: str | None = None) -> Path:
    """Determine a cell's output directory for per-cell artifact resolution.

    Per-cell capture paths (policy, functors, sdk_log, server_replay) are
    relative to the cell's directory, not the run manifest.

    For moved runs, local copies (relative paths, rebased absolute paths)
    are preferred over the original absolute path even when the original
    still exists. This ensures replay uses the selected run's artifacts.
    """
    sim_value = cell.get("sim_relpath") or cell.get("sim_path")
    if sim_value:
        if is_relative_path(sim_value):
            return (manifest_dir / sim_value).parent

        sim_path = Path(sim_value)
        if run_dir:
            try:
                rel = sim_path.relative_to(Path(run_dir))
                candidate = manifest_dir / rel.parent
                if candidate.is_dir():
                    return candidate
            except ValueError:
                pass

        if sim_path.parent.is_dir():
            return sim_path.parent

    cell_id = cell.get("id", "")
    if cell_id.startswith("tau2/"):
        cell_name = cell_id.replace("tau2/", "").replace("/", "_")
        candidate = manifest_dir / cell_name
        if candidate.is_dir():
            return candidate

    return manifest_dir
