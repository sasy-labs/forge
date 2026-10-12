"""Append-only latency capture for the paper runner, including interrupted attempts.

Per-cell capture files store paths relative to the cell directory (their own
manifest base), not the parent run manifest. When reading, resolve paths against
the capture file's directory, not the run's manifest directory.

New manifests store internal artifact paths as RELATIVE values in the primary
field (e.g., ``"policy": "replay-policy/policy.dl"``). External resources and
operational argv/env remain absolute for provenance. Readers resolve relative
paths against the containing directory; legacy absolute paths are rebased using
the manifest's ``run_dir`` if the original no longer exists.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    """Replace a manifest atomically so interruption cannot leave half a JSON file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def file_size(path: Path | None) -> int:
    return path.stat().st_size if path and path.exists() else 0


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _relative_str(path: Path, base: Path) -> str:
    """Relative path string for internal artifacts."""
    return str(path.relative_to(base))


def snapshot_policy(source: Path, out_dir: Path) -> dict:
    """Save exactly the source/includes and companion functors the SDK uploads.

    Returns paths relative to out_dir for portability. External policy sources
    are not copied; this snapshots the resolved+compiled artifacts.
    """
    from sasy.policy import find_functors, resolve_includes

    policy = resolve_includes(source).encode()
    functors = find_functors(source).encode()
    assets = out_dir / "replay-policy"
    policy_path = assets / "policy.dl"
    functors_path = assets / "functors.cpp"

    info = {
        "policy": _relative_str(policy_path, out_dir),
        "policy_sha256": hashlib.sha256(policy).hexdigest(),
        "functors": _relative_str(functors_path, out_dir),
        "functors_sha256": hashlib.sha256(functors).hexdigest(),
    }

    if assets.exists():
        for key, actual in (("policy", policy_path), ("functors", functors_path)):
            if sha256(actual) != info[key + "_sha256"]:
                raise ValueError(f"policy changed since capture for {out_dir}; start a new run")
    else:
        assets.mkdir()
        policy_path.write_bytes(policy)
        functors_path.write_bytes(functors)

    return info


class CaptureAttempt:
    """Capture just this step's server-log suffix, without truncating the live log.

    Paths are stored relative to out_dir for portability. When reading a moved
    run, resolve paths against the manifest's containing directory.
    """

    def __init__(self, out_dir: Path, server_log: Path | None, policy: Path,
                 engine_sha256: str | None = None, concurrency: int | None = None):
        self.out_dir = out_dir
        self.engine_sha256 = engine_sha256
        self.concurrency = concurrency
        self.server_log = server_log
        self.sdk_log = out_dir / "latency.jsonl"
        self.replay = out_dir / "server_replay.jsonl"
        self.path = out_dir / "latency-capture.json"
        if self.path.exists():
            self.info = json.loads(self.path.read_text())
            # Normalize any legacy absolute paths to relative
            self._normalize_loaded_paths()
        else:
            self.info = {
                "version": 1,
                "sdk_log": _relative_str(self.sdk_log, out_dir),
                "server_replay": _relative_str(self.replay, out_dir),
                "attempts": [],
            }
        self.info.update(snapshot_policy(policy, out_dir))
        self.sdk_start = file_size(self.sdk_log)
        self.server_start = file_size(server_log)

    def _normalize_loaded_paths(self) -> None:
        """Normalize any absolute paths in loaded capture to relative paths.

        Old captures may have absolute paths that need to be rebased when the
        run has been moved. Convert them to be relative to out_dir.
        """
        ARTIFACT_KEYS = ("sdk_log", "server_replay", "policy", "functors")
        for key in ARTIFACT_KEYS:
            value = self.info.get(key)
            if not value:
                continue
            if value.startswith("/"):
                path = Path(value)
                # Check if the file exists at the absolute path or under out_dir
                if path.is_file():
                    # Try to make relative to out_dir
                    try:
                        rel = path.relative_to(self.out_dir)
                        self.info[key] = str(rel)
                        continue
                    except ValueError:
                        pass
                # Check if it's under a parent of out_dir (cell within run)
                artifact_name = path.name
                parent_name = path.parent.name
                if parent_name == "replay-policy":
                    candidate = self.out_dir / "replay-policy" / artifact_name
                    if candidate.is_file():
                        self.info[key] = f"replay-policy/{artifact_name}"
                        continue
                # Try the artifact directly under out_dir
                candidate = self.out_dir / artifact_name
                if candidate.is_file():
                    self.info[key] = artifact_name
            # Remove redundant _relpath fields
            self.info.pop(key + "_relpath", None)

    def finish(self, returncode: int, wall_s: float) -> dict:
        replay_start = file_size(self.replay)
        server_end = file_size(self.server_log)
        if server_end < self.server_start:
            raise ValueError("server latency log was truncated during capture")
        if self.server_log and server_end > self.server_start:
            with self.server_log.open("rb") as src, self.replay.open("ab") as dst:
                src.seek(self.server_start)
                remaining = server_end - self.server_start
                while remaining:
                    chunk = src.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError("server latency log shortened during capture")
                    dst.write(chunk)
                    remaining -= len(chunk)
        self.info["attempts"].append({
            "returncode": returncode, "wall_s": wall_s, "engine_sha256": self.engine_sha256,
            "concurrency": self.concurrency,
            "sdk_start": self.sdk_start, "sdk_end": file_size(self.sdk_log),
            "server_start": self.server_start, "server_end": server_end,
            "replay_start": replay_start, "replay_end": file_size(self.replay),
        })
        for key, path in (("sdk_log", self.sdk_log), ("server_replay", self.replay)):
            self.info[key + "_sha256"] = sha256(path) if path.exists() else None
        write_json(self.path, self.info)
        return self.info


def copy_closed_graph(source: Path, destination: Path) -> None:
    """Copy a stopped RocksDB database while holding its POSIX writer lock."""
    import fcntl

    lock = source / "LOCK"
    if not lock.is_file() or not (source / "CURRENT").is_file():
        raise ValueError(f"not a saved engine database: {source}")
    with lock.open("r+") as stream:
        try:
            fcntl.lockf(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("source engine is still running; stop the benchmark before replay") from exc
        try:
            shutil.copytree(source, destination)
        finally:
            fcntl.lockf(stream, fcntl.LOCK_UN)
