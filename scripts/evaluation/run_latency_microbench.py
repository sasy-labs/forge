"""Measure authorization latency and throughput with live runs or saved-request replay.

Live runs capture per-call enforcement timings and task durations. Replay
measures authorization against a saved graph, excluding model calls, tool
execution and graph updates.

Usage::

    # Preferred: replay a stopped main benchmark, with no new model calls.
    uv run python scripts/evaluation/run_latency_microbench.py \\
        --from-benchmark results/<timestamp> \\
        --domains airline retail --model gpt --source-mode enforce-on

    # Instrumented arm, default concurrency sweep.
    uv run python scripts/evaluation/run_latency_microbench.py

    # Single cell, dry-run (prints commands):
    uv run python scripts/evaluation/run_latency_microbench.py \\
        --domains retail --modes sasy --concurrencies 8 --dry-run

    # Reduce already-captured logs to a table without re-running:
    uv run python scripts/evaluation/run_latency_microbench.py \\
        --reduce-only results/latency-microbench/2026-04-29T16-00-00

The script is intentionally a thin wrapper around
``run_benchmark.py``: each (mode, concurrency) cell becomes a
single ``run_benchmark.py`` invocation with its own
``SASY_LATENCY_LOG_FILE``. This means existing flags
(``--cell-timeout``, etc.) propagate
unchanged via the wrapper's ``--passthrough`` argument.

Trial budget defaults to max(8, 2 × concurrency) trials per curated task.
Set ``--num-trials 1`` for a quick sanity run. Previously captured logs can be summarized with ``--reduce-only``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_BENCH = REPO_ROOT / "scripts" / "evaluation" / "run_benchmark.py"
LOADGEN_BIN = REPO_ROOT / "latency" / "loadgen" / "target" / "debug" / "sasy-loadgen"


def _load_dotenv(target: dict[str, str], path: Path) -> None:
    """``target.setdefault`` every ``KEY=VALUE`` from ``path``.

    Existing values in ``target`` win — explicit shell exports and
    operator overrides take precedence over file defaults. Quotes
    on the value are stripped so the file matches what shells
    produce. Comment / blank lines are ignored.
    """
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


# Apply ``.env`` defaults to this process's env so subsequent
# subprocess.Popen calls inherit them. Order matters: ``.env``
# (general settings, including SASY_KEEPALIVE_INTERVAL_MS=250)
# wins over ``.env-credentials`` for keys defined in both, but
# explicit shell exports still beat both.
_load_dotenv(os.environ, REPO_ROOT / ".env")
_load_dotenv(os.environ, REPO_ROOT / ".env-credentials")
SERVER_BIN = REPO_ROOT / "sasy" / "target" / "debug" / "sasy"
SERVER_TLS_CERT = REPO_ROOT / "certs" / "reference-monitor.crt"
SERVER_TLS_KEY = REPO_ROOT / "certs" / "reference-monitor.key"
SERVER_TLS_CA = REPO_ROOT / "certs" / "ca.crt"

# Per-domain policy files. Override via ``--server-policy
# domain=path`` on the command line for non-tau2 domains.
DEFAULT_POLICY_FILES = {
    "airline": REPO_ROOT / "experiments" / "tau2" / "airline_policy.dl",
    "retail":  REPO_ROOT / "experiments" / "tau2" / "retail_policy.dl",
}

# Default live configuration: a single Souffle worker and concurrency 1–32.
# Each cell uses --user-model match, selecting the agent model as user simulator.
DEFAULT_MODEL = "gpt"
DEFAULT_DOMAINS = ("airline",)
DEFAULT_MODES = ("enforce-on",)
DEFAULT_CONCURRENCIES = (1, 2, 4, 8, 16, 32)
# Sentinel for ``--num-trials``: when None, derive per-cell as
# ``max(8, 2 * concurrency)``. The minimum supplies samples at low
# concurrency; scaling adds multiple batches per task at higher concurrency.
DEFAULT_NUM_TRIALS: int | None = None


def _trials_for(concurrency: int, override: int | None) -> int:
    if override is not None:
        return override
    return max(8, 2 * concurrency)


# ── Per-cell driver ───────────────────────────────────────


@dataclass
class CellResult:
    domain: str
    mode: str
    concurrency: int | None
    returncode: int
    wall_clock_s: float
    sim_path: Optional[Path] = None
    latency_log: Optional[Path] = None
    judge_log: Optional[Path] = None
    source_cell_id: str = ""
    source_mode: str = ""
    agent_llm: str = ""
    user_llm: str = ""
    measurement_kind: str = "live"
    capture_concurrencies: str = ""
    # Filled in by reduce step. Latency is the wall RTT the agent
    # waits for an enforcement decision: gRPC RTT for sasy/baseline
    # cells (sourced from SASY_LATENCY_LOG_FILE), elapsed_s × 1e6
    # for judge cells (sourced from the cli_llm_policy judge log).
    num_auth_calls: int = 0
    auth_p50_us: float = float("nan")
    auth_mean_us: float = float("nan")
    auth_p95_us: float = float("nan")
    auth_p99_us: float = float("nan")
    auth_max_us: float = float("nan")
    # Judge-only cost/token aggregates. NaN for non-judge cells.
    judge_total_cost_usd: float = float("nan")
    judge_mean_cost_usd: float = float("nan")
    judge_mean_tokens_in: float = float("nan")
    judge_mean_tokens_out: float = float("nan")
    judge_mean_llm_calls: float = float("nan")
    # End-to-end task throughput from the simulation JSON.
    num_tasks: int = 0
    throughput_tasks_per_s: float = float("nan")
    # Mean and stdev of per-task wall-clock seconds (from the
    # tau2 sim JSON's ``duration`` field). The stdev captures the
    # tail-task variance that drives ``wall_clock_s`` more than
    # ``num_tasks/concurrency`` does — useful for the paper's
    # "did this run get hit by an outlier task?" diagnostic.
    task_dur_mean_s: float = float("nan")
    task_dur_std_s: float = float("nan")
    # Server-side latency captured during the cell — the
    # ``total_us`` field from each ``server_replay.jsonl`` record.
    # This is the actual time the engine spent serving each
    # concurrent call during the real (tau2-driven) workload, with
    # all the queueing / IPC costs that the cell exercised. Sits
    # between the SDK row (which adds GIL + wire) above and the
    # loadgen row (saturated synthetic) below.
    server_replay_log: Optional[Path] = None
    server_n: int = 0
    server_p50_us: float = float("nan")
    server_mean_us: float = float("nan")
    server_p95_us: float = float("nan")
    server_p99_us: float = float("nan")
    server_max_us: float = float("nan")
    # Rust-loadgen replay (filled when --loadgen is enabled). The
    # loadgen issues the same CheckToolCall RPCs the cell observed,
    # at the cell's concurrency, against the warm server — so its
    # per-call RTT measures engine + wire only (no Python SDK, no
    # GIL, no agent-time pacing). ``auth_throughput`` is the rate
    # the engine sustained: total replayed calls / loadgen wall.
    #
    # When ``--loadgen-trials > 1``, the loadgen runs N independent
    # replays and the reducer aggregates them: latency percentiles
    # come from the combined record set; throughput mean/std come
    # from per-trial throughput values. Within-distribution stdev
    # on the mean column is intentionally not reported — p50 / p99
    # already characterize shape.
    loadgen_log: Optional[Path] = None
    loadgen_n: int = 0
    loadgen_n_trials: int = 0
    loadgen_p50_us: float = float("nan")
    loadgen_mean_us: float = float("nan")
    loadgen_p95_us: float = float("nan")
    loadgen_p99_us: float = float("nan")
    loadgen_max_us: float = float("nan")
    loadgen_wall_s: float = float("nan")
    auth_throughput_calls_per_s: float = float("nan")
    auth_throughput_std: float = float("nan")


@dataclass
class ServerHandle:
    """Live handle to a managed sasy server subprocess."""
    proc: subprocess.Popen
    addr: str
    server_log_path: Path
    server_log_fp: object  # text file handle for stdout/stderr capture
    data_dir: Path
    credentials_db: Path
    policy_path: Path
    latency_log_path: Path

    def stop(self, timeout: float = 10.0) -> None:
        """Send SIGTERM and wait. Falls back to SIGKILL on timeout."""
        if self.proc.poll() is not None:
            self.server_log_fp.close()
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()
        try:
            self.server_log_fp.close()
        except OSError:
            pass


def _wait_for_port(host: str, port: int, timeout: float = 60.0) -> bool:
    """Poll a TCP port until something accepts a connection. The
    server's gRPC bind is the last init step before serving, so a
    successful connect means readiness for our purposes.
    """
    import socket
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _spawn_server(
    *,
    domain: str,
    run_root: Path,
    addr: str,
    policy_path: Path,
    server_bin: Path,
    extra_env: dict[str, str],
    keepalive_ms: int,
    ready_timeout: float,
) -> ServerHandle:
    """Spawn a sasy server for ``domain`` with a per-run data dir
    and the latency log routed under ``run_root``. Returns a handle
    the caller is responsible for stopping.
    """
    if not server_bin.exists():
        raise FileNotFoundError(
            f"sasy server binary not found at {server_bin}. "
            "Build with `cargo build --release -p sasy-binary` first."
        )
    if not policy_path.exists():
        raise FileNotFoundError(
            f"policy file not found for domain={domain}: {policy_path}"
        )

    host, _, port_str = addr.rpartition(":")
    if host in ("0.0.0.0", "[::]", ""):
        host = "127.0.0.1"
    try:
        port = int(port_str)
    except ValueError as exc:
        raise ValueError(f"could not parse port from --server-addr {addr!r}") from exc
    try:
        connection = socket.create_connection((host, port), timeout=0.5)
    except OSError:
        pass
    else:
        connection.close()
        raise RuntimeError(f"server port {port} is already in use; choose --server-addr")

    server_root = run_root / "server" / domain
    server_root.mkdir(parents=True, exist_ok=True)
    data_dir = server_root / "graph"
    credentials_db = server_root / "credentials.db"
    latency_log = server_root / "server_latency.jsonl"
    server_log = server_root / "server.log"

    # Truncate so each managed run starts clean. The cell-level
    # rotation in ``_run_cell`` reads/clears this same file per cell.
    latency_log.write_bytes(b"")

    cmd = [
        str(server_bin), "serve",
        "--evaluator", "souffle",
        "--auth-provider", str(REPO_ROOT / "config" / "auth" / "all.json"),
        "--auth-config", str(REPO_ROOT / "config" / "auth_config.yaml"),
        "--transforms", str(REPO_ROOT / "config" / "transforms.json"),
        "--tls-cert", str(SERVER_TLS_CERT),
        "--tls-key", str(SERVER_TLS_KEY),
        "--tls-ca", str(SERVER_TLS_CA),
        "--addr", addr,
        "--data-dir", str(data_dir),
        "--credentials-db", str(credentials_db),
        "--policy", str(policy_path),
        # Development host only. The tau2 cells this server drives
        # bind their domain policy per session, and the Python SDK
        # attaches the companion `<policy>_functors.cpp` next to it
        # (airline_functors.cpp, retail_functors.cpp). The cell
        # credential is not an admin, so without this opt-in every
        # bind is refused with PERMISSION_DENIED. The unsandboxed
        # value is used because a benchmark host is typically a Mac,
        # which has no bubblewrap. Do not copy into a deployment: it
        # lets a non-admin caller compile and run C++ unconfined.
        "--allow-user-functors=unsandboxed",
    ]

    env = os.environ.copy()
    # Re-apply env files in case the operator's shell didn't have
    # them sourced. Module-level apply already populated
    # ``os.environ``; this is belt-and-suspenders for the spawned
    # server in case the caller passed an ``env`` dict directly.
    _load_dotenv(env, REPO_ROOT / ".env")
    _load_dotenv(env, REPO_ROOT / ".env-credentials")
    env.update(extra_env)
    env["SASY_SERVER_LATENCY_LOG_FILE"] = str(latency_log)
    env.setdefault("LLM_ENABLED", "true")
    # Cap server tokio workers so the loadgen + server combined stay
    # within the host's logical-core count (avoids the work-stealing
    # oversubscription tax we measured on a 14-core box). LG=4 SRV=10
    # was the empirical sweet spot at c=8/16/32. Operator's
    # ``--server-env TOKIO_WORKER_THREADS=...`` wins via ``extra_env``
    # already merged above; this is the fallback.
    env.setdefault("TOKIO_WORKER_THREADS", "10")
    # Operator's --server-keepalive-ms always wins over .env's value;
    # ``0`` means "explicitly disable", which the server interprets via
    # ``keepalive_interval.is_zero()`` (see session_evaluator.rs).
    if keepalive_ms >= 0:
        env["SASY_KEEPALIVE_INTERVAL_MS"] = str(keepalive_ms)

    print(f"\n>>> spawning server for domain={domain}")
    print(f"    {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"    SASY_SERVER_LATENCY_LOG_FILE={(latency_log.relative_to(REPO_ROOT) if latency_log.is_relative_to(REPO_ROOT) else latency_log)}")
    print(f"    server.log={(server_log.relative_to(REPO_ROOT) if server_log.is_relative_to(REPO_ROOT) else server_log)}")

    log_fp = server_log.open("w")
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=REPO_ROOT,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    handle = ServerHandle(
        proc=proc,
        addr=addr,
        server_log_path=server_log,
        server_log_fp=log_fp,
        data_dir=data_dir,
        credentials_db=credentials_db,
        policy_path=policy_path,
        latency_log_path=latency_log,
    )
    try:
        if not _wait_for_port(host, port, timeout=ready_timeout) or proc.poll() is not None:
            log_fp.flush()
            try:
                tail = "\n".join(server_log.read_text().splitlines()[-20:])
            except OSError:
                tail = "(log unreadable)"
            raise RuntimeError(
                f"server didn't become ready on {host}:{port} within "
                f"{ready_timeout}s — last log lines:\n{tail}"
            )
    except BaseException:
        handle.stop()
        raise
    print(f"    ready on {host}:{port}")
    return handle


def _run_loadgen(
    *,
    bin_path: Path,
    replay: Path,
    out: Path,
    url: str,
    ca: Optional[Path],
    cert: Optional[Path],
    key: Optional[Path],
    api_key: Optional[str],
    concurrency: int,
    channels: int,
    repeat: int,
    timeout: int,
    mode: str = "session",
) -> tuple[int, float]:
    """Run the Rust loadgen against ``replay``. Returns
    ``(returncode, wall_seconds)``. Skips with a warning when the
    binary or replay file is missing."""
    if not bin_path.exists():
        print(f"  ! loadgen binary not found at {bin_path}; skipping")
        return (-1, float("nan"))
    if not replay.exists() or replay.stat().st_size == 0:
        print(f"  ! replay {replay} empty or missing; skipping loadgen")
        return (-1, float("nan"))
    cmd: list[str] = [
        str(bin_path),
        "--replay", str(replay),
        "--out", str(out),
        "--url", url,
        "--mode", mode,
        "--concurrency", str(concurrency),
        "--channels", str(max(1, channels)),
        "--repeat", str(repeat),
    ]
    if ca: cmd += ["--ca", str(ca)]
    if cert: cmd += ["--cert", str(cert)]
    if key: cmd += ["--key", str(key)]
    if api_key: cmd += ["--api-key", api_key]
    display_cmd = cmd.copy()
    if api_key:
        display_cmd[-1] = "<redacted>"
    print(f"    loadgen: {' '.join(shlex.quote(c) for c in display_cmd)}")
    # Cap loadgen tokio workers for the same reason we cap the server
    # (see ``_spawn_server``). Operator's environment value wins.
    lg_env = os.environ.copy()
    lg_env.setdefault("TOKIO_WORKER_THREADS", "4")
    t0 = time.monotonic()
    try:
        rc = subprocess.run(
            cmd, cwd=REPO_ROOT, timeout=timeout, env=lg_env,
        ).returncode
    except subprocess.TimeoutExpired:
        print(f"  ! loadgen timed out after {timeout}s")
        return (-1, time.monotonic() - t0)
    return (rc, time.monotonic() - t0)


def _run_cell(
    out_root: Path,
    domain: str,
    mode: str,
    model: str,
    concurrency: int,
    num_trials: int,
    cell_timeout: int,
    passthrough: list[str],
    dry_run: bool,
    server_log: Optional[Path] = None,
    loadgen_cfg: Optional[dict] = None,
    server_url: Optional[str] = None,
) -> CellResult:
    """Drive one (domain, mode, concurrency) cell via run_benchmark.py."""
    cell_root = out_root / f"{domain}_{mode}_c{concurrency}"
    cell_root.mkdir(parents=True, exist_ok=True)

    # run_benchmark.py expects --output-root and writes per-cell
    # subdirectories under <root>/<timestamp>/<domain>_<mode>_<model>.
    # We pre-create a stable per-cell directory and pass it as the
    # output root so all per-mode tau2 simulation files land where
    # we can find them deterministically.
    sub_output_root = cell_root / "rb_out"
    sub_output_root.mkdir(parents=True, exist_ok=True)

    latency_log = cell_root / "latency.jsonl"
    # Truncate any previous run so the reducer doesn't see stale
    # records. JSONL is append-only by design.
    if latency_log.exists():
        latency_log.unlink()

    cmd = [
        sys.executable, str(RUN_BENCH),
        # One tau2 cell on the curated tasks, with the agent's own model as
        # the simulated user; no warm-up trial, so the latency log holds
        # only the measured cell.
        "--targets", "tau2",
        "--task-subset", "curated",
        "--user-model", "match",
        "--no-warmup",
        "--no-capture-latency",
        "--domain", domain,
        "--mode", mode,
        "--model", model,
        "--num-trials", str(num_trials),
        "--max-concurrency", str(concurrency),
        "--output-root", str(sub_output_root),
        "--cell-timeout", str(cell_timeout),
        "--continue-on-error",
    ]
    # Never let run_benchmark.py start its own engine (its default):
    # the cell's auth traffic must reach the server whose latency log
    # we capture. With --manage-server that is the microbench's server
    # (SASY_URL set below); with --no-manage-server it is the external
    # engine the caller's SASY_URL already names.
    cmd.append("--no-start-server")
    cmd += passthrough

    env = os.environ.copy()
    env["SASY_LATENCY_LOG_FILE"] = str(latency_log)
    if server_url is not None:
        # Route the SDK at the microbench's own server.
        env["SASY_URL"] = server_url

    print(f"\n>>> {domain} / {mode} / c={concurrency}")
    print(f"    {' '.join(shlex.quote(c) for c in cmd)}")
    try:
        latency_disp = latency_log.relative_to(REPO_ROOT)
    except ValueError:
        latency_disp = latency_log
    print(f"    SASY_LATENCY_LOG_FILE={latency_disp}")

    # Rotate the server-side latency log before the cell runs so
    # whatever the next ``CheckAuthorization`` writes lands in an
    # otherwise-empty file we can capture as the cell's replay.
    server_replay = cell_root / "server_replay.jsonl"
    if server_log is not None and not dry_run:
        # Truncate eagerly so any in-flight records from a previous
        # cell don't leak into this cell's replay capture.
        server_log.write_bytes(b"")

    if dry_run:
        return CellResult(
            domain=domain, mode=mode, concurrency=concurrency,
            returncode=0, wall_clock_s=0.0,
            latency_log=latency_log,
        )

    # Spawn the cell in its own process group so a Ctrl+C in the
    # outer wrapper kills the entire run_benchmark.py + tau2 tree,
    # not just the wrapper. ``subprocess.run`` left the child
    # sharing our group, which meant SIGINT arrived in both
    # processes simultaneously and the wrapper's loop sometimes
    # continued to the next (domain, mode, conc) cell with
    # orphaned tau2 workers from the previous one still alive.
    # Mirrors the pattern in run_benchmark.py:_run_cell.
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, env=env, cwd=REPO_ROOT, start_new_session=True)
    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        print(f"  ! interrupted; killing process group {proc.pid}")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        except ProcessLookupError:
            pass
        raise
    except BaseException:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        raise
    wall = time.monotonic() - t0

    # run_benchmark.py only emits one cell because we passed
    # --domain/--mode/--model singletons. Locate the manifest
    # and pull the sim path out of it.
    sim_path: Optional[Path] = None
    manifests = list(sub_output_root.glob("*/manifest.json"))
    if manifests:
        manifest = json.loads(manifests[-1].read_text())
        cells = manifest.get("cells") or []
        if cells and cells[0].get("sim_path"):
            sim_path = Path(cells[0]["sim_path"])

    # Capture the server-side replay log. Truncation happened at the
    # start of this cell, so the file now contains exactly the
    # records this cell produced.
    loadgen_log: Optional[Path] = None
    loadgen_wall = float("nan")
    if server_log is not None:
        try:
            data = server_log.read_bytes()
        except OSError as e:
            print(f"  ! could not read server log {server_log}: {e}")
            data = b""
        server_replay.write_bytes(data)

    if loadgen_cfg is not None and mode in {"sasy", "enforce-on"} and not dry_run:
        n_trials = max(1, loadgen_cfg.get("trials", 1))
        # Clean up any prior trial files from a previous run.
        for old in cell_root.glob("loadgen-trial-*.jsonl"):
            old.unlink()
        primary = cell_root / "loadgen.jsonl"
        if primary.exists():
            primary.unlink()
        all_ok = True
        for i in range(1, n_trials + 1):
            trial_out = (
                cell_root / "loadgen.jsonl" if n_trials == 1
                else cell_root / f"loadgen-trial-{i}.jsonl"
            )
            print(f"    loadgen trial {i}/{n_trials} → {trial_out.name}")
            rc_lg, loadgen_wall = _run_loadgen(
                bin_path=loadgen_cfg["bin_path"],
                replay=server_replay,
                out=trial_out,
                url=loadgen_cfg["url"],
                ca=loadgen_cfg.get("ca"),
                cert=loadgen_cfg.get("cert"),
                key=loadgen_cfg.get("key"),
                api_key=loadgen_cfg.get("api_key"),
                concurrency=concurrency,
                channels=loadgen_cfg.get("channels", concurrency),
                repeat=loadgen_cfg.get("repeat", 1),
                timeout=loadgen_cfg.get("timeout", 1800),
                mode=loadgen_cfg.get("mode", "session"),
            )
            if rc_lg == 0 and trial_out.exists():
                if loadgen_log is None:
                    loadgen_log = trial_out  # primary handle for the reducer
            else:
                all_ok = False
                print(f"  ! loadgen trial {i} failed (rc={rc_lg}, output={trial_out.exists()})")
        if not all_ok:
            rc = rc or 1

    # The captured server replay doubles as the "actual server-side
    # latency under the cell's real workload" data source. Set it
    # whenever the file has any records (server was managed or the
    # operator pointed --server-latency-log at a writable path).
    server_replay_log_field: Optional[Path] = (
        server_replay if server_log is not None and server_replay.exists()
        and server_replay.stat().st_size > 0 else None
    )

    return CellResult(
        domain=domain, mode=mode, concurrency=concurrency,
        returncode=rc, wall_clock_s=wall,
        sim_path=sim_path, latency_log=latency_log,
        judge_log=_judge_log_for(sim_path) if sim_path else None,
        server_replay_log=server_replay_log_field,
        loadgen_log=loadgen_log,
        loadgen_wall_s=loadgen_wall,
    )


def _judge_log_for(sim_path: Path) -> Path:
    """The judge log run_benchmark.py wrote for a tau2 run.

    ``<run>/judge_log.jsonl`` inside the run directory, or beside the
    run as ``<run>.judge_log.jsonl`` for older runs (see
    ``tau2_results.judge_log_for``).
    """
    from tau2_results import JUDGE_LOG_NAME, judge_log_for, run_base

    return judge_log_for(sim_path) or run_base(sim_path) / JUDGE_LOG_NAME


# ── Reducer ───────────────────────────────────────────────


def _percentile(sorted_xs: list[float], p: float) -> float:
    """Nearest-rank percentile, p in [0, 1]. Returns NaN on empty."""
    if not sorted_xs:
        return float("nan")
    k = max(0, min(len(sorted_xs) - 1, math.ceil(p * len(sorted_xs)) - 1))
    return sorted_xs[k]


def _read_latencies(path: Path) -> list[float]:
    """Pull the wall RTT (us) field out of the JSONL log."""
    if not path or not path.exists():
        return []
    out: list[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rtt = rec.get("rtt_us")
            if rtt is not None and "error" not in rec:
                out.append(float(rtt))
    return out


def _read_server_total_us(path: Path) -> list[float]:
    """Pull ``total_us`` (server-side per-call wall time) from a
    ``server_replay.jsonl`` written by the engine. Records have a
    different shape from the SDK ``latency.jsonl`` — see
    ``sasy_policy::latency_log::ActionRecord``.
    """
    if not path or not path.exists():
        return []
    out: list[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = rec.get("total_us")
            if v is not None:
                out.append(float(v))
    return out


def _read_judge_log(path: Path) -> tuple[list[float], list[dict]]:
    """Return ``(elapsed_us, usage_records)`` for cli_llm_policy logs.

    Each successful record has ``elapsed_s`` (judge wall time, end
    to end including graph query + LLM call + verdict parse) and
    ``verdict.usage`` with cost_usd / tokens_in / tokens_out /
    n_llm_calls. Records with ``error`` are excluded — those are
    judge failures, not normal verdicts, and would skew p99 the
    same way a 30s API timeout would skew RTT.
    """
    if not path or not path.exists():
        return [], []
    elapsed_us: list[float] = []
    usages: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error"):
                continue
            es = rec.get("elapsed_s")
            if es is not None:
                elapsed_us.append(float(es) * 1_000_000)
            verdict = rec.get("verdict") or {}
            usage = verdict.get("usage") or {}
            if usage:
                usages.append(usage)
    return elapsed_us, usages


def _read_throughput(
    sim_path: Optional[Path],
) -> tuple[int, float, float, float]:
    """Return ``(num_tasks_completed, mean_dur_s, std_dur_s,
    sum_dur_s)``.

    tau2 simulations record one entry per task with a ``duration``
    field. Throughput is ``num_tasks / wall_clock_s`` at the
    caller; the per-task ``mean`` / ``std`` lets us flag cells
    where wall time was bottlenecked by a long-tail task rather
    than by total work.
    """
    if not sim_path or not sim_path.exists():
        return 0, float("nan"), float("nan"), float("nan")
    try:
        data = json.loads(sim_path.read_text())
    except (json.JSONDecodeError, OSError):
        return 0, float("nan"), float("nan"), float("nan")
    sims = data.get("simulations") or []
    durations = [
        float(s.get("duration"))
        for s in sims
        if s.get("duration") is not None
    ]
    if not durations:
        return len(sims), float("nan"), float("nan"), float("nan")
    mean_d = statistics.mean(durations)
    std_d = statistics.stdev(durations) if len(durations) > 1 else 0.0
    return len(sims), mean_d, std_d, sum(durations)


def _reduce_cell(cell: CellResult) -> CellResult:
    """Populate latency/throughput stats on a CellResult in place.

    Judge cells source their latency from the cli_llm_policy log
    (``elapsed_s × 1e6``) rather than the gRPC RTT log, since the
    judge replaces ``check_tool_call`` outright and never hits the
    refmon. Other modes use the gRPC RTT log. Both are wall times
    the agent waits for an enforcement decision, which is the
    apples-to-apples comparison the paper needs.
    """
    if cell.mode == "judge":
        elapsed_us, usages = _read_judge_log(cell.judge_log) if cell.judge_log else ([], [])
        rtts = sorted(elapsed_us)
        if usages:
            costs = [float(u.get("cost_usd") or 0.0) for u in usages]
            tin = [float(u.get("tokens_in") or 0) for u in usages]
            tout = [float(u.get("tokens_out") or 0) for u in usages]
            n_calls = [float(u.get("n_llm_calls") or 0) for u in usages]
            cell.judge_total_cost_usd = sum(costs)
            cell.judge_mean_cost_usd = statistics.mean(costs)
            cell.judge_mean_tokens_in = statistics.mean(tin)
            cell.judge_mean_tokens_out = statistics.mean(tout)
            cell.judge_mean_llm_calls = statistics.mean(n_calls)
    else:
        rtts = sorted(_read_latencies(cell.latency_log) if cell.latency_log else [])

    cell.num_auth_calls = len(rtts)
    if rtts:
        cell.auth_p50_us = _percentile(rtts, 0.50)
        cell.auth_mean_us = statistics.mean(rtts)
        cell.auth_p95_us = _percentile(rtts, 0.95)
        cell.auth_p99_us = _percentile(rtts, 0.99)
        cell.auth_max_us = rtts[-1]

    n_tasks, mean_d, std_d, _sum_d = _read_throughput(cell.sim_path)
    cell.num_tasks = n_tasks
    cell.task_dur_mean_s = mean_d
    cell.task_dur_std_s = std_d
    if cell.wall_clock_s > 0 and n_tasks > 0:
        # End-to-end throughput from the operator's perspective —
        # tasks completed per wall-second across the cell.
        cell.throughput_tasks_per_s = n_tasks / cell.wall_clock_s

    # Server-side latency captured during the cell. Records come
    # from ``SASY_SERVER_LATENCY_LOG_FILE`` (rotated per cell into
    # ``server_replay.jsonl``); we summarise their ``total_us``.
    if cell.server_replay_log is not None:
        srtts = sorted(_read_server_total_us(cell.server_replay_log))
        cell.server_n = len(srtts)
        if srtts:
            cell.server_p50_us = _percentile(srtts, 0.50)
            cell.server_mean_us = statistics.mean(srtts)
            cell.server_p95_us = _percentile(srtts, 0.95)
            cell.server_p99_us = _percentile(srtts, 0.99)
            cell.server_max_us = srtts[-1]

    # Loadgen replay (when --loadgen was passed). One or many
    # trial files: with --loadgen-trials N>1 each trial is in
    # ``loadgen-trial-{i}.jsonl``; with N==1 it's just
    # ``loadgen.jsonl`` (back-compat). Aggregate across all trials
    # for stable percentiles + emit a per-trial throughput stdev
    # so the table can show ``mean ± std``.
    if cell.loadgen_log is not None:
        cell_dir = cell.loadgen_log.parent
        trial_paths = sorted(cell_dir.glob("loadgen-trial-*.jsonl"))
        if not trial_paths:
            trial_paths = [cell.loadgen_log]
        cell.loadgen_n_trials = len(trial_paths)

        all_rtts: list[float] = []
        per_trial_thru: list[float] = []
        for tp in trial_paths:
            tp_rtts = _read_latencies(tp)
            sidecar = _read_loadgen_sidecar(tp)
            if sidecar is not None:
                n_records = len(tp_rtts)
                if sidecar["successful_calls"] != n_records:
                    raise ValueError(
                        f"loadgen sidecar for {tp.name}: successful_calls "
                        f"({sidecar['successful_calls']}) != record count "
                        f"({n_records})"
                    )
                per_trial_thru.append(n_records / sidecar["elapsed_s"])
            if not tp_rtts:
                continue
            all_rtts.extend(tp_rtts)
            if sidecar is None:
                ts_span = _loadgen_ts_span(tp)
                if ts_span is not None and ts_span > 0:
                    per_trial_thru.append(len(tp_rtts) / ts_span)

        if all_rtts:
            srtts = sorted(all_rtts)
            cell.loadgen_n = len(srtts)
            cell.loadgen_p50_us = _percentile(srtts, 0.50)
            cell.loadgen_mean_us = statistics.mean(srtts)
            cell.loadgen_p95_us = _percentile(srtts, 0.95)
            cell.loadgen_p99_us = _percentile(srtts, 0.99)
            cell.loadgen_max_us = srtts[-1]

        if per_trial_thru:
            cell.auth_throughput_calls_per_s = statistics.mean(per_trial_thru)
            cell.auth_throughput_std = (
                statistics.stdev(per_trial_thru)
                if len(per_trial_thru) > 1 else 0.0
            )
            # ``loadgen_wall_s`` becomes the mean per-trial wall —
            # representative summary; the per-trial walls live in
            # the per-trial files for anyone who needs them.
            mean_n = statistics.mean(len(_read_latencies(tp)) for tp in trial_paths
                                     if _read_latencies(tp))
            cell.loadgen_wall_s = (
                mean_n / cell.auth_throughput_calls_per_s
                if cell.auth_throughput_calls_per_s > 0 else float("nan")
            )

    # Failed replay cells may retain completed trials for inspection, but
    # regenerated reports must not publish their partial-workload throughput.
    if cell.measurement_kind == "persisted_graph_replay" and cell.returncode != 0:
        cell.auth_throughput_calls_per_s = float("nan")
        cell.auth_throughput_std = float("nan")

    return cell


def _loadgen_ts_span(path: Path) -> Optional[float]:
    """``ts.last - ts.first`` from a loadgen ``.jsonl`` — legacy
    approximation of the timed-pass duration, exclusive of
    subprocess startup, the per-session warmup, and process
    teardown. Returns ``None`` if the file is missing or empty.

    Prefer ``_read_loadgen_sidecar`` when a ``<out>.summary.json``
    sidecar is present: its ``elapsed_s`` is an authoritative
    monotonic measurement from the Rust loadgen binary.
    """
    if not path or not path.exists():
        return None
    first: Optional[float] = None
    last:  Optional[float] = None
    try:
        with path.open() as f:
            for line in f:
                try:
                    ts = json.loads(line).get("ts")
                except json.JSONDecodeError:
                    continue
                if ts is None:
                    continue
                if first is None:
                    first = ts
                last = ts
    except OSError:
        return None
    if first is None or last is None or last < first:
        return None
    return max(last - first, 0.0)


def _read_loadgen_sidecar(trial_path: Path) -> Optional[dict]:
    """Read the ``<trial>.summary.json`` sidecar written by the Rust
    loadgen (version 1+). Returns the parsed dict when valid, or
    ``None`` when the sidecar file does not exist (legacy artifact).

    Raises ``ValueError`` when the sidecar exists but is
    malformed, has an unsupported version, or contains
    inconsistent counts — callers must not silently fall back to
    the legacy timestamp-span calculation for a *present but
    broken* sidecar.
    """
    sidecar = Path(str(trial_path) + ".summary.json")
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"loadgen sidecar {sidecar} is unreadable: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"loadgen sidecar {sidecar}: expected object, got {type(data).__name__}")
    version = data.get("version")
    if isinstance(version, bool) or version != 1:
        raise ValueError(f"loadgen sidecar {sidecar}: unsupported version {version!r}")
    elapsed = data.get("elapsed_s")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError(
            f"loadgen sidecar {sidecar}: elapsed_s must be positive finite, got {elapsed!r}"
        )
    for key in ("expected_calls", "attempted_calls", "successful_calls", "failed_calls"):
        v = data.get(key)
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise ValueError(f"loadgen sidecar {sidecar}: {key} must be non-negative int, got {v!r}")
    if data["expected_calls"] < 1:
        raise ValueError(
            f"loadgen sidecar {sidecar}: expected_calls must be positive, got {data['expected_calls']}"
        )
    if data["attempted_calls"] != data["successful_calls"] + data["failed_calls"]:
        raise ValueError(
            f"loadgen sidecar {sidecar}: attempted != successful + failed "
            f"({data['attempted_calls']} != {data['successful_calls']} + {data['failed_calls']})"
        )
    if data["failed_calls"] != 0:
        raise ValueError(
            f"loadgen sidecar {sidecar}: workload has {data['failed_calls']} failed calls"
        )
    if data["attempted_calls"] != data["expected_calls"]:
        raise ValueError(
            f"loadgen sidecar {sidecar}: attempted_calls ({data['attempted_calls']}) "
            f"!= expected_calls ({data['expected_calls']})"
        )
    if data["successful_calls"] != data["expected_calls"]:
        raise ValueError(
            f"loadgen sidecar {sidecar}: successful_calls ({data['successful_calls']}) "
            f"!= expected_calls ({data['expected_calls']})"
        )
    return data


# ── Output ────────────────────────────────────────────────


CSV_FIELDS = [
    "domain", "mode", "concurrency",
    "source_cell_id", "source_mode", "agent_llm", "user_llm", "measurement_kind", "capture_concurrencies",
    "num_auth_calls",
    "auth_p50_us", "auth_mean_us", "auth_p95_us", "auth_p99_us", "auth_max_us",
    "judge_total_cost_usd", "judge_mean_cost_usd",
    "judge_mean_tokens_in", "judge_mean_tokens_out", "judge_mean_llm_calls",
    "num_tasks", "wall_clock_s", "throughput_tasks_per_s",
    "task_dur_mean_s", "task_dur_std_s",
    # Server-side total_us during the actual cell, captured from
    # ``server_replay.jsonl``. Only populated for sasy cells when
    # the server log was rotated per cell (--server-latency-log or
    # --manage-server).
    "server_n",
    "server_p50_us", "server_mean_us", "server_p95_us",
    "server_p99_us", "server_max_us",
    # Loadgen replay against the warm server (Rust client, no GIL,
    # no agent pacing). Only populated for sasy cells when
    # ``--loadgen`` was passed.
    "loadgen_n", "loadgen_n_trials",
    "loadgen_p50_us", "loadgen_mean_us", "loadgen_p95_us",
    "loadgen_p99_us", "loadgen_max_us",
    "loadgen_wall_s",
    "auth_throughput_calls_per_s", "auth_throughput_std",
    "returncode",
]


def _write_csv(out_path: Path, cells: list[CellResult]) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for c in cells:
            w.writerow({k: getattr(c, k) for k in CSV_FIELDS})


def _fmt_concurrency(cell: CellResult) -> str:
    if cell.concurrency is None:
        return f"mixed ({cell.capture_concurrencies})"
    return str(cell.concurrency)


def _fmt_us(x: float) -> str:
    if math.isnan(x):
        return "—"
    if x < 1000:
        return f"{x:.0f}us"
    return f"{x/1000:.2f}ms"


def _fmt_thru(x: float) -> str:
    return "—" if math.isnan(x) else f"{x:.3f}"


def _fmt_rate(x: float) -> str:
    """Calls/sec — integer for ≥100 cps, two decimals below."""
    if math.isnan(x):
        return "—"
    return f"{x:.0f}" if x >= 100 else f"{x:.2f}"


def _fmt_wall_ms(s: float) -> str:
    """Wall time in milliseconds, one decimal. Loadgen runs against
    a warm server are sub-second; the parent-table seconds column
    rounds them to 0.0, which is misleading."""
    if math.isnan(s):
        return "—"
    return f"{s * 1000:.1f}"


def _fmt_cost(x: float) -> str:
    return "—" if math.isnan(x) else f"${x:.4f}"


def _fmt_n(x: float) -> str:
    return "—" if math.isnan(x) else f"{x:.0f}"


def _markdown_tables(cells: list[CellResult]) -> str:
    """Latency / cost / throughput tables per domain.

    Latency is the wall time the agent waits per enforcement
    decision — gRPC RTT for sasy/baseline, judge LLM elapsed
    time for judge — so a single column is a fair comparison
    across modes.
    """
    by_domain: dict[str, list[CellResult]] = defaultdict(list)
    for c in cells:
        by_domain[c.domain].append(c)

    out: list[str] = []
    for domain in sorted(by_domain):
        cs = by_domain[domain]
        has_judge = any(c.mode == "judge" for c in cs)

        out.append(f"## {domain}")
        out.append("")
        out.append(
            "### Client-side authorization latency"
        )
        out.append("")
        out.append("Per-call wall time as the agent observes it: SDK "
                   "serialization + wire + server processing + response "
                   "deserialization. Includes any GIL / channel-pool "
                   "contention on the client.")
        out.append("")
        out.append("| mode | conc | n | p50 | mean | p95 | p99 | max |")
        out.append("|------|-----:|--:|----:|-----:|----:|----:|----:|")
        for c in sorted(cs, key=lambda x: (x.mode, x.concurrency or 0)):
            out.append(
                f"| {c.mode} | {_fmt_concurrency(c)} | {c.num_auth_calls} | "
                f"{_fmt_us(c.auth_p50_us)} | {_fmt_us(c.auth_mean_us)} | "
                f"{_fmt_us(c.auth_p95_us)} | {_fmt_us(c.auth_p99_us)} | "
                f"{_fmt_us(c.auth_max_us)} |"
            )
        out.append("")

        if has_judge:
            out.append("### Judge cost / token usage (per-call means)")
            out.append("")
            out.append(
                "| conc | n | mean cost | total cost | mean tok_in | "
                "mean tok_out | mean LLM calls |"
            )
            out.append(
                "|-----:|--:|----------:|-----------:|------------:|"
                "-------------:|---------------:|"
            )
            for c in sorted(
                (c for c in cs if c.mode == "judge"),
                key=lambda x: x.concurrency,
            ):
                out.append(
                    f"| {_fmt_concurrency(c)} | {c.num_auth_calls} | "
                    f"{_fmt_cost(c.judge_mean_cost_usd)} | "
                    f"{_fmt_cost(c.judge_total_cost_usd)} | "
                    f"{_fmt_n(c.judge_mean_tokens_in)} | "
                    f"{_fmt_n(c.judge_mean_tokens_out)} | "
                    f"{_fmt_n(c.judge_mean_llm_calls)} |"
                )
            out.append("")

        out.append("### End-to-end throughput")
        out.append("")
        out.append("| mode | conc | tasks | wall (s) | tasks/s | task mean (s) | task stdev (s) |")
        out.append("|------|-----:|------:|---------:|--------:|--------------:|---------------:|")
        for c in sorted(cs, key=lambda x: (x.mode, x.concurrency or 0)):
            mean_s = "—" if math.isnan(c.task_dur_mean_s) else f"{c.task_dur_mean_s:.1f}"
            std_s = "—" if math.isnan(c.task_dur_std_s) else f"{c.task_dur_std_s:.1f}"
            out.append(
                f"| {c.mode} | {_fmt_concurrency(c)} | {c.num_tasks} | "
                f"{c.wall_clock_s:.1f} | {_fmt_thru(c.throughput_tasks_per_s)} | "
                f"{mean_s} | {std_s} |"
            )
        out.append("")

        # Server-side latency under the actual cell workload. Read
        # from the JSONL the engine wrote during the cell, so it
        # reflects whatever queueing / IPC / eval cost the engine
        # paid while the agent was driving traffic at the cell's
        # concurrency.
        has_server = any(c.server_n > 0 for c in cs)
        if has_server:
            out.append("### Server-side authorization latency")
            out.append("")
            out.append("Per-call ``total_us`` from the engine's JSONL "
                       "log captured while the cell ran. This is the "
                       "real server cost under the actual concurrency "
                       "the agent drove — sits between the client-side "
                       "row (client + wire + server) and the replay "
                       "row (server-only at saturation).")
            out.append("")
            out.append("| mode | conc | n | p50 | mean | p95 | p99 | max |")
            out.append("|------|-----:|--:|----:|-----:|----:|----:|----:|")
            for c in sorted(cs, key=lambda x: (x.mode, x.concurrency or 0)):
                if c.server_n == 0:
                    continue
                out.append(
                    f"| {c.mode} | {_fmt_concurrency(c)} | {c.server_n} | "
                    f"{_fmt_us(c.server_p50_us)} | {_fmt_us(c.server_mean_us)} | "
                    f"{_fmt_us(c.server_p95_us)} | {_fmt_us(c.server_p99_us)} | "
                    f"{_fmt_us(c.server_max_us)} |"
                )
            out.append("")

        # Authorization latency from the Rust loadgen + sustained
        # auth throughput (only if any sasy cell actually ran the
        # loadgen). The loadgen drives the same recorded calls
        # against the same warm graph state without a Python SDK in
        # the path, so its RTT is the engine + wire + tonic-client
        # cost — the actual authorization latency before any
        # agent/SDK overhead. Subtract from the end-to-end row
        # above to estimate Python-SDK contribution per call.
        # ``auth/s`` is the engine's sustained authorization rate
        # at this concurrency, untainted by agent / LLM pacing.
        has_loadgen = any(c.loadgen_n > 0 for c in cs)
        if has_loadgen:
            out.append("### Replay authorization latency (Rust loadgen)")
            out.append("")
            out.append("Per-call wall time the engine actually serves: "
                       "tonic client → wire → policy engine. No Python "
                       "SDK, no GIL, no agent pacing. Lower bound on "
                       "what the engine could deliver if the SDK were "
                       "removed; subtract from the client-side row to "
                       "estimate SDK overhead.")
            out.append("")
            any_multi = any(c.loadgen_n_trials > 1 for c in cs)
            trials_col = " trials |" if any_multi else ""
            trials_sep = "------:|" if any_multi else ""
            out.append(
                "| mode | conc | n |"
                + (" trials |" if any_multi else "")
                + " p50 | mean | p95 | p99 | max | wall (ms) | auth/s |"
            )
            out.append(
                "|------|-----:|--:|"
                + ("-------:|" if any_multi else "")
                + "----:|-----:|----:|----:|----:|----------:|-------:|"
            )
            for c in sorted(cs, key=lambda x: (x.mode, x.concurrency or 0)):
                if c.loadgen_n == 0:
                    continue
                trials_cell = (
                    f" {c.loadgen_n_trials} |" if any_multi else ""
                )
                # mean ± std on auth/s — tells the reader how
                # noisy the throughput estimate is. Single-trial
                # rows (std=0 or NaN) collapse to plain mean.
                rate = _fmt_rate(c.auth_throughput_calls_per_s)
                if (c.loadgen_n_trials > 1
                        and not math.isnan(c.auth_throughput_std)
                        and c.auth_throughput_std > 0):
                    rate = f"{rate} ± {_fmt_rate(c.auth_throughput_std)}"
                out.append(
                    f"| {c.mode} | {_fmt_concurrency(c)} | {c.loadgen_n} |"
                    + trials_cell
                    + f" {_fmt_us(c.loadgen_p50_us)} |"
                    f" {_fmt_us(c.loadgen_mean_us)} |"
                    f" {_fmt_us(c.loadgen_p95_us)} |"
                    f" {_fmt_us(c.loadgen_p99_us)} |"
                    f" {_fmt_us(c.loadgen_max_us)} |"
                    f" {_fmt_wall_ms(c.loadgen_wall_s)} |"
                    f" {rate} |"
                )
            out.append("")
    return "\n".join(out)


def _write_outputs(out_root: Path, cells: list[CellResult]) -> None:
    _write_csv(out_root / "results.csv", cells)
    replay_only = bool(cells) and all(c.measurement_kind == "persisted_graph_replay" for c in cells)
    if replay_only:
        from saved_benchmark_replay import replay_latex, replay_markdown
        md = replay_markdown(cells, sys.modules[__name__])
    else:
        md = _markdown_tables(cells)
    (out_root / "results.md").write_text(md)

    # Structured summary for plotting / paper-table generation.
    # Lossless compared to the CSV; avoids re-parsing the markdown
    # in downstream tooling.
    summary = {
        "domains": sorted({c.domain for c in cells}),
        "modes": sorted({c.mode for c in cells}),
        "concurrencies": sorted({c.concurrency for c in cells}),
        "cells": [
            {k: getattr(c, k) for k in CSV_FIELDS}
            for c in cells
        ],
    }
    # ``json.dumps`` chokes on NaN by default — replace with None
    # so the file is valid JSON for any consumer.
    def _scrub(v: object) -> object:
        if isinstance(v, float) and math.isnan(v):
            return None
        return v
    for cell in summary["cells"]:
        for k in list(cell.keys()):
            cell[k] = _scrub(cell[k])
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))

    # LaTeX tables matching the paper's tabular style
    # (\toprule/\midrule/\bottomrule, \cmidrule grouped headers).
    (out_root / "results.tex").write_text(replay_latex(cells) if replay_only else _latex_tables(cells))

    print()
    print(md)
    print()
    print(f"CSV:      {out_root / 'results.csv'}")
    print(f"Markdown: {out_root / 'results.md'}")
    print(f"Summary:  {out_root / 'summary.json'}")
    print(f"LaTeX:    {out_root / 'results.tex'}")


def _latex_tables(cells: list[CellResult]) -> str:
    """Emit LaTeX tables in the same style as the case-study paper.

    One ``\\begin{table}`` per (domain, table-kind) tuple. Loadgen
    rows are only emitted when at least one cell has loadgen data.
    """
    by_domain: dict[str, list[CellResult]] = defaultdict(list)
    for c in cells:
        by_domain[c.domain].append(c)

    out: list[str] = []

    def fmt_us_tex(x: float) -> str:
        if math.isnan(x):
            return "---"
        if x < 1000:
            return f"{x:.0f}\\,\\textmu{{}}s"
        return f"{x/1000:.2f}\\,ms"

    def fmt_rate(x: float) -> str:
        if math.isnan(x):
            return "---"
        return f"{x:.0f}" if x >= 100 else f"{x:.2f}"

    def fmt_s(x: float, prec: int = 1) -> str:
        return "---" if math.isnan(x) else f"{x:.{prec}f}"

    for domain in sorted(by_domain):
        cs = sorted(
            by_domain[domain],
            key=lambda x: (x.mode, x.concurrency or 0),
        )
        sasy_cells = [c for c in cs if c.mode in {"sasy", "enforce-on"}]

        # End-to-end SDK latency + throughput, matching
        # tab:tau2-overhead's grouped-header style.
        out.append("\\begin{table}[t]")
        out.append("\\centering")
        out.append(
            "\\caption{Microbench results on the "
            f"\\textbf{{{domain}}} domain. "
            "Client-side latency (SDK round-trip), server-side "
            "latency (engine \\texttt{total\\_us} from the per-cell "
            "JSONL), and replay (Rust loadgen, engine + wire only).}"
        )
        out.append(f"\\label{{tab:microbench-{domain}-latency}}")
        out.append("\\small")
        out.append("\\setlength{\\tabcolsep}{4pt}")
        out.append("\\begin{tabular}{@{}lrrrrr@{}}")
        out.append("\\toprule")
        out.append(
            "\\textbf{Source} & \\textbf{Conc} & \\textbf{p50} & "
            "\\textbf{Mean} & \\textbf{p95} & \\textbf{p99} \\\\"
        )
        out.append("\\midrule")
        # Three banded sub-tables in one tabular: SDK, server, loadgen.
        for label, p50, mean, p95, p99 in [
            ("Client side",
             "auth_p50_us", "auth_mean_us", "auth_p95_us", "auth_p99_us"),
            ("Server side",
             "server_p50_us", "server_mean_us", "server_p95_us", "server_p99_us"),
            ("Replay",
             "loadgen_p50_us", "loadgen_mean_us", "loadgen_p95_us", "loadgen_p99_us"),
        ]:
            for i, c in enumerate(sasy_cells):
                lbl = f"\\multirow{{{len(sasy_cells)}}}{{*}}{{{label}}}" if i == 0 else ""
                out.append(
                    f"{lbl} & {_fmt_concurrency(c)} & "
                    f"{fmt_us_tex(getattr(c, p50))} & "
                    f"{fmt_us_tex(getattr(c, mean))} & "
                    f"{fmt_us_tex(getattr(c, p95))} & "
                    f"{fmt_us_tex(getattr(c, p99))} \\\\"
                )
            out.append("\\midrule")
        # Drop trailing midrule.
        if out[-1] == "\\midrule":
            out[-1] = "\\bottomrule"
        else:
            out.append("\\bottomrule")
        out.append("\\end{tabular}")
        out.append("\\end{table}")
        out.append("")

        # End-to-end throughput with task duration variance.
        out.append("\\begin{table}[t]")
        out.append("\\centering")
        out.append(
            "\\caption{End-to-end throughput on the "
            f"\\textbf{{{domain}}} domain. ``Task mean'' and "
            "``Task stdev'' are per-task wall-clock duration "
            "across the cell's trials --- high stdev indicates "
            "wall time was dominated by long-tail tasks rather "
            "than total work.}"
        )
        out.append(f"\\label{{tab:microbench-{domain}-throughput}}")
        out.append("\\small")
        out.append("\\setlength{\\tabcolsep}{4pt}")
        out.append("\\begin{tabular}{@{}lrrrrrr@{}}")
        out.append("\\toprule")
        out.append(
            "\\textbf{Mode} & \\textbf{Conc} & \\textbf{Tasks} & "
            "\\textbf{Wall (s)} & \\textbf{Tasks/s} & "
            "\\textbf{Task mean (s)} & \\textbf{Task stdev (s)} \\\\"
        )
        out.append("\\midrule")
        for c in cs:
            out.append(
                f"{c.mode} & {_fmt_concurrency(c)} & {c.num_tasks} & "
                f"{fmt_s(c.wall_clock_s, 1)} & "
                f"{fmt_s(c.throughput_tasks_per_s, 3)} & "
                f"{fmt_s(c.task_dur_mean_s, 1)} & "
                f"{fmt_s(c.task_dur_std_s, 1)} \\\\"
            )
        out.append("\\bottomrule")
        out.append("\\end{tabular}")
        out.append("\\end{table}")
        out.append("")

        # Loadgen-only saturation throughput table, if any
        # cell has loadgen data.
        if any(c.loadgen_n > 0 for c in sasy_cells):
            any_multi = any(c.loadgen_n_trials > 1 for c in sasy_cells)
            out.append("\\begin{table}[t]")
            out.append("\\centering")
            out.append(
                "\\caption{Engine-only authorization throughput "
                "from the Rust loadgen replay against the same "
                f"server on the \\textbf{{{domain}}} domain. "
                "Per-session warmup amortized so the timed window "
                "reflects steady-state cost."
                + (" Auth/s reported as mean $\\pm$ stdev across "
                   "independent replay trials; latency percentiles "
                   "are computed over the pooled distribution."
                   if any_multi else "")
                + "}"
            )
            out.append(f"\\label{{tab:microbench-{domain}-loadgen}}")
            out.append("\\small")
            out.append("\\setlength{\\tabcolsep}{4pt}")
            cols = "@{}rr" + ("r" if any_multi else "") + "rrrr@{}"
            out.append(f"\\begin{{tabular}}{{{cols}}}")
            out.append("\\toprule")
            header = "\\textbf{Conc} & \\textbf{n}"
            if any_multi:
                header += " & \\textbf{Trials}"
            header += (
                " & \\textbf{p50} & \\textbf{p99} & "
                "\\textbf{Wall (ms)} & \\textbf{Auth/s} \\\\"
            )
            out.append(header)
            out.append("\\midrule")
            for c in sasy_cells:
                if c.loadgen_n == 0:
                    continue
                wall_ms = (
                    "---" if math.isnan(c.loadgen_wall_s)
                    else f"{c.loadgen_wall_s * 1000:.1f}"
                )
                rate_str = fmt_rate(c.auth_throughput_calls_per_s)
                if (c.loadgen_n_trials > 1
                        and not math.isnan(c.auth_throughput_std)
                        and c.auth_throughput_std > 0):
                    rate_str = (
                        f"{rate_str} $\\pm$ "
                        f"{fmt_rate(c.auth_throughput_std)}"
                    )
                trials_cell = (
                    f" & {c.loadgen_n_trials}" if any_multi else ""
                )
                out.append(
                    f"{_fmt_concurrency(c)} & {c.loadgen_n}"
                    f"{trials_cell} & "
                    f"{fmt_us_tex(c.loadgen_p50_us)} & "
                    f"{fmt_us_tex(c.loadgen_p99_us)} & "
                    f"{wall_ms} & {rate_str} \\\\"
                )
            out.append("\\bottomrule")
            out.append("\\end{tabular}")
            out.append("\\end{table}")
            out.append("")

    return "\n".join(out)


# ── Reduce-only path ──────────────────────────────────────


def _discover_cells(run_dir: Path) -> list[CellResult]:
    """Rebuild CellResults from a previous run directory.

    Each per-cell subdir is ``<domain>_<mode>_c<conc>/``,
    containing ``latency.jsonl`` and an ``rb_out/...`` tree
    with the tau2 sim path inside the manifest.
    """
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("kind") == "saved-benchmark-replay":
            from saved_benchmark_replay import discover_replay_cells
            return discover_replay_cells(manifest, run_dir)
    cells: list[CellResult] = []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir() or not sub.name.startswith(("retail_", "airline_")):
            continue
        try:
            domain, mode, conc_token = sub.name.split("_", 2)
            conc = int(conc_token.lstrip("c"))
        except ValueError:
            continue
        latency_log = sub / "latency.jsonl"
        # Recover wall_clock_s + sim_path from the run_benchmark
        # manifest if it exists.
        wall = 0.0
        sim_path: Optional[Path] = None
        rc = 0
        manifests = list((sub / "rb_out").glob("*/manifest.json"))
        if manifests:
            try:
                mf_path = manifests[-1]
                mf_dir = mf_path.parent
                manifest = json.loads(mf_path.read_text())
                cell_entries = manifest.get("cells") or []
                if cell_entries:
                    rc = int(cell_entries[0].get("returncode") or 0)
                    sp = cell_entries[0].get("sim_path")
                    if sp:
                        if not Path(sp).is_absolute():
                            sim_path = mf_dir / sp
                        else:
                            sim_path = Path(sp)
            except (json.JSONDecodeError, OSError):
                pass
        # Reduce-only mode can't recover wall_clock_s from disk
        # (run_benchmark.py doesn't log it per-cell). Use the
        # sum of per-task durations as a proxy when available.
        if sim_path and sim_path.exists():
            try:
                data = json.loads(sim_path.read_text())
                durs = [
                    s.get("duration") or 0.0
                    for s in (data.get("simulations") or [])
                ]
                if durs:
                    # Wall ≈ sum(durations) / concurrency, since
                    # tau2 dispatches trials with a thread pool
                    # of size --max-concurrency.
                    wall = sum(durs) / max(conc, 1)
            except (json.JSONDecodeError, OSError):
                pass
        # The captured server replay log doubles as the
        # actual-workload server-side latency source.
        srl: Optional[Path] = None
        candidate_srl = sub / "server_replay.jsonl"
        if candidate_srl.exists() and candidate_srl.stat().st_size > 0:
            srl = candidate_srl

        # Pick up the loadgen output if it's there. With
        # ``--loadgen-trials > 1`` each trial lives in
        # ``loadgen-trial-{i}.jsonl``; otherwise it's the legacy
        # single-file ``loadgen.jsonl``. The reducer keys off
        # ``cell.loadgen_log.parent`` to find sibling trial files,
        # so we just need a representative path that points at the
        # cell directory — first trial when sharded, the legacy
        # file otherwise.
        loadgen_log_path: Optional[Path] = None
        loadgen_wall_s = float("nan")
        trial_files = sorted(sub.glob("loadgen-trial-*.jsonl"))
        single = sub / "loadgen.jsonl"
        if trial_files:
            loadgen_log_path = trial_files[0]
        elif single.exists() and single.stat().st_size > 0:
            loadgen_log_path = single
        if loadgen_log_path is not None:
            # Prefer sidecar for wall time when available.
            sidecar = None
            try:
                sidecar = _read_loadgen_sidecar(loadgen_log_path)
            except ValueError:
                pass  # invalid sidecar in discover — reduce will re-check
            if sidecar is not None:
                loadgen_wall_s = sidecar["elapsed_s"]
            else:
                ts_span = _loadgen_ts_span(loadgen_log_path)
                if ts_span is not None:
                    loadgen_wall_s = ts_span

        cells.append(CellResult(
            domain=domain, mode=mode, concurrency=conc,
            returncode=rc, wall_clock_s=wall,
            sim_path=sim_path, latency_log=latency_log,
            judge_log=_judge_log_for(sim_path) if sim_path else None,
            server_replay_log=srl,
            loadgen_log=loadgen_log_path,
            loadgen_wall_s=loadgen_wall_s,
        ))
    return cells


# ── Main ──────────────────────────────────────────────────


def _install_termination_handler() -> None:
    # Unwind the active cell and managed server just as Ctrl+C does.
    def on_term(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, on_term)


def main() -> int:
    _install_termination_handler()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--domains", nargs="+", default=list(DEFAULT_DOMAINS),
        choices=("retail", "airline"),
        help="Domains to sweep (default: airline).",
    )
    parser.add_argument(
        "--modes", nargs="+", default=list(DEFAULT_MODES),
        choices=("enforce-on",),
        help="Instrumented enforcement mode (enforce-on only).",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        choices=("opus", "gemini", "gpt"),
        help="Single fixed agent model for the sweep "
        "(default: gpt — cheapest, most stable provider for "
        "high-concurrency runs).",
    )
    parser.add_argument(
        "--concurrencies", nargs="+", type=int,
        default=list(DEFAULT_CONCURRENCIES),
        help="Concurrency levels to sweep "
        "(default: 1 2 4 8 16 32).",
    )
    parser.add_argument(
        "--num-trials", type=int, default=DEFAULT_NUM_TRIALS,
        help="Trials per task per cell. Default: per-cell "
        "max(8, 2 × concurrency) — 8 floor for stable p99 at "
        "low conc, 2× concurrency for multiple saturated "
        "batches per task at high conc. Pass an explicit value "
        "to override for all cells.",
    )
    parser.add_argument(
        "--cell-timeout", type=int, default=3000,
        help="Wall-clock timeout per cell in seconds "
        "(default: 3000 = 50 min). Sized to absorb tau2's "
        "airline/gpt long-tail retry trials without false-"
        "positiving; partial latency data still survives a "
        "kill and can be reduced via --reduce-only.",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO_ROOT / "results" / "latency-microbench",
        help="Where to write per-run output trees "
        "(default: results/latency-microbench).",
    )
    parser.add_argument(
        "--passthrough", nargs=argparse.REMAINDER, default=[],
        help="Extra args forwarded to run_benchmark.py (e.g. "
        "--task-subset curated). Anything after this "
        "flag is passed verbatim.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--reduce-only", type=Path, default=None,
        help="Skip running cells; reduce an existing run "
        "directory's logs to a CSV/MD table.",
    )
    parser.add_argument(
        "--from-benchmark", type=Path, default=None, metavar="RUN_DIR",
        help="Replay captured tau2 requests from a stopped main benchmark; no LLM calls."
        " Select with --domains, --model and --source-mode; sweep --concurrencies.",
    )
    parser.add_argument(
        "--source-mode", choices=("enforce-on", "enforce-off", "sasy"), default="enforce-on",
        help="Main benchmark arm selected by --from-benchmark (default: enforce-on).",
    )
    parser.add_argument(
        "--replay-loadgen-only", type=Path, default=None,
        help="Skip running tau2 cells; spawn a fresh server "
        "against the run's saved ``server/<domain>/`` data dir, "
        "replay each cell's captured ``server_replay.jsonl`` "
        "through the Rust loadgen, and re-render the outputs. "
        "Use to backfill loadgen numbers on a tau2 run that was "
        "executed without ``--loadgen``.",
    )
    # ── Loadgen integration ───────────────────────────────
    #
    # When ``--server-latency-log`` points at the path the bench
    # server is configured to write via ``SASY_SERVER_LATENCY_LOG_FILE``,
    # the wrapper truncates that file before each cell, captures
    # whatever the cell produced into ``server_replay.jsonl`` per
    # cell, and (if ``--loadgen`` is also set) replays it through
    # the Rust loadgen at the cell's concurrency to get an
    # engine-only latency + auth-throughput number.
    parser.add_argument(
        "--server-latency-log", type=Path, default=None,
        help="Path to the bench server's SASY_SERVER_LATENCY_LOG_FILE. "
        "When set, the wrapper rotates that file per cell so the "
        "captured replay matches the cell exactly.",
    )
    parser.add_argument(
        "--loadgen", action="store_true",
        help="After each sasy cell, replay its server-side log "
        "through the Rust loadgen and record engine-only latency + "
        "auth throughput. Requires --server-latency-log.",
    )
    parser.add_argument(
        "--loadgen-bin", type=Path, default=LOADGEN_BIN,
        help=f"Path to the sasy-loadgen binary (default: {LOADGEN_BIN}).",
    )
    parser.add_argument(
        "--loadgen-url", default=os.environ.get("SASY_LOADGEN_URL", "https://localhost:10089"),
        help="Server URL for the loadgen (default: env SASY_LOADGEN_URL "
        "or https://localhost:10089).",
    )
    parser.add_argument(
        "--loadgen-ca", type=Path,
        default=Path(os.environ.get("TLS_CA_PATH", REPO_ROOT / "certs" / "ca.crt")),
        help="CA cert for the loadgen.",
    )
    # Admin cert for symmetry with run_benchmark.py: ``.env``'s
    # TLS_CERT_PATH typically points at ``client.crt`` (only
    # reference-monitor-user / observability-writer), which is
    # enough for the current loadgen RPCs but breaks if any future
    # RPC needs observability-reader / policy-upload. Pin admin so
    # the bench is robust to that. Operator can override via
    # ``SASY_ADMIN_CERT_PATH`` / ``SASY_ADMIN_KEY_PATH`` (read by
    # both bench scripts) if their admin material lives elsewhere.
    parser.add_argument(
        "--loadgen-cert", type=Path,
        default=Path(os.environ.get(
            "SASY_ADMIN_CERT_PATH", REPO_ROOT / "certs" / "admin.crt")),
        help="Client cert for the loadgen (admin role).",
    )
    parser.add_argument(
        "--loadgen-key", type=Path,
        default=Path(os.environ.get(
            "SASY_ADMIN_KEY_PATH", REPO_ROOT / "certs" / "admin.key")),
        help="Client key for the loadgen (admin role).",
    )
    parser.add_argument(
        "--loadgen-api-key", default=os.environ.get("SASY_API_KEY"),
        help="API key for the loadgen (default: env SASY_API_KEY).",
    )
    parser.add_argument(
        "--loadgen-channels", type=int, default=64,
        help="Channel pool size for the loadgen (default: 64, matching "
        "SASY_CHANNEL_POOL_SIZE on the SDK side). Setting this equal to "
        "concurrency under-reports engine ceiling because each channel's "
        "I/O task serializes outgoing frames per worker.",
    )
    parser.add_argument(
        "--loadgen-repeat", type=int, default=1,
        help="Replay the cell's call stream N times within a single "
        "loadgen invocation (default 1). Bump to amortize warmup "
        "over more samples within one trial.",
    )
    parser.add_argument(
        "--loadgen-trials", type=int, default=1,
        help="Run the loadgen N independent times per cell (default "
        "1). The reducer aggregates all trials' RTTs for the "
        "latency percentiles and reports mean ± std of the per-"
        "trial throughput. 5–10 typical for paper-grade noise "
        "smoothing; cost is ~ms per extra trial since sessions "
        "stay warm across trials.",
    )
    parser.add_argument(
        "--loadgen-timeout", type=int, default=1800,
        help="Per-cell loadgen wall-clock cap in seconds (default 1800).",
    )
    parser.add_argument(
        "--loadgen-mode", choices=("session", "flat"), default="session",
        help="Replay scheduling: ``session`` (default) drives one "
        "task per session_id with serial within-session calls and a "
        "semaphore capping active sessions; ``flat`` ignores session "
        "boundaries and runs the whole row stream through a single "
        "global semaphore. Use flat to isolate engine/wire ceiling "
        "from session-mode straggler effects.",
    )
    # ── Server-management ──────────────────────────────────
    #
    # By default the wrapper spawns a sasy server for each domain
    # with the right policy + a per-run data dir under
    # ``results/latency-microbench/<ts>/server/<domain>/``, and
    # routes the server-side latency log there too. The operator
    # doesn't have to hand-launch a server with the right env
    # vars; everything stays inside the run dir for a clean
    # post-run review. Pass ``--no-manage-server`` to opt out
    # (e.g. when pointing at a remote cluster).
    parser.add_argument(
        "--manage-server", action=argparse.BooleanOptionalAction, default=True,
        help="Spawn the sasy server per domain (default: on). "
        "Pass --no-manage-server to point at an externally-"
        "running server instead. Implies --loadgen unless "
        "--no-loadgen is given.",
    )
    parser.add_argument(
        "--no-loadgen", action="store_true",
        help="Suppress loadgen even when --manage-server is on.",
    )
    parser.add_argument(
        "--server-bin", type=Path, default=SERVER_BIN,
        help=f"Path to the sasy server binary (default: {SERVER_BIN}).",
    )
    parser.add_argument(
        "--server-addr", default="0.0.0.0:10089",
        type=lambda addr: f"127.0.0.1:{addr}" if addr.isdecimal() else addr,
        help="Listen address or bare port for the managed server (default: "
        "0.0.0.0:10089). A bare port binds to 127.0.0.1.",
    )
    parser.add_argument(
        "--server-policy", action="append", default=[],
        help="Override the policy file for a domain "
        "(format: domain=path). May be passed multiple times. "
        "Defaults to experiments/tau2/<domain>_policy.dl.",
    )
    parser.add_argument(
        "--server-keepalive-ms", type=int, default=0,
        help="SASY_KEEPALIVE_INTERVAL_MS for the managed server "
        "(default 0 — disabled). Keepalive only helps idle sessions "
        "between agent calls; in microbench replay every active "
        "session is hammered continuously, so keepalive is pure "
        "noise (per-session pipe writes + tokio task wakeups). Bump "
        "to 250 if reproducing a production-shaped workload.",
    )
    parser.add_argument(
        "--server-ready-timeout", type=float, default=120.0,
        help="Max seconds to wait for the managed server to bind "
        "its gRPC port (default 120).",
    )
    parser.add_argument(
        "--server-env", action="append", default=[],
        help="Extra env var for the managed server (format: KEY=VALUE). "
        "May be passed multiple times.",
    )

    args = parser.parse_args()

    if args.from_benchmark is not None:
        from saved_benchmark_replay import replay_benchmark
        if args.reduce_only or args.replay_loadgen_only or not args.manage_server or args.no_loadgen:
            parser.error("--from-benchmark requires a managed replay engine and loadgen")
        if args.passthrough or args.server_policy:
            parser.error("--from-benchmark uses saved source policies; passthrough and policy overrides are unsupported")
        try:
            return replay_benchmark(args, sys.modules[__name__])
        except (ValueError, OSError, RuntimeError) as exc:
            print(f"ERROR: {exc}")
            return 1

    if args.manage_server and not args.no_loadgen:
        # Make --manage-server imply --loadgen + a sane default
        # server-latency-log location under run_root. The exact
        # path is per-domain (set by ``_spawn_server``) and gets
        # passed to ``_run_cell`` after the server comes up.
        args.loadgen = True

    if args.loadgen and args.server_latency_log is None and not args.manage_server:
        parser.error(
            "--loadgen requires --server-latency-log unless --manage-server is set"
        )

    # Reduce-only path: just rebuild CellResults from disk and
    # write the tables. Useful for re-running the analysis after
    # tweaking the formatter without re-spending model budget.
    if args.reduce_only is not None:
        run_dir = args.reduce_only.resolve()
        if not run_dir.is_dir():
            print(f"ERROR: --reduce-only path is not a directory: {run_dir}")
            return 1
        cells = _discover_cells(run_dir)
        for c in cells:
            _reduce_cell(c)
        _write_outputs(run_dir, cells)
        return 0

    # Replay-loadgen-only path: backfill loadgen numbers on a run
    # that was executed without ``--loadgen``. Spawns the server
    # against the run's saved ``server/<domain>/`` data dir, then
    # for each cell with a captured ``server_replay.jsonl`` runs
    # the Rust loadgen at that cell's concurrency. Re-renders.
    if args.replay_loadgen_only is not None:
        # Resolve to absolute so ``_spawn_server``'s
        # ``relative_to(REPO_ROOT)`` print works regardless of how
        # the operator typed the path.
        run_dir = args.replay_loadgen_only.resolve()
        if not run_dir.is_dir():
            print(f"ERROR: --replay-loadgen-only path is not a directory: {run_dir}")
            return 1
        cells = _discover_cells(run_dir)
        # Group cells by domain so we spawn one server per domain
        # against the matching saved data dir.
        by_domain: dict[str, list[CellResult]] = defaultdict(list)
        for c in cells:
            by_domain[c.domain].append(c)
        loadgen_cfg = {
            "bin_path": args.loadgen_bin,
            "url": args.loadgen_url,
            "ca": args.loadgen_ca if args.loadgen_ca and args.loadgen_ca.exists() else None,
            "cert": args.loadgen_cert if args.loadgen_cert and args.loadgen_cert.exists() else None,
            "key": args.loadgen_key if args.loadgen_key and args.loadgen_key.exists() else None,
            "api_key": args.loadgen_api_key,
            "channels": args.loadgen_channels,
            "repeat": args.loadgen_repeat,
            "trials": args.loadgen_trials,
            "timeout": args.loadgen_timeout,
            "mode": args.loadgen_mode,
        }
        if not args.loadgen_bin.exists():
            print(f"ERROR: loadgen binary not found at {args.loadgen_bin}")
            return 1
        for domain, cells_for_domain in by_domain.items():
            saved_data = run_dir / "server" / domain / "graph"
            if not saved_data.exists():
                print(f"WARN: no saved server data for domain={domain} at "
                      f"{saved_data}; skipping")
                continue
            policy_path = (
                Path(args.server_policy_for_domain(domain))
                if hasattr(args, "server_policy_for_domain")
                else DEFAULT_POLICY_FILES.get(domain)
            )
            if policy_path is None or not policy_path.exists():
                print(f"WARN: no policy file for domain={domain}; skipping")
                continue
            print(f"\n=== domain={domain}: spawning server against saved data dir ===")
            handle = _spawn_server(
                domain=domain,
                run_root=run_dir,        # reuse same layout
                addr=args.server_addr,
                policy_path=policy_path,
                server_bin=args.server_bin,
                extra_env={},
                keepalive_ms=args.server_keepalive_ms,
                ready_timeout=args.server_ready_timeout,
            )
            try:
                # Point loadgen at the just-spawned server.
                host, _, port = args.server_addr.rpartition(":")
                if host in ("0.0.0.0", "[::]", ""):
                    host = "127.0.0.1"
                cell_loadgen_url = f"https://{host}:{port}"
                for c in cells_for_domain:
                    if c.mode not in {"sasy", "enforce-on"}:
                        continue
                    cell_dir = run_dir / f"{c.domain}_{c.mode}_c{_fmt_concurrency(c)}"
                    replay = cell_dir / "server_replay.jsonl"
                    if not replay.exists() or replay.stat().st_size == 0:
                        print(f"  skip {cell_dir.name}: no captured replay")
                        continue
                    chans = (loadgen_cfg["channels"]
                             if loadgen_cfg["channels"] > 0 else c.concurrency)
                    n_trials = max(1, loadgen_cfg.get("trials", 1))
                    for old in cell_dir.glob("loadgen-trial-*.jsonl"):
                        old.unlink()
                    primary = cell_dir / "loadgen.jsonl"
                    if primary.exists():
                        primary.unlink()
                    for i in range(1, n_trials + 1):
                        trial_out = (
                            primary if n_trials == 1
                            else cell_dir / f"loadgen-trial-{i}.jsonl"
                        )
                        rc, wall = _run_loadgen(
                            bin_path=loadgen_cfg["bin_path"],
                            replay=replay,
                            out=trial_out,
                            url=cell_loadgen_url,
                            ca=loadgen_cfg["ca"],
                            cert=loadgen_cfg["cert"],
                            key=loadgen_cfg["key"],
                            api_key=loadgen_cfg["api_key"],
                            concurrency=c.concurrency,
                            channels=chans,
                            repeat=loadgen_cfg["repeat"],
                            timeout=loadgen_cfg["timeout"],
                            mode=loadgen_cfg["mode"],
                        )
                        if rc == 0 and trial_out.exists() and c.loadgen_log is None:
                            c.loadgen_log = trial_out
            finally:
                handle.stop()

        # Reduce + write with the freshly-produced loadgen.jsonls.
        for c in cells:
            _reduce_cell(c)
        _write_outputs(run_dir, cells)
        return 0

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_root = (args.output_root / timestamp).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    # Manifest written upfront; per-cell results appended as
    # they finish so a Ctrl-C still yields partial data.
    manifest = {
        "timestamp": timestamp,
        "domains": args.domains,
        "modes": args.modes,
        "model": args.model,
        "concurrencies": args.concurrencies,
        "num_trials": args.num_trials,
        "cells": [],
    }
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Sweep order: outer loop over (domain, mode), inner over
    # concurrency. Keeps the tau2 server's per-domain policy
    # stable while concurrency varies — fewer hot-swaps and
    # less noise in the graph-state cache.
    cells: list[CellResult] = []
    plan = [
        (d, m, c)
        for d in args.domains
        for m in args.modes
        for c in args.concurrencies
    ]
    try:
        display_root = run_root.relative_to(REPO_ROOT)
    except ValueError:
        display_root = run_root
    print(f"Sweeping {len(plan)} cells under {display_root}")
    for d, m, c in plan:
        print(f"  - {d} / {m} / c={c}")

    loadgen_cfg: Optional[dict] = None
    if args.loadgen:
        loadgen_cfg = {
            "bin_path": args.loadgen_bin,
            "url": args.loadgen_url,
            "ca": args.loadgen_ca if args.loadgen_ca and args.loadgen_ca.exists() else None,
            "cert": args.loadgen_cert if args.loadgen_cert and args.loadgen_cert.exists() else None,
            "key": args.loadgen_key if args.loadgen_key and args.loadgen_key.exists() else None,
            "api_key": args.loadgen_api_key,
            "channels": args.loadgen_channels,
            "repeat": args.loadgen_repeat,
            "trials": args.loadgen_trials,
            "timeout": args.loadgen_timeout,
            "mode": args.loadgen_mode,
        }
        if not args.loadgen_bin.exists():
            print(f"WARN: loadgen binary not found at {args.loadgen_bin}; "
                  "build with `cargo build --release -p sasy-loadgen` first")

    # Per-domain policy file resolution. Defaults are
    # ``experiments/tau2/<domain>_policy.dl``; ``--server-policy
    # domain=path`` overrides per domain.
    policy_overrides: dict[str, Path] = {}
    for entry in args.server_policy:
        if "=" not in entry:
            parser.error(f"--server-policy expects 'domain=path', got {entry!r}")
        d, _, p = entry.partition("=")
        policy_overrides[d.strip()] = Path(p.strip())
    server_extra_env: dict[str, str] = {}
    for entry in args.server_env:
        if "=" not in entry:
            parser.error(f"--server-env expects 'KEY=VALUE', got {entry!r}")
        k, _, v = entry.partition("=")
        server_extra_env[k.strip()] = v

    def _policy_for(domain: str) -> Path:
        if domain in policy_overrides:
            return policy_overrides[domain]
        if domain in DEFAULT_POLICY_FILES:
            return DEFAULT_POLICY_FILES[domain]
        raise FileNotFoundError(
            f"no policy mapping for domain={domain!r}; pass "
            f"--server-policy {domain}=path/to/policy.dl"
        )

    # Group the plan by domain so we can spawn one server per
    # domain (loading the right policy) and tear it down before
    # the next.
    by_domain: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for d, m, c in plan:
        by_domain[d].append((m, c))

    server_handle: Optional[ServerHandle] = None
    interrupted = False
    try:
        for domain in args.domains:
            domain_plan = by_domain.get(domain, [])
            if not domain_plan:
                continue

            # Per-domain server context. When --manage-server is
            # off we leave the server alone — operator runs it
            # externally and provides --server-latency-log.
            cell_server_log: Optional[Path] = args.server_latency_log
            cell_loadgen_cfg = loadgen_cfg
            cell_server_url: Optional[str] = None

            if args.manage_server and not args.dry_run:
                server_handle = _spawn_server(
                    domain=domain,
                    run_root=run_root,
                    addr=args.server_addr,
                    policy_path=_policy_for(domain),
                    server_bin=args.server_bin,
                    extra_env=server_extra_env,
                    keepalive_ms=args.server_keepalive_ms,
                    ready_timeout=args.server_ready_timeout,
                )
                cell_server_log = server_handle.latency_log_path
                # Compute the URL the SDK and loadgen should use.
                # ``run_benchmark.py``'s default ``--start-server``
                # path would otherwise spawn a *second* server on
                # its own port and the cell's auth traffic would
                # never reach the microbench's server (whose
                # latency log we're capturing). Routing the SDK
                # through ``SASY_URL=<microbench host:port>`` plus
                # ``--no-start-server`` (handled inside _run_cell)
                # closes that loop.
                host, _, port = args.server_addr.rpartition(":")
                if host in ("0.0.0.0", "[::]", ""):
                    host = "127.0.0.1"
                cell_server_url = f"{host}:{port}"
                if cell_loadgen_cfg is not None:
                    cell_loadgen_cfg = {
                        **cell_loadgen_cfg,
                        "url": f"https://{host}:{port}",
                    }

            try:
                for mode, conc in domain_plan:
                    cell = _run_cell(
                        out_root=run_root,
                        domain=domain,
                        mode=mode,
                        model=args.model,
                        concurrency=conc,
                        num_trials=_trials_for(conc, args.num_trials),
                        cell_timeout=args.cell_timeout,
                        passthrough=args.passthrough,
                        dry_run=args.dry_run,
                        server_log=cell_server_log,
                        loadgen_cfg=(
                            {**cell_loadgen_cfg,
                             "channels": cell_loadgen_cfg["channels"]
                                         if cell_loadgen_cfg["channels"] > 0 else conc}
                            if cell_loadgen_cfg is not None else None
                        ),
                        server_url=cell_server_url,
                    )
                    cells.append(_reduce_cell(cell))

                    def _portable(p: Path | None) -> str | None:
                        if p is None:
                            return None
                        try:
                            return str(p.relative_to(run_root))
                        except ValueError:
                            return str(p)
                    manifest["cells"].append({
                        "domain": cell.domain,
                        "mode": cell.mode,
                        "concurrency": cell.concurrency,
                        "returncode": cell.returncode,
                        "wall_clock_s": cell.wall_clock_s,
                        "num_auth_calls": cell.num_auth_calls,
                        "num_tasks": cell.num_tasks,
                        "latency_log": _portable(cell.latency_log),
                        "sim_path": _portable(cell.sim_path),
                    })
                    manifest_path.write_text(json.dumps(manifest, indent=2))
            finally:
                # Tear the per-domain server down before moving to
                # the next domain, even if a cell raised. The server
                # log + replay JSONL stay on disk under run_root.
                if server_handle is not None:
                    print(f"<<< stopping server for domain={domain}")
                    server_handle.stop()
                    server_handle = None
    except KeyboardInterrupt:
        # _run_cell already killed the in-flight cell's process
        # group on its way up. Reduce whatever cells finished, write
        # the partial manifest, then exit non-zero.
        interrupted = True
        print("\n! interrupted; writing partial outputs and exiting")
    finally:
        # Belt and suspenders: if we exit through any path with a
        # live server handle, stop it.
        if server_handle is not None:
            server_handle.stop()

    _write_outputs(run_root, cells)
    if interrupted:
        return 130
    failed = [c for c in cells if c.returncode != 0 or (not args.dry_run and c.num_auth_calls == 0)]
    if failed:
        print(f"! {len(failed)} cell(s) failed or produced no authorization samples")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
