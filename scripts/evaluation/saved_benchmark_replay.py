"""Replay the main runner's saved tau2 workload, without calling agent/user models.

Replays resolve artifact paths portably: relative paths under the manifest
directory are preferred over original absolute paths. This allows replaying
runs that have been moved or copied. Hash verification remains mandatory for
all capture artifacts.
"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

from benchmark_capture import copy_closed_graph, sha256, write_json
from manifest_paths import (
    portable_path,
    resolve_cell_dir,
    resolve_graph_dir,
    resolve_verified_artifact,
)


def _verified_file(info: dict, key: str, cell_dir: Path, *, run_dir: str | None = None) -> Path:
    """Resolve and verify a capture artifact.

    Args:
        info: The latency_capture dict from a manifest cell.
        key: The artifact key (policy, functors, sdk_log, server_replay).
        cell_dir: Directory containing the per-cell capture file.
        run_dir: Original run_dir for legacy absolute path rebasing.

    Raises:
        ValueError: If the artifact is missing or its hash doesn't match.
    """
    return resolve_verified_artifact(info, key, cell_dir, run_dir=run_dir)


def select_sources(run_dir: Path, *, domains: list[str], model: str, mode: str) -> tuple[dict, list[dict]]:
    """Select and validate source cells for replay.

    Resolves paths portably: relative paths under run_dir (for moved runs)
    are tried before original absolute paths.

    Embedded capture paths in the main manifest are relative to run_dir (the
    manifest directory), not the per-cell directory. Legacy per-cell capture
    files (cell/latency-capture.json) have paths relative to the cell directory;
    this function reads from the main manifest's embedded copy.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text())
    capture = manifest.get("latency_capture") or {}
    if capture.get("version") != 1 or not capture.get("enabled") or capture.get("state") != "stopped":
        raise ValueError("source run has no stopped latency capture; run with --capture-latency and stop it first")

    graph = resolve_graph_dir(manifest, run_dir)
    if graph is None or not graph.is_dir():
        raise ValueError("source run has no managed engine graph; external-engine capture is not supported here")

    legacy_run_dir = manifest.get("run_dir")
    if legacy_run_dir == ".":
        legacy_run_dir = None
    sources = []
    for domain in domains:
        matches = [c for c in manifest.get("cells", [])
                   if c.get("target") == "tau2" and c.get("domain") == domain
                   and c.get("model") == model and c.get("mode") == mode]
        if len(matches) != 1 or matches[0].get("returncode") != 0:
            raise ValueError(f"no successfully completed tau2 cell for {domain}/{mode}/{model}")
        cell = matches[0]
        info = cell.get("latency_capture") or {}

        capture_dir = run_dir
        # The earlier cell-relative format used exactly these four fixed names.
        # Select its base before verification; never retry after a hash mismatch.
        if tuple(info.get(k) for k in ("policy", "functors", "sdk_log", "server_replay")) == (
            "replay-policy/policy.dl", "replay-policy/functors.cpp",
            "latency.jsonl", "server_replay.jsonl",
        ):
            capture_dir = resolve_cell_dir(cell, run_dir, run_dir=legacy_run_dir)
        policy = _verified_file(info, "policy", capture_dir, run_dir=legacy_run_dir)
        functors = _verified_file(info, "functors", capture_dir, run_dir=legacy_run_dir)
        sdk = _verified_file(info, "sdk_log", capture_dir, run_dir=legacy_run_dir)
        replay = _verified_file(info, "server_replay", capture_dir, run_dir=legacy_run_dir)
        rows = []
        for line in replay.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("action_type") != "tool_call":
                raise ValueError("capture contains a non-tool or untyped action; rebuild the engine and recapture")
            if not row.get("session_id") or not row.get("fn_name") or "args" not in row:
                raise ValueError(f"incomplete authorization request in {replay}")
            rows.append(row)
        if not rows:
            raise ValueError(f"empty authorization capture: {replay}")
        if any(row.get("action_idx", 0) != 0 for row in rows):
            raise ValueError("multi-action RPC capture is unsupported by the CheckToolCall loadgen")
        sources.append({"cell": cell, "policy": policy, "functors": functors, "sdk": sdk, "replay": replay,
                        "calls": len(rows), "sessions": len({r["session_id"] for r in rows})})
    if not sources:
        raise ValueError("no source cells selected")
    return manifest, sources


def live_concurrencies(cell: dict) -> list[int]:
    """Read cell/attempt settings, never a run-wide value overwritten on resume."""
    command = cell.get("command") or []
    fallback = cell.get("concurrency")
    if "--max-concurrency" in command:
        fallback = int(command[command.index("--max-concurrency") + 1])
    attempts = cell["latency_capture"].get("attempts", [])
    values = [a.get("concurrency") or fallback for a in attempts] or [fallback]
    if any(value is None or int(value) < 1 for value in values):
        raise ValueError("source cell has no saved concurrency setting")
    return sorted({int(value) for value in values})


def _serialize(cell, manifest_dir: Path | None = None) -> dict:
    """Serialize a CellResult to a dict for JSON storage.

    Path fields are stored relative to manifest_dir when inside it,
    else as absolute strings.
    """
    result = {}
    for key, value in asdict(cell).items():
        if isinstance(value, Path):
            if manifest_dir is not None:
                value = portable_path(value, manifest_dir)
            else:
                value = str(value)
        elif isinstance(value, float) and math.isnan(value):
            value = None
        result[key] = value
    return result


def discover_replay_cells(manifest: dict, manifest_dir: Path | None = None) -> list:
    """Load replay CellResults from a manifest, resolving paths.

    Path fields are resolved against manifest_dir; relative paths
    become absolute, and legacy absolute paths from moved runs
    are rebased.
    """
    from run_latency_microbench import CellResult

    cells = []
    for entry in manifest.get("cells", []):
        values = dict(entry["result"])
        for field in fields(CellResult):
            if values.get(field.name) is None and isinstance(field.default, float):
                values[field.name] = field.default
        for key in ("sim_path", "latency_log", "judge_log", "server_replay_log", "loadgen_log"):
            if values.get(key):
                path_value = values[key]
                if manifest_dir is not None and not Path(path_value).is_absolute():
                    values[key] = manifest_dir / path_value
                else:
                    values[key] = Path(path_value)
        cells.append(CellResult(**values))
    return cells


def replay_markdown(cells, micro) -> str:
    out = ["# Authorization replay from saved benchmark records", "",
           "These measurements replay tool-authorization requests against a copy of the "
           "persisted graph at benchmark stop. Sessions are warmed before timing. "
           "They exclude LLM calls, tool execution, SDK overhead, and historical graph "
           "updates; they do not measure end-to-end task throughput.", "",
           "**Throughput timing**: when a `<trial>.summary.json` sidecar is present "
           "(loadgen version 1+), `calls/s` is computed from the sidecar's monotonic "
           "`elapsed_s` — the authoritative wall time of the timed pass, measured "
           "from before dispatch to after all requests and output writes complete, "
           "excluding warmup, session cleanup and subprocess startup. Legacy artifacts "
           "without a sidecar fall back to the first-to-last timestamp span in the "
           "JSONL (an approximation that under-reports the window for short runs). "
           "The graph state is the persisted snapshot at benchmark stop; repeated "
           "sessions (`--repeat N`) re-exercise the same authorization decisions on "
           "that fixed state and do not reconstruct historical graph changes.", "",
           "Original live timings and task durations are in `live-results.csv` and "
           "`live-results.md`, at the source run's concurrency.", "",
           "| source cell | replay concurrency | trials | calls | p50 | p99 | calls/s | status |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in cells:
        out.append(f"| {cell.source_cell_id} | {cell.concurrency} | {cell.loadgen_n_trials} | "
                   f"{cell.loadgen_n} | {micro._fmt_us(cell.loadgen_p50_us)} | "
                   f"{micro._fmt_us(cell.loadgen_p99_us)} | "
                   f"{micro._fmt_rate(cell.auth_throughput_calls_per_s)} | {cell.returncode} |")
    return "\n".join(out) + "\n"


def replay_latex(cells) -> str:
    lines = ["% Steady-state authorization replay on the persisted graph; no task throughput.",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Source cell & Concurrency & Calls & p50 ($\mu$s) & p99 ($\mu$s) \\", r"\midrule"]
    for cell in cells:
        label = cell.source_cell_id.replace("_", r"\_")
        lines.append(f"{label} & {cell.concurrency} & {cell.loadgen_n} & "
                     f"{cell.loadgen_p50_us:.1f} & {cell.loadgen_p99_us:.1f} " + r"\\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n"


def replay_benchmark(args, micro) -> int:
    source_dir = args.from_benchmark.resolve()
    source_manifest, sources = select_sources(source_dir, domains=args.domains,
                                              model=args.model, mode=args.source_mode)
    if any(c < 1 for c in args.concurrencies) or args.loadgen_repeat < 1 or args.loadgen_trials < 1:
        raise ValueError("concurrency, repeat and replay trials must be positive")
    source_manifest_hash = sha256(source_dir / "manifest.json")
    graph = resolve_graph_dir(source_manifest, source_dir)
    if graph is None:
        raise ValueError("could not resolve graph directory from source manifest")
    print(f"Source run: {source_dir}")
    for source in sources:
        print(f"  {source['cell']['id']}: {source['calls']} calls, {source['sessions']} sessions; "
              f"agent={source['cell']['agent_llm']}, user={source['cell']['user_llm']}")
        if args.loadgen_mode == "session" and max(args.concurrencies) > source["sessions"]:
            print("  ! requested concurrency exceeds captured sessions; achieved load will be lower")
    if args.dry_run:
        print(f"[dry-run] copy stopped graph {graph}; replay at {args.concurrencies}; no model calls")
        return 0
    if not args.loadgen_bin.is_file() or not args.server_bin.is_file():
        raise ValueError("build the local engine and sasy-loadgen before replay")
    server_env = {}
    for item in args.server_env:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--server-env expects KEY=VALUE, got {item!r}")
        server_env[key] = value
    server_env["LLM_ENABLED"] = "false"
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    run_dir = args.output_root.resolve() / timestamp
    run_dir.mkdir(parents=True)
    seed = run_dir / "source-graph"
    copy_closed_graph(graph, seed)
    snapshots = []
    for source in sources:
        entry = source["cell"]
        info = entry["latency_capture"]
        directory = run_dir / "source-records" / entry["id"].replace("/", "_")
        directory.mkdir(parents=True)
        for key, filename in (("policy", "policy.dl"), ("functors", "functors.cpp"),
                              ("sdk_log", "latency.jsonl"), ("server_replay", "server_replay.jsonl")):
            src_path = source[{"sdk_log": "sdk", "server_replay": "replay"}.get(key, key)]
            destination = directory / filename
            shutil.copyfile(src_path, destination)
            if sha256(destination) != info[key + "_sha256"]:
                raise ValueError(f"source capture changed while copying: {key}")
            source[{"sdk_log": "sdk", "server_replay": "replay"}.get(key, key)] = destination
        source["sim"] = directory / "results.json"
        cell_dir = resolve_cell_dir(entry, source_dir, run_dir=source_manifest.get("run_dir"))
        # Use cell_results_file to resolve with proper fallback precedence
        from tau2_results import cell_results_file
        sim_src = cell_results_file(entry, source_dir, run_dir=source_manifest.get("run_dir"))
        if sim_src is None or not sim_src.is_file():
            raise ValueError(
                f"source results not found for {entry['id']}; "
                f"ensure the selected run has results at the expected location"
            )
        shutil.copyfile(sim_src, source["sim"])
        snapshots.append({
            "source_cell_id": entry["id"],
            "directory": portable_path(directory, run_dir),
        })
    if sha256(source_dir / "manifest.json") != source_manifest_hash:
        raise ValueError("source manifest changed while copying; stop the benchmark before replay")
    manifest = {
        "kind": "saved-benchmark-replay", "version": 1,
        "source_run": str(source_dir), "source_manifest_sha256": source_manifest_hash,
        "source_validation_fixture": source_manifest.get("validation_fixture"),
        "graph_state": "persisted-at-stop", "llm_calls": False,
        "engine_sha256": sha256(args.server_bin),
        "source_engine_sha256": sorted({a["engine_sha256"] for source in sources
                                         for a in source["cell"]["latency_capture"].get("attempts", [])
                                         if a.get("engine_sha256")}),
        "loadgen_sha256": sha256(args.loadgen_bin),
        "model": args.model, "source_mode": args.source_mode,
        "concurrencies": args.concurrencies, "loadgen_mode": args.loadgen_mode,
        "loadgen_repeat": args.loadgen_repeat, "loadgen_trials": args.loadgen_trials,
        "sources": [s["cell"] for s in sources], "source_snapshots": snapshots, "cells": [],
    }
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)
    live = []
    for source in sources:
        entry = source["cell"]
        attempts = entry["latency_capture"].get("attempts", [])
        concurrencies = live_concurrencies(entry)
        live_cell = micro.CellResult(
            domain=entry["domain"], mode=entry["mode"],
            concurrency=concurrencies[0] if len(concurrencies) == 1 else None, returncode=0,
            capture_concurrencies=",".join(map(str, concurrencies)),
            wall_clock_s=sum(a["wall_s"] for a in attempts),
            sim_path=source["sim"], latency_log=source["sdk"],
            server_replay_log=source["replay"], source_cell_id=entry["id"],
            source_mode=entry["mode"], agent_llm=entry["agent_llm"], user_llm=entry["user_llm"],
        )
        live.append(micro._reduce_cell(live_cell))
    micro._write_csv(run_dir / "live-results.csv", live)
    (run_dir / "live-results.md").write_text(micro._markdown_tables(live))
    cells = []
    handle = None
    interrupted = False
    try:
        for source in sources:
            entry = source["cell"]
            for concurrency in args.concurrencies:
                name = entry["id"].replace("/", "_") + f"_c{concurrency}"
                cell_dir = run_dir / name
                cell_dir.mkdir()
                replay_graph = run_dir / "server" / name / "graph"
                replay_graph.parent.mkdir(parents=True)
                shutil.copytree(seed, replay_graph)
                cell = micro.CellResult(
                    domain=entry["domain"], mode="sasy", concurrency=concurrency,
                    returncode=1, wall_clock_s=0.0,
                    source_cell_id=entry["id"], source_mode=entry["mode"],
                    agent_llm=entry["agent_llm"], user_llm=entry["user_llm"],
                    measurement_kind="persisted_graph_replay",
                    capture_concurrencies=",".join(map(str, live_concurrencies(entry))),
                )
                cells.append(cell)
                statuses = []
                cell_error = ""
                try:
                    handle = micro._spawn_server(
                        domain=name, run_root=run_dir, addr=args.server_addr,
                        policy_path=source["policy"], server_bin=args.server_bin,
                        extra_env=server_env, keepalive_ms=args.server_keepalive_ms,
                        ready_timeout=args.server_ready_timeout,
                    )
                    host, _, port = args.server_addr.rpartition(":")
                    host = "127.0.0.1" if host in ("0.0.0.0", "[::]", "") else host
                    cell.returncode = 0
                    for trial in range(1, args.loadgen_trials + 1):
                        out = cell_dir / f"loadgen-trial-{trial}.jsonl"
                        rc, wall = micro._run_loadgen(
                            bin_path=args.loadgen_bin, replay=source["replay"], out=out,
                            url=f"https://{host}:{port}", ca=args.loadgen_ca,
                            cert=args.loadgen_cert, key=args.loadgen_key, api_key=args.loadgen_api_key,
                            concurrency=concurrency,
                            channels=args.loadgen_channels if args.loadgen_channels > 0 else concurrency,
                            repeat=args.loadgen_repeat, timeout=args.loadgen_timeout, mode=args.loadgen_mode,
                        )
                        count = len(micro._read_latencies(out))
                        expected = source["calls"] * args.loadgen_repeat
                        statuses.append({"trial": trial, "returncode": rc, "wall_s": wall if math.isfinite(wall) else None,
                                         "expected_calls": expected, "recorded_calls": count})
                        if rc != 0 or count != expected:
                            cell.returncode = rc or 1
                            print(f"! incomplete replay: rc={rc}, {count}/{expected} responses")
                        if out.exists():
                            cell.loadgen_log = out
                except KeyboardInterrupt:
                    cell.returncode = 130
                    raise
                except Exception as exc:
                    cell.returncode = 1
                    cell_error = f"cell failed: {exc}"
                finally:
                    if handle is not None:
                        handle.stop()
                        handle = None
                    try:
                        micro._reduce_cell(cell)
                    except Exception as exc:
                        if cell.returncode == 0:
                            cell.returncode = 1
                        msg = f"reduction failed: {exc}"
                        cell_error = f"{cell_error}; {msg}" if cell_error else msg
                    if cell.returncode != 0:
                        cell.auth_throughput_calls_per_s = float("nan")
                        cell.auth_throughput_std = float("nan")
                    manifest_entry = {"result": _serialize(cell, run_dir), "trials": statuses}
                    if cell_error:
                        manifest_entry["error"] = cell_error
                    manifest["cells"].append(manifest_entry)
                    write_json(manifest_path, manifest)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if handle is not None:
            handle.stop()
        micro._write_outputs(run_dir, cells)
    if interrupted:
        return 130
    return int(any(c.returncode != 0 for c in cells))
