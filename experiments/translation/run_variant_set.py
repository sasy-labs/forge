#!/usr/bin/env python3
"""Run independent policy translations and grade them on held-out tau2 tasks.

Each branch captures traces, samples decision points with its run-index seed,
labels them with the oracle, translates a policy, and grades it. The trajectory
mode also supports shared samples and comparisons with trace-blind branches.
Construction and evaluation tasks must be disjoint.

Prerequisites: a running engine, model and oracle credentials, and an identity
with observability-reader access. Use --preflight to check dependencies.

The translator runs with permission prompts disabled under the same OS user
as this harness. It can modify checkout files and leave background processes;
this harness does not provide process or filesystem isolation. Gate scripts,
functor sources, build tools, the Python environment, and compiled caches are
trusted dependencies and are not protected from modification by that user.

Within an invocation, the harness copies and hashes the inlined policy,
functors and scenario evidence, verifies these copies, and rechecks their
hashes before grading. The parent process retains the original labeled
scenarios in memory. Graded summary entries require provenance stamps.

Resumed stage subsets use snapshots on disk, which have weaker integrity than
in-memory evidence because the translator shares filesystem access. Resume
in the original run directory. The translator's finalize marker is checked
independently and does not replace harness verification.

Usage:
    uv run python experiments/translation/run_variant_set.py \\
        --runs 5 --out outputs/variant-set
    # stage subsets, for resuming a partial set. `grade` re-runs the
    # finalize gate before grading, against the snapshot the labeling
    # invocation persisted under <out>/.harness-run<i>/ — so resume a
    # set in the directory that produced it, not a copy of it:
    uv run python experiments/translation/run_variant_set.py \\
        --runs 5 --out outputs/variant-set --stages grade
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from policy_translation.evaluation.ground_truth_labels import import_labels, parse_entries
from policy_translation.evaluation.labels import (
    scenario_snapshot,
    scenario_snapshot_of_file,
    snapshot_digest,
    snapshot_problem,
    snapshot_record_problem,
)

# The SDK's own include resolver. The verified copy is inlined with
# exactly the function that resolves includes when the policy is
# loaded for grading, so the copy and the graded source agree about
# what the policy says.
from sasy.policy.api import resolve_includes  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# The three held-out tasks used by the paper translation experiment.
DEFAULT_GRADE_TASKS = ["8", "43", "47"]
AIRLINE_TASKS = [str(i) for i in range(50)]
STAGES = ("capture", "sample", "label", "translate", "grade")
TRAJECTORY_MODES = ("independent", "shared", "one-arm")

# The scenarios as they stood when the translator was launched —
# every scenario's name and identity, plus how many of them carried a
# label. Held in memory for a run that labels and translates in one
# invocation; this file exists so a resumed invocation
# (``--stages translate,grade``) has something to compare against, and
# a resumed invocation therefore trusts a file the agent could write.
SNAPSHOT_FILE = "scenarios.snapshot.json"
POLICY_FINALIZE = "scripts/policy_finalize.py"

# Stamped into every graded summary entry this script writes. Pooling
# a prior entry into the reported mean requires it: an entry without
# it was not produced by this harness, so nothing is known about how
# its number was obtained. The stamp is provenance, not proof — it is
# as forgeable as the file it sits in.
HARNESS_STAMP = "run_variant_set/v2"

# Prefix of the per-verification directory. One is created per call to
# `verify_run`, so the bytes a gate ran on are never the bytes another
# gate left behind.
VERIFY_PREFIX = "verify-"

# What each per-verification directory held when its gate finished:
# {str(directory): {"sources": {filename: sha256}, ...}}. Parent
# memory, never read back from disk — the directory itself is writable
# by the agent, so a manifest written there would prove nothing.
_VERIFIED: dict[str, dict[str, Any]] = {}


def run(cmd: list[str], env: dict[str, str] | None = None, cwd: Path = REPO_ROOT, check: bool = False) -> int:
    print(f"    $ {' '.join(cmd[:6])}{' ...' if len(cmd) > 6 else ''}")
    full_env = {**os.environ, **(env or {})}
    proc = subprocess.run(cmd, cwd=cwd, env=full_env)
    if check and proc.returncode != 0:
        # A plain exception, not sys.exit. Under --concurrency the per-run
        # handler catches Exception; SystemExit is not one, so it unwinds past
        # the executor and kills the driver before any run is recorded or the
        # summary written — losing exactly the diagnostics that exist to say
        # which run failed and why.
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd[:6])}")
    return proc.returncode


def capture_pool(grade_tasks: list[str], size: int, seed: int) -> list[str]:
    """Pick capture tasks disjoint from the graded ones."""
    pool = [t for t in AIRLINE_TASKS if t not in grade_tasks]
    rng = random.Random(seed)
    return sorted(rng.sample(pool, min(size, len(pool))), key=int)


def trajectory_access(mode: str, arm: int, source_arm: int) -> dict[str, Any]:
    """Describe what one branch may see in a branch-set experiment.

    ``independent`` preserves the historical runner: every branch owns a fresh
    capture, sample, labels and profile.  ``shared`` copies the source arm's
    complete labeled sample and profile into every branch.  ``one-arm`` copies
    only the source profile; the other branches are authored without a
    trajectory path in their prompt and are verified later on an unlabeled copy
    of the source rows.

    This is prompt/file-layout isolation, not an OS sandbox.  The translating
    process runs as the same user and could search the campaign directory; the
    campaign report records that residual explicitly.
    """
    if mode not in TRAJECTORY_MODES:
        raise ValueError(f"unknown trajectory mode {mode!r}")
    is_source = arm == source_arm
    if mode == "independent" or is_source:
        return {
            "capture": "independent",
            "profile": "own-capture",
            "trajectory_rows": "own-sample",
            "labels": "own-sample",
            "visible_to_author": True,
            "source_arm": arm,
        }
    if mode == "shared":
        return {
            "capture": "shared",
            "profile": "source-arm",
            "trajectory_rows": "source-arm",
            "labels": "source-arm",
            "visible_to_author": True,
            "source_arm": source_arm,
        }
    return {
        "capture": "none",
        "profile": "source-arm",
        "trajectory_rows": "hidden-evaluation-only",
        "labels": "none",
        "visible_to_author": False,
        "source_arm": source_arm,
    }


def write_unlabeled_scenarios(source: Path, destination: Path) -> None:
    """Copy concrete rows while removing every normative answer.

    The blind arms use this only for the parent's post-authoring validation.
    Removing ``label``, ``expected`` and either location of ``ack`` keeps the
    action/graph substrate but gives the gate no decision target it could force
    the branch to fit.
    """
    data = json.loads(source.read_text())
    rows = data.get("scenarios") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{source} has no scenarios to blind")
    blinded: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        row.pop("label", None)
        row.pop("expected", None)
        row.pop("ack", None)
        blinded.append(row)
    if not blinded:
        raise ValueError(f"{source} has no object scenarios to blind")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({**data, "scenarios": blinded}, indent=2) + "\n")


def stage_capture(
    run_dir: Path,
    tasks: list[str],
    cert: Path,
    key: Path,
    *,
    agent_llm: str = "vertex_ai/claude-opus-4-5@20251101",
    user_llm: str = "vertex_ai/claude-opus-4-5@20251101",
) -> Path:
    cap = run_dir / "capture.jsonl"
    done = run_dir / ".capture-complete"
    # Non-empty is not the same as complete, and complete is not the
    # same as applicable. A capture that crashes part-way leaves
    # records on disk; a capture from a previous task split is worse
    # still, because resuming after a change to --capture-tasks or
    # --grade-tasks would sample traces that may cover a task now
    # being graded, leaking evaluation data into the translation. The
    # sentinel therefore records the tasks it was built from.
    stamp = json.dumps(tasks)
    if cap.exists() and cap.stat().st_size and done.exists() and done.read_text().strip() == stamp:
        print(f"    (reusing {cap})")
        return cap
    # Not reusable, so start clean. The capture writer opens the file
    # O_APPEND, so retrying after a partial run would append a complete
    # recording to an abandoned one and then mark the union complete —
    # the sample would carry decisions from a conversation that was
    # thrown away.
    cap.unlink(missing_ok=True)
    done.unlink(missing_ok=True)
    rc = run(
        [
            "uv",
            "run",
            "tau2-instrumented",
            "run",
            "--domain",
            "airline",
            "--task-ids",
            *tasks,
            "--num-trials",
            "1",
            "--agent-llm",
            agent_llm,
            "--user-llm",
            user_llm,
        ],
        env={
            "SASY_POLICY_PATH": str(REPO_ROOT / "experiments/translation/allow_all.dl"),
            "SASY_CAPTURE_FILE": str(cap),
            "SASY_CAPTURE_RUN_ID": run_dir.name,
            # Capture's backward-slice query needs
            # `observability-reader`, which the default client
            # identity does not hold. Without it every record still
            # gets written — just with no graph in it — so the whole
            # run completes and produces nothing usable.
            "TLS_CERT_PATH": str(cert),
            "TLS_KEY_PATH": str(key),
        },
    )
    if rc != 0:
        raise RuntimeError(f"capture exited {rc}; refusing to translate from a partial recording ({cap})")
    if not cap.exists() or not cap.stat().st_size:
        raise RuntimeError(
            "capture command exited zero but recorded no decision points; "
            "tau2 can isolate LLM authentication/startup failures as "
            "AGENT_ERROR while still returning success"
        )
    done.write_text(stamp)
    return cap


def stage_sample(
    run_dir: Path,
    cap: Path,
    seed: int,
    sample: int,
    max_per_tool: int,
    *,
    priority_report: Path | None = None,
    fuzz_per_scenario: int = 0,
    fuzz_families: list[str] | None = None,
) -> Path:
    out = run_dir / "scenarios.json"
    cmd = [
        "uv",
        "run",
        "python",
        "experiments/translation/capture_to_scenarios.py",
        "--capture",
        str(cap),
        "--out",
        str(out),
        "--seed",
        str(seed),
    ]
    # 0 means "keep everything"; stratification only binds when a
    # budget is set, which is the point of allowing full traces.
    if sample:
        cmd += ["--sample", str(sample)]
    if max_per_tool:
        cmd += ["--max-per-tool", str(max_per_tool)]
    if priority_report is not None:
        cmd += ["--priority-report", str(priority_report)]
    run(cmd, check=True)
    if fuzz_per_scenario:
        fuzzed = run_dir / "scenarios.fuzz.json"
        fuzz_cmd = [
            "uv",
            "run",
            "python",
            "experiments/translation/trace_fuzz.py",
            "--scenarios",
            str(out),
            "--out",
            str(fuzzed),
            "--seed",
            str(seed),
            "--per-scenario",
            str(fuzz_per_scenario),
            "--include-originals",
        ]
        if fuzz_families:
            fuzz_cmd += ["--families", *fuzz_families]
        run(fuzz_cmd, check=True)
        fuzzed.replace(out)
    return out


def stage_label(
    run_dir: Path,
    scenarios: Path,
    policy_nl: Path,
    provider: str,
    model: str | None,
    harness: Path,
    *,
    fresh: bool,
    label_schema: str = "decision",
    predicate_vocabulary: Path | None = None,
    ground_truth_labels: Path | None = None,
) -> dict[str, Any]:
    """Label this run's own decision points; return the snapshot.

    The cache is deliberately per-run: a shared cache would make two
    runs that happened to sample the same point share a verdict, which
    is exactly the correlation these runs exist to avoid.

    The returned snapshot is the evidence the rest of the run is
    checked against, and the caller holds it in memory. The copy
    written under `harness` is only for a later invocation that
    resumes at translate or grade.

    The record is ALWAYS written, including for a run where the
    labeler produced no labels at all: it carries the label census, so
    a later invocation reads "this run has no labels" off the record
    rather than off the file it is supposed to be checking.

    `fresh` says the sample stage ran in this invocation, so the
    scenarios file was just rebuilt from the capture and there is
    nothing earlier to preserve. Without it, re-running the label
    stage over a run that already has a record may only ADD
    scenarios, and a run whose record was removed is refused — see
    `write_snapshot`.
    """
    cmd = [
        "uv",
        "run",
        "python",
        "experiments/translation/oracle_label.py",
        "--scenarios",
        str(scenarios),
        "--policy-nl",
        str(policy_nl),
        "--cache",
        str(run_dir / "oracle-cache.json"),
        "--provider",
        provider,
        "--label-schema",
        label_schema,
    ]
    if predicate_vocabulary is not None:
        cmd += ["--predicate-vocabulary", str(predicate_vocabulary)]
    if model:
        cmd += ["--model", model]
    run(cmd, check=True)

    # A judge verdict is a measurement, not ground truth.  Reviewed
    # adjudications therefore enter after the judge has spoken (so its
    # original verdict can be preserved for an accuracy report) but before
    # the immutable pre-authoring snapshot is taken.  The labels file may be
    # a campaign-wide corpus: each independently sampled arm imports only the
    # rows it actually contains, and main() later refuses a campaign in which
    # any supplied entry matched no arm at all.
    if ground_truth_labels is not None:
        data = json.loads(scenarios.read_text())
        entries = parse_entries(json.loads(ground_truth_labels.read_text()))
        scenario_names = {
            scenario.get("name")
            for scenario in data.get("scenarios", [])
            if isinstance(scenario, dict) and isinstance(scenario.get("name"), str)
        }
        matched = [entry for entry in entries if entry["name"] in scenario_names]
        scenarios_by_name = {
            scenario["name"]: scenario
            for scenario in data.get("scenarios", [])
            if isinstance(scenario, dict) and isinstance(scenario.get("name"), str)
        }
        to_import: list[dict[str, Any]] = []
        for entry in matched:
            existing = scenarios_by_name[entry["name"]].get("label") or {}
            if existing.get("source") != "ground-truth":
                to_import.append(entry)
                continue
            existing_outcome = (existing.get("verdict") or {}).get("outcome")
            if existing_outcome != entry["verdict"] or existing.get("ref", "") != entry.get("ref", ""):
                raise RuntimeError(
                    f"reviewed label {entry['name']!r} differs from the ground-truth "
                    "already frozen in this run; use a fresh campaign directory"
                )
        if to_import:
            import_labels(data, to_import)
            scenarios.write_text(json.dumps(data, indent=2) + "\n")
        labels_digest = hashlib.sha256(ground_truth_labels.read_bytes()).hexdigest()
        import_report = {
            "schema": "ground-truth-import/v1",
            "labels": str(ground_truth_labels.resolve()),
            "labels_sha256": labels_digest,
            "entries": len(entries),
            "matched": len(matched),
            "newly_imported": len(to_import),
            "matched_names": sorted(entry["name"] for entry in matched),
        }
        (run_dir / "ground-truth-import.json").write_text(json.dumps(import_report, indent=2) + "\n")
        print(
            f"Matched {len(matched)}/{len(entries)} reviewed ground-truth "
            f"label(s), importing {len(to_import)} into {scenarios}"
        )
    return write_snapshot(harness, scenarios, fresh=fresh)


RECOVERY = (
    "re-run this variant from the sample stage "
    "(--stages sample,label,translate,grade), or point --out at a "
    "fresh directory; those are the only recoveries that do not "
    "start from something already on disk"
)


def harness_dir_for(run_dir: Path) -> Path:
    """Where the parent keeps its own copy of a run's evidence.

    A SIBLING of the run directory: the agent is told to work inside
    the run directory, so a subdirectory of it would sit in the tree
    it is editing.

    The copies do NOT sit at the run directory's depth. Each
    verification gets a fresh subdirectory of its own in here (see
    `verify_run`), so a copied policy is one level further from the
    repository root than the file it was copied from. Nothing depends
    on that distance any more: the copy's `#include` graph is inlined
    before it is hashed, so the copy names no path relative to its own
    file.

    This is not a sandbox and the name does not make it one. The
    agent runs without permission prompts, as the same operating-
    system user, and can write here too. What the location buys is
    only that ordinary work in the run directory does not collide
    with it.
    """
    return run_dir.parent / f".harness-{run_dir.name}"


def no_record_refusal(harness: Path, scenarios: Path) -> str:
    """Why a harness directory holding no snapshot record is refused.

    One wording, used by both the place that would write a record and
    the place that would proceed without one. `stage_label` writes the
    record for every run it labels — including a run whose scenarios
    carry no labels, where the census is simply zero — so a harness
    directory with no record in it either never had a label stage or
    has had its record removed, and this process cannot tell those
    apart. Writing a new record now would baseline whatever is on
    disk, which is exactly the edit a record exists to detect.
    """
    return (
        f"{harness} exists but holds no snapshot of {scenarios}. The "
        f"label stage writes that record for every run it labels — a "
        f"run with no labels included, where the census is zero — so "
        f"a harness directory without one has either lost its record "
        f"or never had a label stage, and recording the file as it is "
        f"now would certify whatever is on disk; {RECOVERY}"
    )


def write_snapshot(harness: Path, scenarios: Path, *, fresh: bool = False) -> dict[str, Any]:
    """Record every scenario's identity and the label census.

    Called by `stage_label` for every run it labels, whether or not
    anything in the file carried a label: the record is what a later
    invocation reads to answer "was this run labeled", and it answers
    that from the census it carries rather than from the file being
    checked.

    Unless `fresh` says the sample stage just rebuilt the scenarios
    file in this invocation, the existing record decides:

      * a record is held — the new one may only ADD scenarios to it.
        A change or a removal is refused. Re-taking a snapshot is how
        a run gets re-baselined: the labeler leaves an already-labeled
        scenario alone when its recorded signatures match, so "just
        re-run from the label stage" over an edited file would relabel
        nothing, snapshot the edit, and make it the evidence. Adding
        is allowed, exactly as it is during translation — an addition
        cannot weaken evidence already recorded;
      * no record, but the harness directory is there — refused, with
        the same wording `verify_run` uses. Deleting the record alone
        must not re-open the re-baseline that keeping it closes;
      * no record and no harness directory — this is the first
        labeling of this run, and the record is written.
    """
    record = scenario_snapshot_of_file(scenarios)
    if not fresh:
        held = load_snapshot(harness)
        if held is not None:
            problem = snapshot_problem(held["identities"], record["identities"])
            if problem:
                raise RuntimeError(
                    f"refusing to re-snapshot {scenarios}: this run "
                    f"already has a snapshot and the file no longer "
                    f"matches it ({problem}). Taking a new one would "
                    f"record the file as it is now as the evidence this "
                    f"run is judged against; {RECOVERY}"
                )
        elif harness.exists():
            raise RuntimeError(f"refusing to snapshot {scenarios}: " + no_record_refusal(harness, scenarios))
    harness.mkdir(parents=True, exist_ok=True)
    (harness / SNAPSHOT_FILE).write_text(
        json.dumps(
            {**record, "scenarios": str(scenarios)},
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    return record


def load_snapshot(harness: Path) -> dict[str, Any] | None:
    """The persisted snapshot for a resumed invocation, or None.

    Never re-takes one from the current file: re-taking after the
    agent has run would record whatever it left behind as the
    evidence, which is precisely the edit a snapshot exists to
    detect. A snapshot whose contents do not hash to their own
    recorded digest is refused rather than replaced, for the same
    reason.

    This file is on disk, in a directory the agent can write, so what
    it says is a claim like any other. It exists because a resumed
    invocation has nothing better; a run that labels and translates in
    one invocation never reads it. `None` means "no snapshot held" —
    `verify_run` decides what that is worth.
    """
    path = harness / SNAPSHOT_FILE
    if not path.exists():
        return None
    record: Any = None
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        # ValueError covers a truncated file (JSONDecodeError).
        record = None
    problem = snapshot_record_problem(record)
    if problem:
        raise ValueError(
            f"{path} is not a usable scenario snapshot ({problem}); it "
            f"cannot be repaired from the scenarios file, so {RECOVERY}"
        )
    return record


def marker_evidence_problem(
    marker: Any,
    scenarios: Path,
    *,
    current_digest: str,
    scenarios_sha256: str,
    labels_present: bool,
    policy_sha256: str | None = None,
    repo_root: Path = REPO_ROOT,
) -> str | None:
    """Why this finalize marker does not certify this run, or None.

    Read this as a check on the PARENT's own finalize invocation, not
    as a way to audit someone else's marker: `verify_run` runs the
    gate and then asks whether the marker that came back describes
    the run it meant to check. It has to name this run's scenarios
    file, it has to have hashed the bytes the gate CONSUMED (the
    startup hashes finalize records, not only the ones it takes at the
    end), and — when those scenarios carry labels — it has to record a
    convergence run over that same file whose evidence digest matches
    it.

    Pure: everything it inspects is an argument.
    """
    if not isinstance(marker, dict):
        return "the finalize marker is not a JSON object"
    named = marker.get("scenarios")
    if not isinstance(named, str) or not named:
        return "the finalize marker names no scenarios file"
    if _resolve(named, repo_root) != scenarios.resolve():
        return f"finalize was run against {named}, not this run's labeled scenarios {scenarios}"
    recorded_sha = marker.get("scenarios_sha256")
    if not isinstance(recorded_sha, str) or not recorded_sha:
        return "the finalize marker records no scenarios_sha256; re-run the finalize gate with this checkout"
    if recorded_sha != scenarios_sha256:
        return (
            f"finalize ran against different bytes of {named} "
            f"({recorded_sha[:12]} != {scenarios_sha256[:12]}); the "
            f"file was edited after it was certified"
        )
    # The hashes finalize took before it read anything. Without them
    # the marker says only what the files looked like once the checks
    # were over, which is not what the checks started on.
    for field, expected in (
        ("scenarios_sha256_at_start", scenarios_sha256),
        ("policy_sha256_at_start", policy_sha256),
    ):
        if expected is None:
            continue
        opening = marker.get(field)
        if not isinstance(opening, str) or not opening:
            return (
                f"the finalize marker records no {field}, so nothing "
                f"says which bytes the gate started on; re-run the "
                f"finalize gate with this checkout"
            )
        if opening != expected:
            return (
                f"the finalize gate started on different bytes "
                f"({field} {opening[:12]} != {expected[:12]}); what it "
                f"checked is not what this harness handed it"
            )
    if not labels_present:
        return None
    if marker.get("convergence_ran") is not True:
        return "the scenarios carry labels but the finalize marker records no convergence run"
    entries = marker.get("evidence")
    entries = entries if isinstance(entries, list) else []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        if not isinstance(path, str) or (_resolve(path, repo_root) != scenarios.resolve()):
            continue
        if entry.get("sha256") != scenarios_sha256:
            return (
                f"the finalize marker's evidence entry for "
                f"{scenarios} hashes other bytes "
                f"({str(entry.get('sha256'))[:12]} != "
                f"{scenarios_sha256[:12]})"
            )
        if entry.get("evidence_digest") != current_digest:
            return (
                f"the finalize marker's evidence digest "
                f"{str(entry.get('evidence_digest'))[:12]} does not "
                f"describe {scenarios} ({current_digest[:12]})"
            )
        return None
    return (
        f"the finalize marker records no convergence over {scenarios}; "
        f"the labels in it were never checked against the policy"
    )


def _resolve(path: str, repo_root: Path) -> Path:
    """A marker path as an absolute path.

    policy_finalize.py runs `make` from the repository root, so a
    relative path recorded in a marker is relative to it.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved.resolve()


def companion_files(policy: Path) -> list[Path]:
    """The functor sources the SDK would pick up beside a policy.

    Mirrors `sasy.policy.api.find_functors`, which looks next to the
    policy file. They are part of what the gate checked — a content
    rule lives in C++ as often as in Datalog — so the verified copy
    carries them, and grading the copy compiles the same sources that
    were verified.
    """
    names = companion_names(policy)
    found: list[Path] = []
    for name in names:
        candidate = policy.parent / name
        if candidate.exists():
            found.append(candidate)
    return found


def companion_names(policy: Path) -> list[str]:
    """Every companion-functor filename the SDK may load for ``policy``."""
    stem = policy.stem
    names = ["functors.cpp", stem.replace("_policy", "_functors") + ".cpp"]
    if "_policy" in stem:
        names.append(stem.split("_policy")[0] + "_functors.cpp")
    return list(dict.fromkeys(names))


def policy_sources(policy: Path) -> dict[str, bytes]:
    """The policy's bytes plus its companion functor sources."""
    sources = {policy.name: policy.read_bytes()}
    for extra in companion_files(policy):
        sources[extra.name] = extra.read_bytes()
    return sources


def source_digests(policy: Path) -> dict[str, str]:
    """sha256 per file of `policy_sources`, keyed by file name."""
    return {name: hashlib.sha256(body).hexdigest() for name, body in policy_sources(policy).items()}


def common_policy_include(policy: Path) -> str:
    """The output-relative include path for the shared policy vocabulary."""
    common = (REPO_ROOT / "policies" / "common_policy.dl").resolve()
    return Path(os.path.relpath(common, policy.parent.resolve())).as_posix()


def policy_preflight_problem(policy: Path) -> str | None:
    """Return an include-resolution problem before an expensive gate run.

    This is intentionally parent-owned and deterministic.  It does not replace
    Souffle validation in the finalize gate; it catches malformed/missing
    include graphs early enough to give a bounded repair attempt a useful,
    local error instead of treating the candidate as merely non-convergent.
    """
    if not policy.exists():
        return f"translation produced no policy at {policy}"
    try:
        inlined_policy(policy)
    except (OSError, RuntimeError, ValueError) as exc:
        return f"could not resolve the policy include graph: {exc}"
    return None


def capture_candidate(policy: Path) -> dict[str, bytes]:
    """Capture exactly the generated policy/functor substrate in memory."""
    if not policy.exists():
        raise RuntimeError(f"translation produced no policy at {policy}")
    return policy_sources(policy)


def restore_candidate(policy: Path, candidate: dict[str, bytes]) -> None:
    """Restore one generated candidate and remove later companion additions."""
    allowed = {policy.name, *companion_names(policy)}
    unknown = set(candidate) - allowed
    if unknown:
        raise RuntimeError(f"candidate contains unexpected generated sources: {sorted(unknown)}")
    for name in allowed:
        path = policy.parent / name
        if name in candidate:
            path.write_bytes(candidate[name])
        elif path.exists():
            path.unlink()


def candidate_digests(candidate: dict[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(body).hexdigest() for name, body in sorted(candidate.items())}


def assessment_score(assessment: dict[str, Any], candidate: dict[str, bytes]) -> tuple[int, ...]:
    """A monotone ordering for repair candidates; larger is better.

    Executability and full convergence dominate row counts.  Among incomplete
    candidates, fewer blockers wins, then more matches, then the smaller
    generated substrate.  Equal candidates do not replace the incumbent.
    """
    summary = assessment.get("summary")
    measured = isinstance(summary, dict)
    values = summary if measured else {}
    blockers = sum(int(values.get(field, 0) or 0) for field in ("mismatch", "errors", "unlabeled")) + int(
        assessment.get("unacked_disputes", 0) or 0
    )
    size = sum(len(body) for body in candidate.values())
    return (
        int(assessment.get("preflight_problem") is None),
        int(measured),
        int(assessment.get("converged") is True),
        -blockers,
        int(values.get("match", 0) or 0),
        -size,
    )


def assess_translation_candidate(
    policy: Path,
    scenarios: Path,
    spec: Path,
    report_path: Path,
    *,
    trajectory_visible: bool = True,
) -> dict[str, Any]:
    """Compile/replay a candidate into a fresh parent-owned repair report."""
    report_path.unlink(missing_ok=True)
    problem = policy_preflight_problem(policy)
    if problem:
        result = {
            "preflight_problem": problem,
            "converged": False,
            "summary": None,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2) + "\n")
        return result
    if not trajectory_visible:
        proc = subprocess.run(
            ["make", "souffle-validate", f"FILE={policy}"],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )
        result = {
            "preflight_problem": None,
            "command_exit": proc.returncode,
            "converged": False,
            "summary": (
                {"total": 0, "match": 0, "mismatch": 0, "errors": 0, "unlabeled": 0} if proc.returncode == 0 else None
            ),
            "unacked_disputes": 0,
            "report": None,
            "output": (proc.stdout + proc.stderr)[-4000:],
            "blind_assessment": "include resolution and Souffle validation only",
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2) + "\n")
        return result
    command = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "translation" / "policy_convergence.py"),
        "--policy",
        str(policy),
        "--scenarios",
        str(scenarios),
        "--policy-nl",
        str(spec),
        "--report",
        str(report_path),
    ]
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    report: dict[str, Any] = {}
    if report_path.exists():
        try:
            loaded = json.loads(report_path.read_text())
            if isinstance(loaded, dict):
                report = loaded
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "preflight_problem": None,
        "command_exit": proc.returncode,
        "converged": report.get("converged") is True,
        "summary": report.get("summary"),
        "unacked_disputes": report.get("unacked_disputes", 0),
        "report": str(report_path) if report else None,
        "output": (proc.stdout + proc.stderr)[-4000:],
    }


def repair_translation(
    run_dir: Path,
    policy: Path,
    scenarios: Path,
    spec: Path,
    profile: Path,
    snapshot: dict[str, Any] | None,
    harness: Path,
    *,
    rounds: int,
    initial_error: str,
    trajectory_visible: bool,
) -> Path:
    """Run bounded, local repair and retain only a strictly better candidate."""
    repair_root = harness / "repair"
    if repair_root.is_symlink():
        raise RuntimeError(f"{repair_root} is a symbolic link; refusing to write repair evidence through it")
    repair_root.mkdir(parents=True, exist_ok=True)

    # The first authoring pass may add top-up rows.  Once repair begins, that
    # resulting evidence is frozen byte-for-byte: repair is policy work, not a
    # second opportunity to alter the questions or their labels.
    repair_evidence = scenarios.read_bytes()
    if snapshot is not None:
        current = scenario_snapshot(json.loads(repair_evidence))
        problem = snapshot_problem(snapshot["identities"], current)
        if problem:
            raise RuntimeError(
                f"the scenarios changed before repair began: {problem}; repair cannot re-baseline labeled evidence"
            )

    best = capture_candidate(policy)
    initial_assessment = assess_translation_candidate(
        policy,
        scenarios,
        spec,
        repair_root / "assessment-0.json",
        trajectory_visible=trajectory_visible,
    )
    best_score = assessment_score(initial_assessment, best)
    history: dict[str, Any] = {
        "schema": "translation-repair-history/v1",
        "policy": str(policy),
        "scenarios": str(scenarios),
        "round_budget": rounds,
        "initial_gate_error": initial_error,
        "attempts": [
            {
                "round": 0,
                "source_digests": candidate_digests(best),
                "score": list(best_score),
                "assessment": initial_assessment,
                "accepted": True,
            }
        ],
    }

    exact_include = common_policy_include(policy)
    last_error = initial_error
    for round_number in range(1, rounds + 1):
        assessment_path = repair_root / f"assessment-{round_number}.json"
        evidence_instruction = (
            f"Do not edit the policy specification at {spec}, the system "
            f"profile at {profile}, or the scenario evidence at {scenarios}."
            if trajectory_visible
            else (
                f"Do not inspect captures, scenarios, labels, or other branches. "
                f"The parent assessment is compile-only for this blind arm. Do "
                f"not edit the policy specification at {spec} or the system "
                f"profile at {profile}."
            )
        )
        repair_prompt = (
            f"Repair the generated Souffle policy at {policy}. This is bounded "
            f"repair round {round_number}/{rounds}, not a fresh translation.\n\n"
            f"The parent-owned gate rejected the current candidate:\n{last_error}\n\n"
            f"A machine-readable assessment of the incumbent is at "
            f"{repair_root / f'assessment-{round_number - 1}.json'}. "
            "Change only the localized Datalog clause(s) or companion functor "
            f"implementation needed to remove those blockers. {evidence_instruction} "
            "Do not select or copy a "
            "whole alternative branch.\n\n"
            f"If the policy includes the common vocabulary, the directive for "
            f'this output is exactly: #include "{exact_include}"\n'
            "Validate include resolution and compilation immediately after the "
            "edit, then run the convergence/finalize loop."
        )
        prompt_path = run_dir / f"repair-round-{round_number}.txt"
        prompt_path.write_text(repair_prompt)
        marker_dir = run_dir / f".policy-repair-round-{round_number}"
        if marker_dir.is_symlink():
            raise RuntimeError(f"{marker_dir} is a symbolic link; refusing to clear it")
        if marker_dir.exists():
            shutil.rmtree(marker_dir)
        rc = run(
            [
                "claude",
                "--dangerously-skip-permissions",
                "-p",
                "--plugin-dir",
                "./plugins/policy-compiler",
                repair_prompt,
            ],
            env={"SASY_POLICY_RUN_DIR": str(marker_dir)},
        )
        if scenarios.read_bytes() != repair_evidence:
            restore_candidate(policy, best)
            raise RuntimeError(
                f"repair round {round_number} changed {scenarios}; policy repair may not alter frozen evidence"
            )
        candidate = capture_candidate(policy)
        assessment = assess_translation_candidate(
            policy,
            scenarios,
            spec,
            assessment_path,
            trajectory_visible=trajectory_visible,
        )
        score = assessment_score(assessment, candidate)
        accepted = score > best_score
        attempt = {
            "round": round_number,
            "author_exit": rc,
            "source_digests": candidate_digests(candidate),
            "score": list(score),
            "assessment": assessment,
            "accepted": accepted,
        }
        if accepted:
            best = candidate
            best_score = score
            try:
                verified = verify_run(run_dir, scenarios, policy, snapshot, harness, policy_nl=spec)
            except RuntimeError as exc:
                last_error = str(exc)
                attempt["gate_error"] = last_error
            else:
                attempt["strict_gate"] = "passed"
                history["attempts"].append(attempt)
                history["result"] = "strict-gate-passed"
                (repair_root / "history.json").write_text(json.dumps(history, indent=2) + "\n")
                return verified
        else:
            restore_candidate(policy, best)
            attempt["rolled_back"] = True
            last_error = (
                f"round {round_number} did not improve the parent assessment "
                f"({list(score)} <= {list(best_score)}); the incumbent was restored"
            )
        history["attempts"].append(attempt)

    restore_candidate(policy, best)
    history["result"] = "budget-exhausted"
    history["final_source_digests"] = candidate_digests(best)
    (repair_root / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    raise RuntimeError(
        f"translation remained unfinalized after {rounds} repair round(s); "
        f"best candidate restored; last blocker: {last_error}"
    )


def inlined_policy(policy: Path) -> bytes:
    """The policy with its whole `#include` graph resolved into it.

    A generated policy includes `policies/common_policy.dl`, and that
    file — plus anything it includes in turn — is as much a part of
    the evaluated policy as the rules beside it. It lives in the
    checkout, so hashing the policy file alone would leave the rules
    that actually decide outside the gate. Inlining first puts them
    inside it: the copy that is hashed, gated and graded carries no
    includes at all, and editing an include target afterwards changes
    nothing about what was measured.

    `sasy.policy.api.resolve_includes` is used rather than a local
    copy of the same idea, because it is the resolver that runs when
    the policy is loaded for grading.
    """
    text = resolve_includes(policy)
    if not text.endswith("\n"):
        text += "\n"
    return text.encode("utf-8")


def stage_copies(
    verify_dir: Path,
    policy: Path,
    scenarios_name: str,
    scenario_bytes: bytes,
) -> tuple[Path, Path]:
    """Put everything about to be gated inside `verify_dir`.

    The policy inlined, every companion functor source byte for byte,
    and the scenarios bytes the caller already read. Nothing here
    reads the scenarios file a second time: two reads are two
    different files if something is rewriting it.
    """
    policy_copy = verify_dir / policy.name
    policy_copy.write_bytes(inlined_policy(policy))
    for extra in companion_files(policy):
        (verify_dir / extra.name).write_bytes(extra.read_bytes())
    scenarios_copy = verify_dir / scenarios_name
    scenarios_copy.write_bytes(scenario_bytes)
    return policy_copy, scenarios_copy


def verified_copy_problem(verified: Path, sources: dict[str, str]) -> str | None:
    """Why the verified copy is no longer what the gate ran on, or None.

    Compares the copy AND the companion functor sources beside it
    against the hashes taken when the gate finished — the same set of
    files `sasy.policy.api.find_functors` will pick up. A functor
    source added afterwards is caught by the set differing, not only
    by a hash differing: dropping a new `*_functors.cpp` next to a
    policy adds C++ to it without touching a byte of the policy.
    """
    if verified.is_symlink():
        return f"{verified} is a symbolic link, not the copy this gate wrote"
    try:
        now = source_digests(verified)
    except OSError as e:
        return f"could not re-read {verified}: {e}"
    if now == sources:
        return None
    parts = []
    added = sorted(set(now) - set(sources))
    missing = sorted(set(sources) - set(now))
    changed = sorted(name for name in set(now) & set(sources) if now[name] != sources[name])
    if changed:
        parts.append(f"changed: {', '.join(changed)}")
    if added:
        parts.append(f"appeared since the gate ran: {', '.join(added)}")
    if missing:
        parts.append(f"gone: {', '.join(missing)}")
    return f"the verified copy in {verified.parent} is not the set of files the gate certified ({'; '.join(parts)})"


def convergence_report_problem(report_path: Path, policy: Path, scenarios: Path) -> str | None:
    """Why the gate's convergence report does not cover this run, or None.

    The report is the artifact `policy_convergence.py` writes as it
    runs. Requiring it — in a directory this process created, so
    nothing left over can stand in for it — is how the harness tells
    "convergence ran and converged" from "a marker says it did".
    A gate script rewritten to fake both is still a gate script this
    harness runs (see the module docstring); this check catches the
    marker being written without the work, not the work being faked.
    """
    if not report_path.exists():
        return (
            f"the gate wrote no convergence report at {report_path}, so "
            f"nothing shows the labels were compared against the policy"
        )
    try:
        report = json.loads(report_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return f"could not read the convergence report {report_path}: {e}"
    if not isinstance(report, dict):
        return f"{report_path} is not a JSON object"
    for field, expected in (("policy", policy), ("scenarios_file", scenarios)):
        named = report.get(field)
        if not isinstance(named, str) or Path(named) != expected:
            return f"the convergence report's {field} is {named!r}, not {str(expected)!r}"
    if report.get("converged") is not True:
        return "the convergence report does not say the policy converged"
    summary = report.get("summary")
    total = summary.get("total") if isinstance(summary, dict) else None
    if not isinstance(total, int) or total <= 0:
        return "the convergence report covers no scenarios"
    return None


def verify_run(
    run_dir: Path,
    scenarios: Path,
    policy: Path,
    snapshot: dict[str, Any] | None,
    harness: Path,
    *,
    policy_nl: Path | None = None,
) -> Path:
    """Gate a copy of this run's evidence; return the copy to grade.

    The order matters, and it is the whole point: the artifacts are
    copied FIRST, and everything afterwards — hashing, the gate,
    grading — happens on the copies. The originals are read once. A
    process rewriting the run directory while the gate runs is then
    rewriting files nothing is looking at any more.

      1. the scenarios bytes are read ONCE and compared against the
         snapshot the parent took before the agent started — every
         scenario that was labeled is still there and unchanged,
         additions allowed. Whether this run was labeled comes from
         that snapshot, never from the file being checked;
      2. those bytes, the policy with its `#include` graph inlined,
         and every companion functor source are written into a
         directory this process just created, and hashed there;
      3. the finalize gate is RE-RUN by this process against the
         COPIES (validate, config-sync, scenario harness,
         convergence), with its marker directory inside that same
         directory;
      4. the copies are re-hashed and must be unchanged, the marker
         for the copied policy must exist, and it must record the
         startup hashes of the copies it consumed;
      5. when the run carries labels, the convergence report the gate
         should have written has to be there, name the copies, and
         say the policy converged.

    Returns the path of the copied policy. `stage_grade` re-hashes it
    and its companions again immediately before grading.

    What this does NOT establish is in the module docstring: the gate
    scripts, the Makefile targets and the Python environment are the
    trusted base, and an agent that rewrites them is not caught here.
    """
    if not policy.exists():
        raise RuntimeError(f"translation produced no policy at {policy}")
    # The policy text drives the gate's prompt-signature check. Left
    # to fall through, a path that does not exist would drop the
    # argument and the gate would run without that check, reporting
    # PASS because one of its inputs was missing — the same failure
    # the missing system profile is refused for in `main`.
    if policy_nl is not None and not Path(policy_nl).exists():
        raise RuntimeError(
            f"no policy text at {policy_nl}; the convergence report "
            f"recomputes each oracle label's prompt signature from it, "
            f"so running the gate without it would drop that check "
            f"silently"
        )
    try:
        scenario_bytes = scenarios.read_bytes()
        scenario_data = json.loads(scenario_bytes)
        current = scenario_snapshot(scenario_data)
    except (json.JSONDecodeError, OSError, ValueError) as e:
        raise RuntimeError(f"could not read the evidence for {policy}: {e}") from e

    # 1. The snapshot decides, in both directions. Its identity check
    #    runs before anything reads a label off the file: stripping
    #    every label changes every identity, so that refusal is
    #    reached rather than answered by the stripped file.
    if snapshot is not None:
        problem = snapshot_record_problem(snapshot)
        if problem:
            raise RuntimeError(f"the snapshot held for {run_dir} is not usable ({problem}); {RECOVERY}")
        problem = snapshot_problem(snapshot["identities"], current)
        if problem:
            raise RuntimeError(
                f"the scenarios changed since they were labeled: "
                f"{problem} — verdicts, provenance, the action they "
                f"judge and the graph behind it are evidence, not "
                f"working state"
            )
        labels_present = snapshot["labeled"] > 0
    else:
        # No snapshot in memory and no record on disk. `stage_label`
        # writes that record for every run it labels — a run with no
        # labels included — so its absence beside an existing harness
        # directory means the record was removed, or no label stage
        # ever ran here. Either way there is nothing to check the file
        # against, including the case where the labels were removed
        # from the scenarios file in the same move, which is what
        # makes reading "were there labels?" off that file useless.
        #
        # What this does not reach: deleting the whole directory
        # leaves a resumed invocation with nothing to tell a labeled
        # run from one that never was. Resuming is only as good as the
        # directory being resumed from; see the module docstring.
        if harness.exists():
            raise RuntimeError(no_record_refusal(harness, scenarios))
        # Reading the file here can only ADD a refusal: an unlabeled
        # run needs no snapshot, a labeled one has just been shown not
        # to have its own.
        if any(s.get("label") for s in scenario_data.get("scenarios", []) if isinstance(s, dict)):
            raise RuntimeError(
                f"{scenarios} carries labels but this invocation holds "
                f"no snapshot of them from before translation, and "
                f"taking one now would certify whatever the agent left "
                f"behind; {RECOVERY}"
            )
        labels_present = False

    # 2. A directory this process created, holding the only bytes
    #    that are going to be measured.
    if harness.is_symlink():
        raise RuntimeError(
            f"{harness} is a symbolic link; refusing to keep this run's evidence somewhere else is pointing"
        )
    stale_marker = harness / ".policy-run"
    if stale_marker.is_symlink():
        raise RuntimeError(
            f"{stale_marker} is a symbolic link; this harness only "
            f"ever creates that name as a directory, so something "
            f"else put it there"
        )
    harness.mkdir(parents=True, exist_ok=True)
    verify_dir = Path(tempfile.mkdtemp(prefix=VERIFY_PREFIX, dir=harness))
    try:
        policy_copy, scenarios_copy = stage_copies(verify_dir, policy, scenarios.name, scenario_bytes)
        before = source_digests(policy_copy)
    except OSError as e:
        raise RuntimeError(f"could not copy {policy} and {scenarios} into {verify_dir}: {e}") from e
    scenarios_sha = hashlib.sha256(scenario_bytes).hexdigest()
    print(f"    verifying a copy in {verify_dir}")

    # 3. The gate, against the copies, into a marker directory inside
    #    the same fresh directory — so no earlier marker can stand in
    #    for this one and there is nothing to delete first.
    marker_dir = verify_dir / ".policy-run"
    cmd = [
        "uv",
        "run",
        "python",
        POLICY_FINALIZE,
        "--policy",
        str(policy_copy),
        "--scenarios",
        str(scenarios_copy),
    ]
    if policy_nl is not None:
        # Lets the convergence report recompute each oracle label's
        # prompt signature from the scenario it judged. Checked for
        # existence at the top of this function, not dropped here.
        cmd += ["--policy-nl", str(Path(policy_nl).resolve())]
    rc = run(cmd, env={"SASY_POLICY_RUN_DIR": str(marker_dir)})
    if rc != 0:
        raise RuntimeError(
            f"the finalize gate failed when this harness re-ran it for {policy} (exit {rc}); the run is not measurable"
        )

    # 4. The copies, and the marker that should describe them.
    problem = verified_copy_problem(policy_copy, before)
    if problem:
        raise RuntimeError(f"{problem}; nothing certifies the bytes now in {verify_dir}")
    try:
        if hashlib.sha256(scenarios_copy.read_bytes()).hexdigest() != (scenarios_sha):
            raise RuntimeError(
                f"the copy of {scenarios.name} changed while the gate "
                f"was running; what was checked is not what was copied"
            )
    except OSError as e:
        raise RuntimeError(f"could not re-read {scenarios_copy} after the gate: {e}") from e
    digest = before[policy_copy.name]
    marker_path = marker_dir / "finalized" / f"{digest}.ok"
    if not marker_path.exists():
        raise RuntimeError(
            f"the finalize gate wrote no marker for the content of {policy_copy} (expected {marker_path})"
        )
    try:
        marker = json.loads(marker_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"could not read {marker_path}: {e}") from e
    evidence = snapshot_digest(current)
    problem = marker_evidence_problem(
        marker,
        scenarios_copy,
        current_digest=evidence,
        scenarios_sha256=scenarios_sha,
        labels_present=labels_present,
        policy_sha256=digest,
    )
    if problem:
        raise RuntimeError(f"the finalize gate this harness ran does not certify {policy}: {problem}")

    # 5. The work itself left an artifact; require it.
    if labels_present:
        problem = convergence_report_problem(
            scenarios_copy.with_suffix(".convergence.json"),
            policy_copy,
            scenarios_copy,
        )
        if problem:
            raise RuntimeError(f"the finalize gate this harness ran does not certify {policy}: {problem}")

    _VERIFIED[str(verify_dir)] = {
        "sources": before,
        "policy_sha256": digest,
        "evidence_digest": evidence,
        "scenarios_sha256": scenarios_sha,
    }
    return policy_copy


def stage_translate(
    run_dir: Path,
    scenarios: Path,
    spec: Path,
    profile: Path,
    snapshot: dict[str, Any] | None,
    harness: Path,
    *,
    trajectory_visible: bool = True,
    repair_rounds: int = 1,
) -> Path:
    policy = run_dir / "airline_policy.dl"
    exact_include = common_policy_include(policy)
    trajectory_prompt = (
        (
            f"Captured, oracle-labeled decision points from a live run are "
            f"at {scenarios} — use them as the Mode L trace evidence and "
            f"run the convergence loop against them.\n"
            f"Those scenarios are the evidence you are measured against, "
            f"the label AND the decision it judges alike: do not remove a "
            f"labeled scenario, do not edit any label's verdict, source, "
            f"context_mode, policy_class or expected assertion, and do "
            f"not edit the scenario's action (fn_name, args), graph, "
            f"roles or entity either. Finalize against {scenarios} "
            f"itself, not a copy of it. ADDING author-written top-up "
            f"scenarios to that file is expected, as is recording a "
            f"disagreement you cannot resolve as an `ack` on the "
            f"scenario. After you exit, this harness re-runs the finalize "
            f"gate itself against the file above and abandons the run if "
            f"anything it snapshotted before you started has changed. "
            f"The convergence report localizes structured disagreements. "
            f"If its route is `environment`, do not edit the policy to fit "
            f"the label: repair or escalate the fact producer/label. For a "
            f"clause disagreement, repair only the localized clause and add "
            f"a concrete boundary twin; never select or merge a whole blind "
            f"branch from one row. Predicate labels are advisory evidence, "
            f"not runtime EDB facts.\n\n"
        )
        if trajectory_visible
        else (
            "This is an oracle-blind Datalog branch. Translate from the English "
            "policy and the system profile only. Do not inspect captures, "
            "scenario files, labels, policies, prompts, convergence reports or "
            "other branch directories in this campaign. Create independent "
            "author-written boundary tests and run the normal validation/finalize "
            "loop on those tests. The parent harness will evaluate the resulting "
            "policy on held-out concrete rows only after authoring has ended.\n\n"
        )
    )
    prompt = (
        f"Translate the policy spec at {spec} into Soufflé Datalog "
        f"using the /write-policy skill.\n\n"
        f"{trajectory_prompt}"
        f"A trace-derived system profile is at {profile}; it lists the "
        f"agent names, which roles each one speaks with, and the "
        f"observed tool argument and result fields.\n\n"
        f"For this output location, the shared vocabulary directive is "
        f'exactly `#include "{exact_include}"`. Use that path if the '
        f"policy includes common_policy.dl, and resolve the include graph "
        f"immediately after writing it.\n\n"
        f"Write the policy to {policy}."
    )
    (run_dir / "prompt.txt").write_text(prompt)
    # NOT resolved: `Path.resolve()` follows a symbolic link, and
    # handing rmtree the target of one deletes that tree instead of
    # the link — rmtree's own refusal to walk a link never fires,
    # because it is not given a link. The name is one the agent knows
    # and the run directory is one it can write.
    agent_marker_dir = run_dir / ".policy-run"
    if agent_marker_dir.is_symlink():
        raise RuntimeError(
            f"{agent_marker_dir} is a symbolic link; this harness only "
            f"ever creates that name as a directory, and deleting what "
            f"it points at is not something to do on a guess"
        )
    # The agent's own marker directory, for the Stop hook that gates
    # its session. Markers are keyed by policy content alone, so one
    # left by an earlier attempt would satisfy that hook without the
    # agent re-running anything; clear them so its loop behaves. This
    # harness does not read what lands here — see verify_run.
    if agent_marker_dir.exists():
        shutil.rmtree(agent_marker_dir)
    rc = run(
        [
            "claude",
            "--dangerously-skip-permissions",
            "-p",
            "--plugin-dir",
            "./plugins/policy-compiler",
            prompt,
        ],
        # Per-session finalize markers. Concurrent runs share one
        # checkout, and a project-root marker directory would let one
        # session's unfinalized policy block another's Stop hook.
        env={"SASY_POLICY_RUN_DIR": str(agent_marker_dir)},
    )
    # The model writes the policy file well before validation finishes,
    # so the file existing proves nothing. With no repair budget retain the
    # historical refusal. With a budget, a nonzero author/Stop-hook exit is a
    # localizable failure: only a later parent-owned strict gate may accept it.
    if rc != 0 and repair_rounds == 0:
        raise RuntimeError(
            f"translation exited {rc}; refusing to grade an unfinalized policy (see {run_dir}/prompt.txt)"
        )
    if not policy.exists():
        raise RuntimeError(f"translation produced no policy at {policy}")
    # Logged, never believed. Every field in the agent's marker is
    # derivable from files the agent can read and write, so it says
    # what the session claims, not what happened. The gate below is
    # the parent re-running the checks itself.
    agent_digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    agent_marker = agent_marker_dir / "finalized" / f"{agent_digest}.ok"
    print(
        f"    agent finalize marker for the final policy: "
        f"{'present' if agent_marker.exists() else 'absent'} "
        f"(not consulted)"
    )
    preflight_problem = policy_preflight_problem(policy)
    if preflight_problem is not None:
        if repair_rounds == 0:
            raise RuntimeError(preflight_problem)
        return repair_translation(
            run_dir,
            policy,
            scenarios,
            spec,
            profile,
            snapshot,
            harness,
            rounds=repair_rounds,
            initial_error=preflight_problem,
            trajectory_visible=trajectory_visible,
        )
    try:
        return verify_run(run_dir, scenarios, policy, snapshot, harness, policy_nl=spec)
    except RuntimeError as exc:
        if repair_rounds == 0:
            raise
        return repair_translation(
            run_dir,
            policy,
            scenarios,
            spec,
            profile,
            snapshot,
            harness,
            rounds=repair_rounds,
            initial_error=str(exc),
            trajectory_visible=trajectory_visible,
        )


SIM_DIR = REPO_ROOT / "experiments/tau2-bench/data/simulations"


def stage_grade(
    run_dir: Path,
    policy: Path,
    tasks: list[str],
    trials: int,
    *,
    scenarios: Path,
    snapshot: dict[str, Any] | None,
    harness: Path,
    verified: Path | None = None,
    policy_nl: Path | None = None,
    agent_llm: str = "vertex_ai/claude-opus-4-5@20251101",
    user_llm: str = "vertex_ai/claude-opus-4-5@20251101",
    cert: Path | None = None,
    key: Path | None = None,
) -> dict[str, Any]:
    """Grade one variant, pinned to its own policy per session.

    Grades the VERIFIED copy of the policy, never the run
    directory's file. `verified` is passed when this invocation just
    translated (the gate has already run); otherwise — resuming with
    `--stages grade` — the gate runs here first, so a policy that was
    never certified, or one whose scenarios were edited afterwards,
    cannot be graded into the results.

    The copy and its companion functor sources are re-hashed here,
    immediately before tau2 is started, against what the gate left.
    They sit in a directory the agent can write, and tau2 reads them
    once per trial, so the check is close to the read rather than
    close to the gate. It does not cover the window while tau2 runs;
    nothing in this harness does (see the module docstring).

    Deliberately NOT scripts/evaluation/run_benchmark.py: that hot-
    swaps the tenant default, and the server holds one of those at a
    time, so variants would serialise and would clobber anything else
    using the same server. tau2-instrumented instead pins every trial
    with sasy.session(sid, policy=...) from SASY_POLICY_PATH (see
    experiments/tau2/forge_tau2/cli.py), and the server
    dedupes uploads by content hash — so variants can grade
    concurrently, against a shared server, without interfering.
    """
    if verified is None:
        verified = verify_run(
            run_dir,
            scenarios,
            policy,
            snapshot,
            harness,
            policy_nl=policy_nl,
        )
    record = _VERIFIED.get(str(verified.parent))
    if record is None:
        raise RuntimeError(
            f"{verified} was not verified by this invocation, so "
            f"nothing here knows what it is supposed to contain; "
            f"{RECOVERY}"
        )
    problem = verified_copy_problem(verified, record["sources"])
    if problem:
        raise RuntimeError(f"refusing to grade: {problem}")
    stamp = {
        "policy_sha256": record["policy_sha256"],
        "evidence_digest": record["evidence_digest"],
        "harness": HARNESS_STAMP,
    }
    # The graded policy's hash is in the file name because that is the
    # only place tau2 will carry it: its simulation JSON records the
    # domain, the trial count and the git commit, and nothing about
    # the policy that was enforced.
    save_to = f"variantset_{run_dir.name}_{record['policy_sha256'][:12]}"
    # tau2 1.0 writes <save_to>/results.json; earlier releases wrote
    # the flat <save_to>.json, which is read when it is the only one.
    out_dir = SIM_DIR / save_to
    legacy = SIM_DIR / f"{save_to}.json"
    # tau2 resumes into an existing file of the same name, and a failed
    # command leaves the previous round's results in place. Reading
    # that back would report an earlier policy's score as this
    # variant's, so clear the target first -- in both layouts -- and
    # refuse to parse unless the command succeeded.
    if out_dir.is_dir():
        shutil.rmtree(out_dir)
    if legacy.exists():
        legacy.unlink()
    rc = run(
        [
            "uv",
            "run",
            "tau2-instrumented",
            "run",
            "--domain",
            "airline",
            "--task-ids",
            *tasks,
            "--num-trials",
            str(trials),
            "--agent-llm",
            agent_llm,
            "--user-llm",
            user_llm,
            "--save-to",
            save_to,
        ],
        # The verified copy, not the run directory's file: what was
        # checked and what is measured have to be the same bytes.
        env={
            "SASY_POLICY_PATH": str(verified.resolve()),
            **({"TLS_CERT_PATH": str(cert)} if cert is not None else {}),
            **({"TLS_KEY_PATH": str(key)} if key is not None else {}),
        },
    )
    if rc != 0:
        raise RuntimeError(f"grading exited {rc} for {verified}")
    out = out_dir / "results.json"
    if not out.exists() and legacy.exists():
        out = legacy
    result = read_reward(out, tasks, stamp=stamp)
    expected = len(tasks) * trials
    if result["sims"] != expected:
        raise RuntimeError(
            f"grading produced {result['sims']} simulations, expected "
            f"{expected} — partial results are not a measurement"
        )
    return result


def read_reward(path: Path, tasks: list[str], *, stamp: dict[str, str] | None = None) -> dict[str, Any]:
    """Pull mean reward + per-task pass counts out of a tau2 run.

    `stamp` is the provenance of the policy this file is supposed to
    be the result of, and it is copied into the returned record.
    tau2's simulation JSON records nothing that identifies the
    enforced policy, so the binding available is the file NAME, and
    `stage_grade` builds that name from the graded copy's hash,
    deletes anything already sitting at it, and refuses to come here
    at all unless tau2 exited zero. The name therefore carries the
    graded policy's hash BY CONSTRUCTION; this function does not
    re-check it, because checking it here would only compare
    `stage_grade`'s arithmetic against itself. That is a weak binding
    and it is the only one there is — the numbers still come out of a
    file in a directory the agent can write.
    """
    sims: list[dict] = []
    stamped: dict[str, Any] = {"provenance": stamp} if stamp is not None else {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        print(f"    WARNING: no readable simulation file at {path}")
        return {"reward": None, "sims": 0, "per_task": {}, **stamped}
    found = data.get("simulations") if isinstance(data, dict) else data
    if isinstance(found, list):
        sims = [s for s in found if isinstance(s, dict)]
    if not sims:
        return {"reward": None, "sims": 0, "per_task": {}, **stamped}
    # Keep each reward with its own simulation. Filtering the reward
    # list on its own shortens it, and the zip below then attributes
    # every later reward to the wrong task.
    scored = [(s, (s.get("reward_info", {}) or {}).get("reward", s.get("reward"))) for s in sims]
    scored = [(s, r) for s, r in scored if isinstance(r, (int, float))]
    rewards = [r for _, r in scored]
    per_task: dict[str, list[float]] = {}
    for s, r in scored:
        tid = str(s.get("task_id", "?"))
        per_task.setdefault(tid, []).append(r)
    return {
        "reward": (sum(rewards) / len(rewards)) if rewards else None,
        "sims": len(rewards),
        "per_task": {
            t: f"{sum(1 for r in rs if r >= 1.0)}/{len(rs)}"
            for t, rs in sorted(per_task.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)
        },
        **stamped,
    }


def pooled_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """A prior summary entry, or a failure standing in for it.

    A resumed set merges the entries already in `summary.json` with
    the ones it just produced. Those entries are read from a file in
    a directory the agent can write, so an entry that carries a REWARD
    and no provenance stamp is not pooled: nothing says which policy
    bytes or which labeled scenarios its number came out of, or that
    this harness produced it at all.

    It is replaced by a failure rather than dropped. Dropping it would
    shrink the set quietly — the mean would be over fewer runs and the
    exit code would still say the set completed — while a failure
    names the run and makes the invocation exit nonzero.

    An entry with neither a reward nor a stamp is carried through
    unchanged. That is what the two-phase flow writes: `--stages
    translate` records one entry per run with no `grade` key and no
    `provenance` key at all, and the later `--stages grade --only N`
    pools those back while it grades run N. There is no number in them
    to attribute, so demanding provenance would fail every run the
    second phase did not touch. An entry that recorded its own error
    keeps it, and still counts as a failed run.

    A stamp is written only when this harness grades a run, so an
    entry that carries a stamp but no reward is a graded run whose
    number has gone missing from the file, not a run the second phase
    has not reached yet. It is replaced by a failure for the same
    reason an unstamped number is: otherwise a grade could leave the
    pooled mean by deleting one key, with the set still exiting zero.

    The stamp is provenance, not proof. Whoever can write the summary
    can write a stamp into it, or strip the stamp and the grade
    together; what the check buys is that a number this harness never
    produced does not silently join the mean, and that a number it did
    produce does not silently leave it while its stamp remains.
    """
    grade = entry.get("grade")
    reward = grade.get("reward") if isinstance(grade, dict) else None
    prov = entry.get("provenance")
    stamped = (
        isinstance(prov, dict)
        and prov.get("harness") == HARNESS_STAMP
        and isinstance(prov.get("policy_sha256"), str)
        and bool(prov["policy_sha256"])
        and isinstance(prov.get("evidence_digest"), str)
        and bool(prov["evidence_digest"])
    )
    if reward is None and not stamped:
        return entry
    if reward is not None and stamped:
        return entry
    recorded = entry.get("error")
    if stamped:
        why = (
            f"prior summary entry carries a {HARNESS_STAMP} provenance "
            f"stamp but no reward, so a graded number has gone missing "
            f"from the summary"
        )
    else:
        why = (
            f"prior summary entry carries no {HARNESS_STAMP} "
            f"provenance stamp, so nothing says which policy or which "
            f"labeled scenarios produced its number"
        )
    return {
        "run": int(entry["run"]),
        "error": (why + (f" (it recorded: {recorded})" if recorded else "") + "; re-grade this variant"),
        "grade": {"reward": None, "sims": 0, "per_task": {}},
    }


SLICE_PROBE = """
import sasy
from sasy.observability.api import record_events_with_dependencies, backward_slice
from sasy.proto.observability_pb2 import Event
with sasy.session("preflight-slice-probe"):
    ids = record_events_with_dependencies(
        [Event(text="a", agent="Preflight", role=1),
         Event(text="b", agent="Preflight", role=2)], [])
    backward_slice(ids[-1])
print("slice read OK")
"""


def preflight(args: argparse.Namespace) -> int:
    """Check what a full run needs, before it burns any of it.

    Every check here corresponds to a failure that otherwise shows up
    late and quietly: no server means the capture stage produces
    nothing, a missing observability-reader role means it produces
    records with no graph, and a missing oracle key means the labels
    never arrive — each after the previous stages have already run.
    """
    import shutil

    problems: list[str] = []

    def ok(label: str, good: bool, detail: str = "") -> None:
        print(f"  [{'ok' if good else 'FAIL'}] {label}{' — ' + detail if detail else ''}")
        if not good:
            problems.append(label)

    for tool in ("claude", "uv"):
        ok(f"{tool} on PATH", shutil.which(tool) is not None)

    # Probe the shipped path, not a port guess: a reachable server the
    # capture identity cannot read slices from looks healthy and
    # produces empty captures.
    probe = subprocess.run(
        [sys.executable, "-c", SLICE_PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "TLS_CERT_PATH": str(args.capture_cert), "TLS_KEY_PATH": str(args.capture_key)},
    )
    detail = (probe.stdout or probe.stderr or "").strip().splitlines()
    ok("server reachable + capture can read slices", probe.returncode == 0, detail[-1][:120] if detail else "")

    if args.oracle_provider == "litellm":
        creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        vertex = bool(creds and Path(creds).exists() and os.environ.get("VERTEXAI_PROJECT"))
        azure = bool(os.environ.get("AZURE_API_KEY"))
        ok(
            "litellm judge credentials",
            vertex or azure,
            "Vertex needs GOOGLE_APPLICATION_CREDENTIALS + VERTEXAI_PROJECT (+ VERTEXAI_LOCATION, default global)",
        )
        ok(
            "oracle model named",
            bool(args.oracle_model or os.environ.get("SASY_ORACLE_MODEL")),
            "pass --oracle-model, e.g. vertex_ai/claude-opus-4-5@20251101",
        )
        if vertex:
            token_probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import google.auth; from google.auth.transport.requests "
                    "import Request; c,_=google.auth.default(); c.refresh(Request())",
                ],
                capture_output=True,
                text=True,
            )
            ok(
                "Vertex credentials refresh",
                token_probe.returncode == 0,
                "run gcloud auth application-default login if reauthentication is required",
            )
    else:
        ok(
            "ANTHROPIC_API_KEY",
            bool(os.environ.get("ANTHROPIC_API_KEY")),
            "or use --oracle-provider litellm with Vertex/Azure",
        )

    ok("spec file", args.spec.exists(), str(args.spec))
    if args.label_schema == "evidence":
        ok(
            "predicate vocabulary",
            bool(args.predicate_vocabulary and args.predicate_vocabulary.exists()),
            str(args.predicate_vocabulary),
        )
    ok("allow_all policy", (REPO_ROOT / "experiments/translation/allow_all.dl").exists())

    overlap = set(args.grade_tasks) & set(capture_pool(args.grade_tasks, args.capture_tasks, seed=1))
    ok("capture/grade tasks disjoint", not overlap, str(sorted(overlap)))

    print(f"\n  {'ready' if not problems else str(len(problems)) + ' blocker(s)'}")
    return 1 if problems else 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the translator N times and grade each variant",
    )
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument(
        "--only",
        default="",
        help="comma-separated run indices to execute (default: all). "
        "Run i always draws the same capture tasks and seed, so "
        "resuming a subset reproduces exactly what a full set "
        "would have done for those runs.",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="variants to run at once (default 1). Safe above 1: "
        "policies pin per session, and captures, oracle caches "
        "and finalize markers are all per-run.",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--grade-tasks",
        nargs="+",
        default=DEFAULT_GRADE_TASKS,
        help="tau2 airline tasks to grade on (default: the three run_benchmark.py uses)",
    )
    ap.add_argument(
        "--capture-tasks",
        type=int,
        default=10,
        help="how many held-out tasks each run captures from",
    )
    ap.add_argument(
        "--agent-llm",
        default="vertex_ai/claude-opus-4-5@20251101",
        help="tau2 agent model used for both capture and held-out grading",
    )
    ap.add_argument(
        "--user-llm",
        default="vertex_ai/claude-opus-4-5@20251101",
        help="tau2 user-simulator model used for both capture and grading",
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=0,
        help="cap on labeled decision points per run (0 = full traces)",
    )
    ap.add_argument(
        "--trajectory-mode",
        choices=TRAJECTORY_MODES,
        default="independent",
        help=(
            "independent gives every branch its own capture/sample/labels; "
            "shared gives every branch the source arm's same labeled rows; "
            "one-arm gives labeled rows only to the source arm and authors "
            "the other branches from the shared system profile"
        ),
    )
    ap.add_argument(
        "--trajectory-source-arm",
        type=int,
        default=1,
        help="source arm for shared/one-arm trajectory modes (default 1)",
    )
    ap.add_argument(
        "--repair-rounds",
        type=int,
        default=1,
        help=(
            "bounded parent-evaluated policy repair attempts after a failed "
            "preflight/finalize gate (default 1; 0 preserves fail-fast behavior)"
        ),
    )
    ap.add_argument("--max-per-tool", type=int, default=0)
    ap.add_argument(
        "--priority-report",
        type=Path,
        default=None,
        help="prior convergence report for disagreement-directed sampling",
    )
    ap.add_argument(
        "--fuzz-per-scenario",
        type=int,
        default=0,
        help="add up to N concrete substrate mutations per sampled trace",
    )
    ap.add_argument(
        "--fuzz-families",
        nargs="+",
        default=None,
        choices=[
            "argument-change",
            "argument-drop",
            "graph-edge-drop",
            "role-drop",
            "metadata-drop",
            "attestation-flip",
            "binding-mismatch",
        ],
    )
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument(
        "--spec",
        type=Path,
        default=REPO_ROOT / "experiments/translation/spec.md",
    )
    ap.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"comma-separated subset of {','.join(STAGES)}",
    )
    ap.add_argument(
        "--capture-cert",
        type=Path,
        default=REPO_ROOT / "certs/admin.crt",
        help="mTLS cert for capture; must map to an entity holding observability-reader (see config/auth_config.yaml)",
    )
    ap.add_argument(
        "--capture-key",
        type=Path,
        default=REPO_ROOT / "certs/admin.key",
    )
    ap.add_argument(
        "--oracle-provider",
        choices=["anthropic", "litellm"],
        default="anthropic",
        help="litellm routes the judge through Vertex/Azure — needed "
        "when there is no ANTHROPIC_API_KEY, and preferable "
        "anyway when it lets the judge come from a different "
        "model family than the translator",
    )
    ap.add_argument(
        "--oracle-model",
        default=None,
        help="e.g. vertex_ai/claude-opus-4-5@20251101 or vertex_ai/gemini-3-pro-preview",
    )
    ap.add_argument(
        "--label-schema",
        choices=["decision", "evidence"],
        default="decision",
        help=("decision judges ALLOW/ASK/DENY; evidence additionally labels a declared predicate/clause ABI"),
    )
    ap.add_argument(
        "--predicate-vocabulary",
        type=Path,
        default=None,
        help="JSON ABI required when --label-schema evidence",
    )
    ap.add_argument(
        "--ground-truth-labels",
        type=Path,
        default=None,
        help=(
            "reviewed verdicts to import after oracle labeling and before "
            "the pre-authoring snapshot; entries are matched by scenario "
            "name across sampled arms and every entry must match at least one arm"
        ),
    )
    ap.add_argument(
        "--preflight",
        action="store_true",
        help="check prerequisites and exit without running anything",
    )
    args = ap.parse_args()

    # Claude Code's Vertex setup and LiteLLM name the same ADC configuration
    # differently.  Reuse the authenticated Claude variables when the LiteLLM
    # spellings are absent; this changes no credential bytes and keeps the
    # provider/model explicit on the campaign command line.
    if not os.environ.get("VERTEXAI_PROJECT") and os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID"):
        os.environ["VERTEXAI_PROJECT"] = os.environ["ANTHROPIC_VERTEX_PROJECT_ID"]
    if not os.environ.get("VERTEXAI_LOCATION") and os.environ.get("CLOUD_ML_REGION"):
        os.environ["VERTEXAI_LOCATION"] = os.environ["CLOUD_ML_REGION"]

    if args.label_schema == "evidence" and args.predicate_vocabulary is None:
        ap.error("--label-schema evidence requires --predicate-vocabulary")
    if args.ground_truth_labels is not None and not args.ground_truth_labels.exists():
        ap.error(f"--ground-truth-labels does not exist: {args.ground_truth_labels}")
    if not 1 <= args.trajectory_source_arm <= args.runs:
        ap.error("--trajectory-source-arm must be between 1 and --runs")
    if args.repair_rounds < 0:
        ap.error("--repair-rounds must be non-negative")
    if args.preflight:
        sys.exit(preflight(args))

    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    unknown = stages - set(STAGES)
    if unknown:
        print(f"ERROR: unknown stage(s): {sorted(unknown)}", file=sys.stderr)
        sys.exit(2)

    args.out.mkdir(parents=True, exist_ok=True)

    branch_evidence = args.out / ".branch-set-evidence"
    branch_evidence.mkdir(parents=True, exist_ok=True)

    def evidence_scenarios(arm: int) -> Path:
        return branch_evidence / f"source-arm{arm}-scenarios.json"

    def evidence_profile(arm: int) -> Path:
        return branch_evidence / f"source-arm{arm}-system.md"

    def evidence_profile_json(arm: int) -> Path:
        return branch_evidence / f"source-arm{arm}-system.json"

    def do_run(i: int) -> dict[str, Any]:
        run_dir = args.out / f"run{i}"
        run_dir.mkdir(parents=True, exist_ok=True)
        access = trajectory_access(args.trajectory_mode, i, args.trajectory_source_arm)
        owns_sample = access["source_arm"] == i
        source_dir = args.out / f"run{access['source_arm']}"
        if owns_sample:
            tasks = capture_pool(args.grade_tasks, args.capture_tasks, seed=i)
        else:
            source_config = source_dir / "config.json"
            if not source_config.exists():
                raise RuntimeError(
                    f"trajectory source arm {access['source_arm']} has not run; "
                    "run the source arm first or execute the full branch set"
                )
            source_data = json.loads(source_config.read_text())
            tasks = [str(value) for value in source_data.get("capture_tasks", [])]
        overlap = set(tasks) & set(args.grade_tasks)
        if overlap:
            print(f"ERROR: capture/grade overlap {sorted(overlap)}", file=sys.stderr)
            sys.exit(1)
        print(f"\n=== run {i}/{args.runs} — capture tasks {tasks} ===")
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "run": i,
                    "seed": i,
                    "capture_tasks": tasks,
                    "grade_tasks": args.grade_tasks,
                    "trials": args.trials,
                    "agent_llm": args.agent_llm,
                    "user_llm": args.user_llm,
                    "label_schema": args.label_schema,
                    "priority_report": (str(args.priority_report) if args.priority_report else None),
                    "fuzz_per_scenario": args.fuzz_per_scenario,
                    "fuzz_families": args.fuzz_families,
                    "predicate_vocabulary": (str(args.predicate_vocabulary) if args.predicate_vocabulary else None),
                    "ground_truth_labels": (
                        str(args.ground_truth_labels.resolve()) if args.ground_truth_labels else None
                    ),
                    "ground_truth_labels_sha256": (
                        hashlib.sha256(args.ground_truth_labels.read_bytes()).hexdigest()
                        if args.ground_truth_labels
                        else None
                    ),
                    "trajectory_mode": args.trajectory_mode,
                    "repair_rounds": args.repair_rounds,
                    "trajectory_access": access,
                },
                indent=2,
            )
            + "\n"
        )

        cap = run_dir / "capture.jsonl"
        scen = run_dir / "scenarios.json"
        policy = run_dir / "airline_policy.dl"
        profile = run_dir / "system.md"
        harness = harness_dir_for(run_dir)
        # The evidence this invocation holds itself, in memory,
        # from the moment labeling finishes. `None` until then.
        snapshot: dict[str, Any] | None = None

        if owns_sample:
            if "capture" in stages:
                cap = stage_capture(
                    run_dir,
                    tasks,
                    args.capture_cert,
                    args.capture_key,
                    agent_llm=args.agent_llm,
                    user_llm=args.user_llm,
                )
            if "sample" in stages:
                scen = stage_sample(
                    run_dir,
                    cap,
                    i,
                    args.sample,
                    args.max_per_tool,
                    priority_report=args.priority_report,
                    fuzz_per_scenario=args.fuzz_per_scenario,
                    fuzz_families=args.fuzz_families,
                )
                profile.unlink(missing_ok=True)
                rc = run(
                    [
                        "uv",
                        "run",
                        "python",
                        "experiments/translation/extract_system.py",
                        "--capture",
                        str(cap),
                        "--report",
                        str(profile),
                        "--out",
                        str(run_dir / "system.json"),
                    ]
                )
                # The profile is the translator's account of the deployment.
                # Failing quietly here leaves either no profile or a stale
                # one from an earlier resume, and the policy is then written
                # against structural facts from the wrong capture.
                if rc != 0 or not profile.exists():
                    raise RuntimeError(
                        f"system extraction exited {rc}; refusing to translate against a missing or stale profile"
                    )
            if "label" in stages:
                snapshot = stage_label(
                    run_dir,
                    scen,
                    args.spec,
                    args.oracle_provider,
                    args.oracle_model,
                    harness,
                    fresh="sample" in stages,
                    label_schema=args.label_schema,
                    predicate_vocabulary=(args.predicate_vocabulary),
                    ground_truth_labels=args.ground_truth_labels,
                )
            elif "translate" in stages or "grade" in stages:
                # Resuming. The snapshot from the labeling invocation is
                # the only one that can be used; verify_run refuses to
                # proceed on a labeled file without it rather than
                # re-taking one from whatever is on disk now.
                snapshot = load_snapshot(harness)

            # Freeze the pre-authoring evidence.  Dependent branches consume
            # this copy, not a scenario file to which the source author may
            # later append top-up tests.
            if snapshot is not None and scen.exists() and profile.exists():
                frozen_scenarios = evidence_scenarios(i)
                if frozen_scenarios.exists():
                    frozen_record = scenario_snapshot_of_file(frozen_scenarios)
                    if (
                        frozen_record["identities"] != snapshot["identities"]
                        or frozen_record["labeled"] != snapshot["labeled"]
                    ):
                        raise RuntimeError(
                            f"pre-authoring evidence for arm {i} no longer "
                            "matches its label-stage snapshot; refusing to "
                            "overwrite it during resume"
                        )
                else:
                    shutil.copy2(scen, frozen_scenarios)
                frozen_profile = evidence_profile(i)
                if not frozen_profile.exists():
                    shutil.copy2(profile, frozen_profile)
                system_json = run_dir / "system.json"
                if system_json.exists() and not evidence_profile_json(i).exists():
                    shutil.copy2(system_json, evidence_profile_json(i))
        else:
            source_scenarios = evidence_scenarios(access["source_arm"])
            source_profile = evidence_profile(access["source_arm"])
            if not source_scenarios.exists() or not source_profile.exists():
                raise RuntimeError(
                    f"source arm {access['source_arm']} has no frozen pre-authoring evidence; "
                    "run it from sample,label before dependent branches"
                )
            shutil.copy2(source_profile, profile)
            source_profile_json = evidence_profile_json(access["source_arm"])
            if source_profile_json.exists():
                shutil.copy2(source_profile_json, run_dir / "system.json")
            if access["visible_to_author"]:
                shutil.copy2(source_scenarios, scen)
                snapshot = write_snapshot(harness, scen, fresh=True)
            else:
                write_unlabeled_scenarios(source_scenarios, scen)
                # Freeze the hidden concrete substrate too.  Its label census
                # is zero, so no normative answer is exposed to the author,
                # but repeated parent gates (including bounded repair) can now
                # prove that the actions/graphs being evaluated did not move.
                snapshot = write_snapshot(harness, scen, fresh=True)
        verified: Path | None = None
        if "translate" in stages:
            verified = stage_translate(
                run_dir,
                scen,
                args.spec,
                profile,
                snapshot,
                harness,
                trajectory_visible=bool(access["visible_to_author"]),
                repair_rounds=args.repair_rounds,
            )
        entry: dict[str, Any] = {
            "run": i,
            "capture_tasks": tasks,
            "trajectory_access": access,
        }
        if "grade" in stages:
            if not policy.exists():
                # Requested but impossible. Recording a null grade with no
                # error made the run look merely unmeasured, so the final
                # failure check passed and a set that graded nothing exited 0.
                raise RuntimeError(
                    f"grade requested but no policy at {policy}; run the "
                    f"translate stage first or point --out at the right set"
                )
            grade = stage_grade(
                run_dir,
                policy,
                args.grade_tasks,
                args.trials,
                scenarios=scen,
                snapshot=snapshot,
                harness=harness,
                verified=verified,
                policy_nl=args.spec,
                agent_llm=args.agent_llm,
                user_llm=args.user_llm,
                cert=args.capture_cert,
                key=args.capture_key,
            )
            # The stamp says which policy bytes and which labeled
            # scenarios this number came out of, and that this harness
            # produced it. A later invocation pooling prior results
            # requires it — see `pooled_entry`.
            entry["provenance"] = grade.pop("provenance", None)
            entry["grade"] = grade
        return entry

    indices = [int(x) for x in args.only.split(",") if x.strip()] if args.only else list(range(1, args.runs + 1))
    if args.trajectory_mode != "independent" and args.trajectory_source_arm in indices:
        indices = [args.trajectory_source_arm] + [i for i in indices if i != args.trajectory_source_arm]
    results: list[dict[str, Any]] = []
    # A shared/one-arm source is a real dependency, not just a preferred order:
    # it must finish labeling and freeze its evidence before any dependent arm
    # can be authored.  Once frozen, the remaining arms are independent and may
    # run concurrently.
    if args.trajectory_mode != "independent" and args.trajectory_source_arm in indices:
        source = args.trajectory_source_arm
        print(f"Running trajectory source arm {source} before dependent arms")
        try:
            results.append(do_run(source))
        except Exception as exc:  # noqa: BLE001 — retain the failed treatment
            print(f"  run {source} FAILED: {exc}")
            results.append(
                {
                    "run": source,
                    "error": str(exc),
                    "grade": {"reward": None, "sims": 0, "per_task": {}},
                }
            )
        indices = [i for i in indices if i != source]
    if args.concurrency > 1:
        # Runs share the server but not their state: policies pin per
        # session, captures and oracle caches are per-run, and the
        # finalize markers are per-run via SASY_POLICY_RUN_DIR.
        print(f"Running {len(indices)} variants, {args.concurrency} at a time")
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(do_run, i): i for i in indices}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:  # noqa: BLE001 — one run must not sink the set
                    print(f"  run {i} FAILED: {exc}")
                    results.append({"run": i, "error": str(exc), "grade": {"reward": None, "sims": 0, "per_task": {}}})
        results.sort(key=lambda r: r["run"])
    else:
        results.extend(do_run(i) for i in indices)

    summary_path = args.out / "summary.json"
    # Merge with any prior summary rather than replacing it. `--only 3`
    # after a five-run set would otherwise rewrite the aggregate as a
    # one-run measurement, discarding four results whose artifacts are
    # still on disk — a partial resume must not silently redefine what
    # was measured.
    merged: dict[int, dict[str, Any]] = {}
    if summary_path.exists():
        try:
            prior = json.loads(summary_path.read_text())
            # Only merge results measured the same way. Grades from a
            # different task set or trial count are a different
            # population, and averaging them under the current
            # arguments would report a mean no run actually produced.
            prior_config = (
                prior.get("grade_tasks"),
                prior.get("trials"),
                prior.get("trajectory_mode", "independent"),
                prior.get("trajectory_source_arm", 1),
                prior.get("agent_llm", "vertex_ai/claude-opus-4-5@20251101"),
                prior.get("user_llm", "vertex_ai/claude-opus-4-5@20251101"),
                prior.get("ground_truth_labels_sha256"),
            )
            this_config = (
                args.grade_tasks,
                args.trials,
                args.trajectory_mode,
                args.trajectory_source_arm,
                args.agent_llm,
                args.user_llm,
                (
                    hashlib.sha256(args.ground_truth_labels.read_bytes()).hexdigest()
                    if args.ground_truth_labels
                    else None
                ),
            )
            if prior_config != this_config:
                print(
                    f"    WARNING: prior summary measured "
                    f"{prior_config[0]} at {prior_config[1]} trials, this run "
                    f"{this_config[0]} at {this_config[1]}; discarding the "
                    f"prior results rather than pooling them"
                )
            else:
                for entry in prior.get("per_run", []):
                    if isinstance(entry, dict) and "run" in entry:
                        merged[int(entry["run"])] = pooled_entry(entry)
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            print(f"    WARNING: could not read prior {summary_path}")
    for entry in results:
        merged[int(entry["run"])] = entry
    results = [merged[k] for k in sorted(merged)]

    failures = [r["run"] for r in results if r.get("error")]
    rewards = [r["grade"]["reward"] for r in results if r.get("grade", {}).get("reward") is not None]
    ground_truth_digest = (
        hashlib.sha256(args.ground_truth_labels.read_bytes()).hexdigest() if args.ground_truth_labels else None
    )
    ground_truth_matched: set[str] = set()
    if ground_truth_digest is not None:
        for run_dir in sorted(args.out.glob("run*")):
            report_path = run_dir / "ground-truth-import.json"
            if not report_path.exists():
                continue
            try:
                report = json.loads(report_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if report.get("labels_sha256") == ground_truth_digest:
                ground_truth_matched.update(str(name) for name in report.get("matched_names", []))
        supplied = {entry["name"] for entry in parse_entries(json.loads(args.ground_truth_labels.read_text()))}
        ground_truth_unmatched = sorted(supplied - ground_truth_matched)
    else:
        ground_truth_unmatched = []
    summary = {
        "runs": len(results),
        "failed_runs": failures,
        "grade_tasks": args.grade_tasks,
        "trials": args.trials,
        "trajectory_mode": args.trajectory_mode,
        "trajectory_source_arm": args.trajectory_source_arm,
        "repair_rounds": args.repair_rounds,
        "agent_llm": args.agent_llm,
        "user_llm": args.user_llm,
        "ground_truth_labels": (str(args.ground_truth_labels.resolve()) if args.ground_truth_labels else None),
        "ground_truth_labels_sha256": ground_truth_digest,
        "ground_truth_matched_names": sorted(ground_truth_matched),
        "ground_truth_unmatched_names": ground_truth_unmatched,
        "sampling_claim": ("prompt-and-file-layout isolation only; all translators run as the same OS user"),
        "rewards": rewards,
        "mean": statistics.fmean(rewards) if rewards else None,
        "stdev": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "per_run": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n=== variant set: {len(rewards)}/{args.runs} graded ===")
    for r in results:
        g = r.get("grade") or {}
        rw = g.get("reward")
        print(
            f"  run {r['run']}: reward="
            f"{'n/a' if rw is None else f'{rw:.3f}'} "
            f"({g.get('sims', 0)} sims)  {g.get('per_task', {})}"
        )
    if rewards:
        print(f"\n  mean {summary['mean']:.3f} ± {summary['stdev']:.3f} (n={len(rewards)})")
    print(f"  → {summary_path}")
    # A mean over the surviving runs is not the experiment that was
    # asked for. Report it, write the diagnostics, then exit nonzero so
    # automation cannot publish a partial set as a complete one.
    if failures:
        print(f"\n  {len(failures)} run(s) failed: {failures}", file=sys.stderr)
        sys.exit(1)
    if ground_truth_unmatched:
        print(
            f"\n  reviewed ground-truth entries matched no sampled arm: {ground_truth_unmatched}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
