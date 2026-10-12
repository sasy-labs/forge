"""Run the live AgentDojo benchmark in one process, one thread per run.

The work is network-bound — the policy engine and its evaluators sit nearly
idle while the model provider answers — so threads are the right shape: one
AgentDojo import, one engine connection, and as many concurrent conversations
as the provider will take. (A process per shard pays ~100 MB per process just
to import AgentDojo, which is what limits concurrency, not the API.) For more
than one machine, `--shard K/N` splits the job list deterministically, and
every process writes into the same result tree.

Three things are not thread-safe as they stand, and each is handled here.

1. AgentDojo's logger stack is a `ContextVar` whose default is a *mutable list*.
   `LOGGER_STACK.get()` therefore returns the SAME list in every thread, and
   `Logger.get()` — which decides the file a run is written to — returns
   whichever logger some other thread entered last. Results would be written
   into the wrong task's file, silently. Each worker calls
   `LOGGER_STACK.set([])` first, which gives that thread its own stack.

2. `get_suite` returns a module-level singleton, and the per-task session
   wrapper is installed by assigning to `suite.run_task_with_pipeline`.
   Patching a shared object from many threads is a race, so each worker patches
   its own `copy.copy` of the suite. The copy shares the task registries, which
   are read-only once the suite is built.

3. The graph recorder is per-conversation state. Each worker builds its own
   pipeline and its own recorder.

The SASY SDK needs nothing special: its context variables all have immutable
defaults, so a worker thread starts clean and `sasy.session()` inside the worker
isolates that task.

Results are AgentDojo's own per-run JSON files under the log directory, keyed
by pipeline name. A run whose result file already exists is skipped unless
`force_rerun` is set, so an interrupted sweep resumes where it stopped.
"""

from __future__ import annotations

import copy
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import agentdojo.logging as adj_logging
import sasy
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks
from agentdojo.logging import OutputLogger
from agentdojo.models import MODEL_NAMES
from agentdojo.task_suite.load_suites import get_suite
from sasy.policy.api import set_session_policy, update_policy_metadata

from . import functor_source
from .baselines import build_baseline_pipeline, build_defended_pipeline
from .connect import REPO_ROOT, connect_local
from .pipeline import build_pipeline
from .policy_metadata import build as build_metadata
from .providers import build_llm

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
ANALYSIS = EXAMPLE_DIR / "analysis"
POLICY = EXAMPLE_DIR / "policies" / "agentdojo_policy.dl"
FUNCTORS = EXAMPLE_DIR / "policies" / "agentdojo_functors.cpp"
DEFAULT_ENV = REPO_ROOT / ".env-credentials"
DEFAULT_LOGDIR = REPO_ROOT / "outputs" / "agentdojo" / "runs"
SUITES = ("banking", "slack", "travel", "workspace")

# AgentDojo's attack templates resolve a model family from the pipeline name.
# Its registry predates this Azure deployment version. Register the exact id
# before any workers start; preserve the label and any upstream mapping.
MODEL_NAMES.setdefault("gpt-4o-2024-08-06", "GPT-4")



# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmConfig:
    """What one table cell runs: the arm, its defense or policy switches, and
    the tag its results are filed under. Hashable, because it keys each
    thread's pipeline cache."""

    arm: str  # "sasy" or "baseline"
    tag: str | None = None
    defense: str | None = None
    switches: frozenset[str] = field(default_factory=frozenset)
    repeat_user_prompt: bool = False
    check_response: bool = False
    vouch_channels: bool = False

    def pipeline_name(self, label: str) -> str:
        """`<model label>[-<tag>]-<suffix>`, the directory AgentDojo writes to.

        `analysis/make_table.py` finds a sweep by `-<tag>-r<n>` in this name,
        so the shape must not change.
        """
        base = label + (f"-{self.tag}" if self.tag else "")
        if self.arm == "sasy":
            return f"{base}-sasy"
        return f"{base}-{self.defense}" if self.defense else f"{base}-baseline"


@dataclass
class RunSettings:
    """Settings shared by every cell of one invocation."""

    provider: str = "azure"
    deployment: str = "gpt-4o"
    model_label: str | None = None
    env_file: Path = DEFAULT_ENV
    attack: str | None = "tool_knowledge"
    version: str = "v1.2.2"
    logdir: Path = DEFAULT_LOGDIR
    threads: int = 96
    force_rerun: bool = False
    fail_open: bool = False
    decisions: Path | None = None

    @property
    def label(self) -> str:
        return self.model_label or self.deployment


# --------------------------------------------------------------------------
# Policy upload and operator data
# --------------------------------------------------------------------------


def _analysis_module(name: str):
    if str(ANALYSIS) not in sys.path:
        sys.path.insert(0, str(ANALYSIS))
    return __import__(name)


def load_directory(suite: str, version: str, vouch_channels: bool = False) -> frozenset[str]:
    """The operator's own list of destinations, which the monitor resolves
    against at check time. Built from the suite environment, which is what a
    real deployment's directory API would return."""
    directory = _analysis_module("directory")
    return frozenset(directory.build(version, vouch_channels).get(suite, set()))


def load_principals(suite: str, version: str) -> frozenset[str]:
    """The people the operator's own workspace lists, for this suite.

    Same source as the directory, so there is one roster rather than a second
    copy written out by hand. Empty for suites whose senders are addresses —
    there `own_domains` answers the question, and answers it better.
    """
    directory = _analysis_module("directory")
    return frozenset(directory.principals(version).get(suite, set()))


def metadata_for(switches: frozenset[str], suite: str, version: str) -> list[tuple[str, str, str]]:
    """The policy metadata for one suite under the selected switches."""
    on = {s: True for s in switches if s != "no_target_exemption"}
    return build_metadata(
        suite,
        **on,
        target_exemption="no_target_exemption" not in switches,
        own_principals=load_principals(suite, version),
    )


def upload_session_policy(
    session_id: str,
    facts: list[tuple[str, str, str]],
    policy_path: Path = POLICY,
    functor_path: Path = FUNCTORS,
) -> None:
    """Bind one session to the AgentDojo policy, then give it its taxonomy.

    The taxonomy travels as *session-scoped* `PolicyMetadata` facts rather than
    on the `SetPolicy` call. All four suites share one policy source and differ
    only in configuration, and `SetPolicyRequest.policy_metadata` is persisted
    keyed by the policy's *content hash* — so concurrent binds of an identical
    source would hand sessions each other's configuration. A session holding a
    sibling suite's taxonomy recognises none of its own tools as sinks, matches
    no rule and denies nothing: a silent fail-open that appears only under
    concurrency. `tests/test_policy_metadata_race.py` (repository root)
    reproduces it; `UpdatePolicyMetadata` is keyed by session id, so this route
    cannot collide.
    """
    response = set_session_policy(
        session_id,
        policy_path.read_text(),
        functor_source=functor_source.load(functor_path),
        timeout=900,
    )
    if not response.accepted:
        raise SystemExit(
            f"policy upload failed: {response.message}\n{response.error_output}"
        )
    update_policy_metadata(session_id, facts)


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------

_local = threading.local()
_print_lock = threading.Lock()

#: Runs lost to a malformed tool-call argument string from the model. Counted
#: rather than hidden — see `scored_run` for why they are scored as they are.
_malformed: list[str] = []
_malformed_lock = threading.Lock()


def scored_run(original_run, label: str):
    """AgentDojo's own failure convention, extended to the one case it misses.

    `benchmark.py` catches context-length, `ApiError` and `ServerError` and
    records `utility=False, security=True` — an infrastructure failure counts as
    BOTH a lost user task and a successful injection. That is deliberately
    pessimistic for a defended arm, and it is what published results inherit
    unchanged, so it is what is used here.

    A `json.JSONDecodeError` from a malformed tool-call argument string is NOT
    caught upstream: it escapes `run_task_with_pipeline` and kills the process.
    Scoring it like every other infrastructure failure is the same convention
    applied consistently; the count is reported separately, and it can only
    ever count against the arm it happens in.
    """

    def run(*a, **kw):
        try:
            return original_run(*a, **kw)
        except json.JSONDecodeError as exc:
            with _malformed_lock:
                _malformed.append(f"{label}: {exc}")
            # Record it in the result file too, so the scored failure is
            # auditable from the results alone and cannot be mistaken for a
            # genuine attack success.
            adj_logging.Logger.get().log_error(
                f"malformed tool-call arguments, scored as an infrastructure "
                f"failure (utility=False, security=True): {exc}"
            )
            return False, True

    return run


@dataclass(frozen=True)
class Job:
    """One user task, and for an attacked run ONE injection task.

    The finer unit matters for wall clock. `run_task_with_injection_tasks` runs
    every injection for a user task in sequence, so a workspace attacked task is
    fourteen conversations deep. Splitting the pair out gives ~1046 jobs of
    roughly equal length per sweep and the tail disappears.
    """

    suite: str
    config: ArmConfig
    attack: str | None
    user_task: str
    injection_task: str | None = None

    def __str__(self) -> str:
        tail = f"/{self.injection_task}" if self.injection_task else ""
        tag = self.config.tag or self.config.arm
        return f"{self.suite}/{tag}/{self.attack or 'none'}/{self.user_task}{tail}"


class _Decisions:
    """Optional JSONL log of every monitor decision, shared across threads."""

    def __init__(self, path: Path | None) -> None:
        self._fh = path.open("a") if path else None
        self._lock = threading.Lock()

    def callback(self, suite: str):
        if self._fh is None:
            return None

        def record(fn: str, call_args: dict[str, Any], decision) -> None:
            line = json.dumps(
                {
                    "suite": suite,
                    "fn": fn,
                    "args": call_args,
                    "authorized": decision.authorized,
                    "reasons": decision.reasons,
                },
                default=str,
            )
            with self._lock:
                self._fh.write(line + "\n")
                self._fh.flush()

        return record

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


def _worker_state(settings: RunSettings, decisions: _Decisions, job: Job):
    """Per-thread pipeline, recorder and suite copy, built once per thread.

    Keyed by (suite, config): a pipeline is bound to a suite's tool set and a
    config's element list, and the config carries the tag — a pipeline cached
    from an earlier sweep carries that sweep's name and would write this sweep's
    results into the earlier sweep's directory.
    """
    if not getattr(_local, "ready", False):
        # (1) in the module docstring: without this every thread shares one
        # logger stack.
        adj_logging.LOGGER_STACK.set([])
        _local.ready = True
        _local.states = {}

    key = (job.suite, job.config)
    if key in _local.states:
        return _local.states[key]

    cfg = job.config
    llm = build_llm(settings.provider, settings.deployment, settings.env_file)
    name = cfg.pipeline_name(settings.label)

    # (2): patch a private copy, never the shared singleton.
    suite = copy.copy(get_suite(settings.version, job.suite))

    if cfg.arm == "baseline":
        if cfg.defense:
            pipeline = build_defended_pipeline(llm, name, cfg.defense)
        else:
            pipeline = build_baseline_pipeline(llm, name)
        # The baseline is scored by the same failure rule, so neither arm is
        # flattered by a lost run.
        suite.run_task_with_pipeline = scored_run(  # type: ignore[method-assign]
            suite.run_task_with_pipeline, f"{job.suite}/{cfg.tag or 'baseline'}"
        )
    else:
        directory = load_directory(job.suite, settings.version, cfg.vouch_channels)
        facts = metadata_for(cfg.switches, job.suite, settings.version)
        pipeline, recorder = build_pipeline(
            job.suite,
            settings.deployment,
            directory,
            fail_closed=not settings.fail_open,
            on_decision=decisions.callback(job.suite),
            llm=llm,
            check_response=cfg.check_response,
            repeat_user_prompt=cfg.repeat_user_prompt,
        )
        pipeline.name = name
        original_run = suite.run_task_with_pipeline

        def run_task_in_own_session(*a, **kw):  # noqa: ANN202
            # One SASY session per task. A shared session's graph grows without
            # bound while the evaluator is fed asynchronously, and a check whose
            # input node has not been ingested yet sees an empty slice, which is
            # a silent fail-open. A session per task keeps the graph small; the
            # policy upload is deduplicated server-side by content hash, so the
            # extra binds are cheap.
            with sasy.session() as handle:
                upload_session_policy(handle.session_id, facts)
                recorder.reset()
                return original_run(*a, **kw)

        suite.run_task_with_pipeline = scored_run(  # type: ignore[method-assign]
            run_task_in_own_session, f"{job.suite}/{cfg.tag or 'sasy'}"
        )

    _local.states[key] = (suite, pipeline)
    return _local.states[key]


def _run_job(settings: RunSettings, decisions: _Decisions, job: Job) -> str:
    suite, pipeline = _worker_state(settings, decisions, job)
    with OutputLogger(str(settings.logdir)):
        if job.attack is None:
            task = suite.get_user_task_by_id(job.user_task)
            run_task_without_injection_tasks(
                suite, pipeline, task, settings.logdir, settings.force_rerun, settings.version
            )
        else:
            attack = load_attack(job.attack, suite, pipeline)
            run_task_with_injection_tasks(
                suite,
                pipeline,
                suite.get_user_task_by_id(job.user_task),
                attack,
                settings.logdir,
                settings.force_rerun,
                injection_tasks=[job.injection_task] if job.injection_task else None,
                benchmark_version=settings.version,
            )
    return str(job)


def build_jobs(
    configs: list[ArmConfig],
    settings: RunSettings,
    suites: list[str] | None = None,
    user_tasks: list[str] | None = None,
    injection_tasks: list[str] | None = None,
    shard: tuple[int, int] | None = None,
) -> list[Job]:
    """Every (config, suite, user task[, injection task]) job, in a fixed order.

    Clean runs are always included; attacked runs only when an attack is set.
    `shard=(k, n)` keeps every job whose position is `k` modulo `n`, so `n`
    processes given `0/n` .. `n-1/n` cover the list exactly once.
    """
    jobs: list[Job] = []
    for cfg in configs:
        for suite_name in suites or SUITES:
            s = get_suite(settings.version, suite_name)
            tasks = [t for t in s.user_tasks if not user_tasks or t in user_tasks]
            injections = [
                i for i in s.injection_tasks if not injection_tasks or i in injection_tasks
            ]
            jobs += [Job(suite_name, cfg, None, t) for t in tasks]
            if settings.attack:
                jobs += [
                    Job(suite_name, cfg, settings.attack, t, i)
                    for t in tasks
                    for i in injections
                ]
    if shard is not None:
        k, n = shard
        jobs = [j for idx, j in enumerate(jobs) if idx % n == k]
    return jobs


def run_sweep(settings: RunSettings, jobs: list[Job]) -> int:
    """Run `jobs` on `settings.threads` threads; return how many failed."""
    print(f"{len(jobs)} runs, {settings.threads} threads, one process", flush=True)
    decisions = _Decisions(settings.decisions)
    started = time.time()
    done = failed = 0
    malformed_before = len(_malformed)
    try:
        with ThreadPoolExecutor(max_workers=settings.threads) as pool:
            futures = {pool.submit(_run_job, settings, decisions, j): j for j in jobs}
            for future in as_completed(futures):
                job = futures[future]
                done += 1
                try:
                    future.result()
                    status = "ok  "
                except Exception as exc:  # noqa: BLE001 — one run must not stop the sweep
                    failed += 1
                    status = "FAIL"
                    with _print_lock:
                        print(f"  {job}: {type(exc).__name__}: {exc}", flush=True)
                elapsed = (time.time() - started) / 60
                eta = elapsed / done * (len(jobs) - done)
                with _print_lock:
                    print(
                        f"[{done}/{len(jobs)}] {status} {job}  "
                        f"elapsed {elapsed:.1f}m  eta {eta:.1f}m",
                        flush=True,
                    )
    finally:
        decisions.close()

    print(f"\n{len(jobs) - failed}/{len(jobs)} runs ok in "
          f"{(time.time() - started) / 60:.1f}m", flush=True)
    lost = _malformed[malformed_before:]
    if lost:
        # Reported, not buried: these were scored utility=False, security=True,
        # which counts against the arm on both axes.
        print(f"{len(lost)} run(s) lost to a malformed tool-call argument string "
              f"and scored as an infrastructure failure "
              f"(utility=False, security=True):", flush=True)
        for m in lost[:10]:
            print(f"  {m}", flush=True)
    return failed


def run(
    configs: list[ArmConfig],
    settings: RunSettings,
    *,
    repeats: int = 1,
    suites: list[str] | None = None,
    user_tasks: list[str] | None = None,
    injection_tasks: list[str] | None = None,
    shard: tuple[int, int] | None = None,
) -> int:
    """Run every config `repeats` times; return the number of failed runs.

    Each sweep is tagged `<tag>-r<i>`, a single sweep included (`<tag>-r1`), so
    the sweeps land in their own directories and `analysis/make_table.py` and
    `analysis/summarize_repeats.py`, which find a sweep by its `-r<n>`, read
    them as separate samples.
    """
    if any(c.arm == "sasy" for c in configs):
        connect_local()
        # HTTP instrumentation stays off: it would route the agent's own calls
        # to its model provider through the reference monitor, and the policy
        # governs the agent's tools, not how it talks to its model. Tool calls
        # reach the monitor explicitly, through the executor.
        sasy.instrument(http=False, langroid=False)

    settings.logdir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for sweep in range(1, repeats + 1):
        sweep_configs = [
            replace(c, tag=f"{c.tag}-r{sweep}" if c.tag else f"r{sweep}") for c in configs
        ]
        tags = ", ".join(str(c.tag) for c in sweep_configs)
        print(f"\n=== sweep {sweep}/{repeats}  tags={tags}", flush=True)
        jobs = build_jobs(sweep_configs, settings, suites, user_tasks, injection_tasks, shard)
        failed += run_sweep(settings, jobs)
    return failed
