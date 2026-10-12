"""Run the paper's benchmark matrix; every result goes under ``results/<timestamp>/``.

With no arguments (an output root is optional) this runs all four evaluation
targets the paper reports:

AgentDojo
    The eight cells of the results table: four defenses (none, AgentDojo's
    tool filter, the information-flow policy, the provenance policy), each with
    AgentDojo's ``repeat_user_prompt`` defense off and on.  Each cell is one
    ``experiments/agentdojo/run.py --preset <name>`` invocation; the presets fix
    the model (Azure ``gpt-4o``), the ``tool_knowledge`` attack, AgentDojo
    v1.2.2, all four suites and five sweeps.  Results are written where
    ``experiments/agentdojo/analysis/make_table.py --logdir`` reads them.

tau2
    Both domains in full (airline: 50 tasks, retail: 114 tasks, tau2's ``base``
    split), 4 trials per task, three agents: Claude Opus 4.5
    (``vertex_ai/claude-opus-4-5@20251101``), GPT-5.2 (``azure/gpt-5.2``) and
    Gemini 3.1 Pro (``vertex_ai/gemini-3.1-pro-preview``).  Two arms, paired:

    ``enforce-off``  ``tau2-instrumented`` with the natural-language policy in
                     the agent's prompt (``TAU2_SKIP_NL_POLICY=``), and
                     every trial pinned to ``experiments/translation/allow_all.dl``: the same
                     instrumented code path, but nothing is ever refused.
    ``enforce-on``   the same, pinned to the domain policy
                     ``experiments/tau2/{domain}_policy.dl``.

    The simulated user is GPT-5.2 for every agent (``--user-model``),
    so every agent faces the same user.  Retail's tasks are scored partly by
    natural-language assertions, which tau2 has an LLM judge; that judge is
    Claude Opus 4.5 (``TAU2_NL_JUDGE_MODEL``, ``--nl-judge-model``).

infoflow
    Toxic-flow and two multi-level-security configurations with enforcement,
    plus one natural-language-policy baseline; five trials at five workers.

malade
    Instrumented and natural-language-policy arms; five trials at five workers.

Other cells stay available behind flags and are not part of the default run:
``--mode sasy`` (tau2 with the policy text hidden), ``--task-subset curated`` (the three
hand-picked tasks per domain).

No policy is uploaded or swapped between cells.  Every cell pins its own
policy for each session it opens: tau2 trials through ``SASY_POLICY_PATH``,
AgentDojo per task, the information-flow and MALADE examples from their own
policy files.  Before the first cell that uses a policy, one trial of one task
runs on a single worker (``--no-warmup`` skips it): sessions that start at the
same moment on an uncached policy each compile their own copy, and dozens of
concurrent compiles can overload the machine.

Layout of one run::

    results/<timestamp>/
      manifest.json                       every cell: command, env, return code, results path
      server.log, server-data/            the run's own engine (--start-server, the default)
      server_latency.jsonl                append-only engine request/timing log
      <tau2 cell>/latency.jsonl            SDK timings (warm-ups recorded separately)
      <tau2 cell>/server_replay.jsonl      this cell's authorization request stream
      <tau2 cell>/latency-capture.json     attempt boundaries, durations, file hashes
      <tau2 cell>/replay-policy/           resolved policy and companion functors
      agentdojo/runs/<pipeline>/...       AgentDojo results, one directory per sweep
      airline_enforce-on_opus/results.json   one directory per tau2 cell
      warmup/...                          the warm-up trials (not part of any table)

Read the results with::

    uv run python scripts/evaluation/summarize_tau2.py results/<ts>/manifest.json
    uv run python scripts/evaluation/summarize_benchmark.py results/<ts>
    <agentdojo python> experiments/agentdojo/analysis/make_table.py --logdir results/<ts>/agentdojo/runs

Prerequisites: a built engine (``make build-rust-dev``); provider credentials in
the environment, ``.env`` or ``.env-credentials`` (``VERTEXAI_PROJECT`` for the
Vertex models, ``AZURE_API_KEY`` / ``AZURE_API_BASE`` / ``AZURE_API_VERSION``
for the Azure ones). For AgentDojo, run ``experiments/agentdojo/setup.sh``;
the runner selects its dedicated environment automatically, with overrides
via ``--agentdojo-python`` or ``AGENTDOJO_PYTHON``. Managed engines derive the
policy-arm key from ``SASY_API_KEY_SUFFIX``; external engines require
``SASY_API_KEY``. An explicit full key takes precedence.

Examples::

    # Print every cell, its command and its output path; run nothing:
    uv run python scripts/evaluation/run_benchmark.py --dry-run

    # The full paper matrix:
    uv run python scripts/evaluation/run_benchmark.py

    # Only tau2 retail, one agent:
    uv run python scripts/evaluation/run_benchmark.py --targets tau2 --domain retail --model opus

    # Continue an interrupted run in place (finished cells are skipped):
    uv run python scripts/evaluation/run_benchmark.py --resume results/<ts>
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from benchmark_capture import CaptureAttempt, sha256, write_json
from benchmark_retry import archive_infrastructure_failures, finished_cells, pending_cell_entries
from manifest_paths import portable_path, rebase_capture_paths

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AGENTDOJO_DIR = REPO_ROOT / "experiments" / "agentdojo"
AGENTDOJO_VENV_PYTHON = AGENTDOJO_DIR / ".venv" / "bin" / "python"
TAU2_DOMAINS_DIR = REPO_ROOT / "experiments" / "tau2-bench" / "data" / "tau2" / "domains"

sys.path.append(str(AGENTDOJO_DIR))
# The presets module imports nothing from AgentDojo, so the matrix can be
# planned (and dry-run) without the benchmark installed.
from sasy_agentdojo.presets import TABLE_SETTINGS as AGENTDOJO_SETTINGS  # noqa: E402
from sasy_agentdojo.presets import Preset  # noqa: E402
from sasy_agentdojo.presets import resolve as resolve_presets  # noqa: E402

# ── AgentDojo Python interpreter resolution ───────────────────────────
#
# Precedence: --agentdojo-python > AGENTDOJO_PYTHON > repo-local venv > current


def resolve_agentdojo_python() -> str:
    """Return the AgentDojo Python interpreter path, resolving defaults.

    Precedence:
    1. AGENTDOJO_PYTHON environment variable (if set)
    2. Repo-local venv at experiments/agentdojo/.venv/bin/python (if exists)
    3. Current interpreter (sys.executable)

    The --agentdojo-python CLI flag overrides this at parse time.
    """
    env_python = os.environ.get("AGENTDOJO_PYTHON")
    if env_python:
        return env_python
    if AGENTDOJO_VENV_PYTHON.exists():
        return str(AGENTDOJO_VENV_PYTHON)
    return sys.executable


# ── API key resolution for managed local engine ───────────────────────
#
# For managed runs (--start-server), compose the API key from
# SASY_API_KEY_SUFFIX using the engine's exact auth semantics.
# This avoids requiring SASY_API_KEY when SASY_API_KEY_SUFFIX is set.


def resolve_sasy_api_key(start_server: bool) -> str | None:
    """Resolve the SASY_API_KEY for AgentDojo policy arms.

    Args:
        start_server: Whether the benchmark is starting its own engine.

    Returns:
        The resolved API key, or None if unavailable for external engines.

    Precedence:
    1. Explicit SASY_API_KEY environment variable always wins.
    2. For managed local-engine runs (start_server=True), compose
       "developer-key-<SASY_API_KEY_SUFFIX>" using the engine's auth.
    3. For external engines (start_server=False), return None to require
       explicit SASY_API_KEY (the suffix might not match the remote).
    """
    explicit = os.environ.get("SASY_API_KEY")
    if explicit:
        return explicit

    if not start_server:
        return None

    suffix = os.environ.get("SASY_API_KEY_SUFFIX", "").strip()
    if suffix:
        return f"developer-key-{suffix}"
    return None

# ── .env loader ────────────────────────────────────────────────────
#
# The bench scripts run from a fresh shell often (CI, cron, scripts
# invoked from another tool) and shouldn't require the operator to
# manually source ``.env``. ``main`` applies ``.env`` and
# ``.env-credentials`` defaults to ``os.environ`` before anything else
# so every subprocess inherits them (importing this module, as the
# tests do, changes nothing). Existing values in the operator's shell
# win — we ``setdefault``, never overwrite.

def _load_dotenv(target, path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        k = k.strip()
        if k:
            target.setdefault(k, v)



# ── Vertex OAuth token pre-warm ────────────────────────────────────
#
# Refresh OAuth credentials in the parent and share the token and expiry with
# workers to avoid concurrent refresh requests. The worker adapter reads these
# environment variables. Refresh before a cell when fewer than 15 minutes remain.
from datetime import timedelta as _td
from datetime import timezone as _tz  # noqa: E402

_VERTEX_TOKEN_ENV = "SASY_VERTEX_ACCESS_TOKEN"
_VERTEX_TOKEN_EXPIRY_ENV = "SASY_VERTEX_ACCESS_TOKEN_EXPIRY"
_VERTEX_TOKEN_REFRESH_MARGIN = _td(minutes=15)
# Cached as (token: str, expiry: datetime [naive UTC]) | None.
_VERTEX_TOKEN_CACHE: tuple[str, datetime] | None = None


def _model_uses_vertex_oauth(model_string: str) -> bool:
    """True for litellm model strings that go through Google OAuth.

    Covers the ``vertex_ai/`` provider (claude-on-vertex,
    gemini-on-vertex, etc.). The ``gemini/`` Direct API uses an API
    key and doesn't hit ``oauth2.googleapis.com``.
    """
    return model_string.startswith("vertex_ai/")


def _refresh_vertex_token() -> tuple[str, datetime]:
    """Run one OAuth refresh in this process. Returns (token, naive-UTC expiry)."""
    import google.auth
    from google.auth.transport.requests import Request

    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(Request())
    if not isinstance(creds.token, str) or creds.expiry is None:
        raise RuntimeError(
            "google.auth.default().refresh() did not yield a token+expiry; "
            "check VERTEXAI_PROJECT and `gcloud auth application-default login`"
        )
    # google-auth uses naive UTC for ``Credentials.expiry``; preserve that.
    return creds.token, creds.expiry


def _inject_vertex_oauth_token(env: dict, *models: str) -> None:
    """Add SASY_VERTEX_ACCESS_TOKEN / _EXPIRY to ``env`` for vertex cells.

    No-op if none of the agent, simulated user or assertion judge uses Vertex.
    Refreshes the cached token when it's within 15 minutes of expiry,
    so workers in this cell will see a token valid for the cell's
    lifetime (tau2 trials run in single-digit minutes).
    """
    global _VERTEX_TOKEN_CACHE
    if not any(_model_uses_vertex_oauth(model) for model in models):
        return

    now = datetime.utcnow()
    if (
        _VERTEX_TOKEN_CACHE is None
        or (_VERTEX_TOKEN_CACHE[1] - now) < _VERTEX_TOKEN_REFRESH_MARGIN
    ):
        token, expiry = _refresh_vertex_token()
        _VERTEX_TOKEN_CACHE = (token, expiry)
        print(
            f"  [vertex-auth] pre-warmed OAuth token; expires "
            f"{expiry.replace(tzinfo=_tz.utc).isoformat()} "
            f"(~{int((expiry - now).total_seconds() / 60)}m from now)"
        )

    token, expiry = _VERTEX_TOKEN_CACHE
    env[_VERTEX_TOKEN_ENV] = token
    # Store as ISO without tz suffix — both sides (parent + worker)
    # treat the value as naive UTC, matching google-auth's convention
    # for ``Credentials.expiry``.
    env[_VERTEX_TOKEN_EXPIRY_ENV] = expiry.isoformat()


def _disp(p: Path) -> str:
    """Render ``p`` relative to the repo root when possible, else
    return its string form. Used for human-readable progress lines —
    the operator may pass an absolute or repo-relative output root,
    and either should print sensibly."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


# ── tau2 ───────────────────────────────────────────────────────────

# The hand-picked subset, selected with ``--task-subset curated`` (see
# scripts/evaluation/translate_tau2_domain.py for why these three).
CURATED_TASK_IDS = {
    "retail": ["15", "16", "91"],
    "airline": ["8", "43", "47"],
}
TAU2_DOMAINS = ("airline", "retail")

# Agent models. Keys are the short names --model and --user-model take.
MODELS = {
    "opus":   "vertex_ai/claude-opus-4-5@20251101",
    "gpt":    "azure/gpt-5.2",
    # gemini-3-pro-preview is retired; 3.1 is the current Gemini Pro.
    "gemini": "vertex_ai/gemini-3.1-pro-preview",
}
# Per-model env defaults (region pins). Only set when the operator has
# not exported the variable already.
MODEL_ENV = {
    "opus":   {"VERTEXAI_LOCATION": "global"},
    "gemini": {"VERTEXAI_LOCATION": "global"},
    "gpt":    {},
}
DEFAULT_USER_MODEL = "gpt"
DEFAULT_NL_JUDGE_MODEL = MODELS["opus"]

ALLOW_ALL_POLICY = "experiments/translation/allow_all.dl"
DOMAIN_POLICY = "experiments/tau2/{domain}_policy.dl"

# Mode → entry point, env it sets, and the policy every trial is pinned
# to (``None``: the entry point pins nothing that decides anything).
TAU2_MODES: dict[str, dict] = {
    # The paper's paired arms. Identical but for the pinned policy.
    "enforce-off": {
        "bin": "tau2-instrumented",
        "env": {"TAU2_SKIP_NL_POLICY": ""},
        "policy": ALLOW_ALL_POLICY,
    },
    "enforce-on": {
        "bin": "tau2-instrumented",
        "env": {"TAU2_SKIP_NL_POLICY": ""},
        "policy": DOMAIN_POLICY,
    },
}
PAPER_TAU2_MODES = ("enforce-off", "enforce-on")

# Inherited variables that would silently change what a tau2 cell measures;
# a cell sets the ones it needs and clears the rest.
_TAU2_SCRUBBED_ENV = (
    "SASY_POLICY_PATH",
    "SASY_TAU2_ALLOW_TENANT_DEFAULT",
    "TAU2_EVALUATION_TYPE",
    "TAU2_NL_JUDGE_MODEL",
)


def tau2_task_ids(domain: str) -> list[str]:
    """Every task id of a domain's ``base`` split, read from the vendored tau2."""
    domain_dir = TAU2_DOMAINS_DIR / domain
    split = json.loads((domain_dir / "split_tasks.json").read_text())
    return [str(t) for t in split["base"]]


def model_string(name: str) -> str:
    """A short model name, or a litellm model string given in full."""
    return MODELS.get(name, name)


# ── info-flow + MALADE ─────────────────────────────────────────────
#
# Both run a single Python entry point with --instrumented True/False
# toggling enforcement vs. NL-prompt-only, against the same engine.
# Each trial pins its own policy (main.py picks the .dl from the
# policy label; omop_observability.py pins malade_policy.dl), so only
# the client certificate differs per cell:
#
#   info-flow toxic       → toxic_flow_policy.dl + developer cert
#   info-flow mls(jsmith) → mls_policy.dl + jsmith cert (TOP_SECRET)
#   info-flow mls(achen)  → mls_policy.dl + achen cert (SECRET)
#   malade                → malade_policy.dl + medical-client cert
#                           (FDA access requires the fda-access role)

INFOFLOW_CELLS: list[dict] = [
    {
        "name": "toxic",
        "policy": "experiments/information-flow/toxic_flow_policy.dl",
        "cert":   "developer",
        "policy_label": "Toxic Flow",
    },
    {
        "name": "mls-jsmith",
        "policy": "experiments/information-flow/mls_policy.dl",
        "cert":   "jsmith",
        "policy_label": "MLS",
        "clearance_label": "TOP_SECRET",
    },
    {
        "name": "mls-achen",
        "policy": "experiments/information-flow/mls_policy.dl",
        "cert":   "achen",
        "policy_label": "MLS",
        "clearance_label": "SECRET",
    },
]

# One NL-only baseline for the whole info-flow target: without
# enforcement the agent never consults the engine, so the policy and
# certificate make no difference and one data point is enough.
_INFOFLOW_NL_SPEC: dict = {
    "name": "nl",
    "policy": None,
    "cert":   "developer",
    "policy_label": "NL Prompt",
}

MALADE_POLICY = "experiments/malade/malade_policy.dl"
MALADE_CERT = "medical-client"


# ── The plan ───────────────────────────────────────────────────────


@dataclass
class Step:
    """One subprocess the run starts: a table cell or a warm-up trial."""

    id: str
    target: str
    command: list[str]
    out_dir: Path
    # Variables set for this step (shown in the plan and the manifest)…
    env: dict[str, str] = field(default_factory=dict)
    # …and inherited variables removed.
    unset: tuple[str, ...] = ()
    # Models the step calls, for the Vertex token pre-warm.
    models: tuple[str, ...] = ()
    # The policy whose compile the warm-up primes (warm-ups only).
    warms: str | None = None
    # Where the cell's results are, for the manifest and the readers.
    result_path: Path | None = None
    meta: dict = field(default_factory=dict)
    warmup: bool = False
    capture: dict | None = None
    # Runtime-only env vars (secrets); excluded from repr, logs, and manifest.
    private_env: dict[str, str] = field(default_factory=dict, repr=False)


def _admin_cert_env() -> dict[str, str]:
    # mTLS cert with policy-upload and observability permissions for tau2.
    # SASY_ADMIN_CERT_PATH /
    # SASY_ADMIN_KEY_PATH point elsewhere when the admin material lives
    # elsewhere.
    return {
        "TLS_CERT_PATH": os.environ.get(
            "SASY_ADMIN_CERT_PATH", str(REPO_ROOT / "certs" / "admin.crt")
        ),
        "TLS_KEY_PATH": os.environ.get(
            "SASY_ADMIN_KEY_PATH", str(REPO_ROOT / "certs" / "admin.key")
        ),
    }


def _tau2_policy(mode: str, domain: str) -> Path | None:
    policy = TAU2_MODES[mode]["policy"]
    return REPO_ROOT / policy.format(domain=domain) if policy else None


def _tau2_step(
    *,
    args: argparse.Namespace,
    run_root: Path,
    domain: str,
    mode: str,
    model: str,
    warmup: bool = False,
) -> Step:
    """The command, env and output directory of one tau2 cell (or its warm-up)."""
    spec = TAU2_MODES[mode]
    cell_name = f"{domain}_{mode}_{model}"
    agent_llm = model_string(model)
    user_llm = agent_llm if args.user_model == "match" else model_string(args.user_model)
    policy = _tau2_policy(mode, domain)

    if warmup:
        out_dir = run_root / "warmup" / cell_name
        task_ids: list[str] | None = [tau2_task_ids(domain)[0]]
        num_trials, concurrency = 1, 1
    else:
        out_dir = run_root / cell_name
        # Explicit --task-ids override --task-subset
        if args.task_ids is not None:
            task_ids = list(args.task_ids)
        elif args.task_subset == "curated":
            task_ids = CURATED_TASK_IDS[domain]
        else:
            task_ids = None
        num_trials, concurrency = args.num_trials, args.max_concurrency

    # An absolute --save-to: tau2 joins it as DATA_DIR / "simulations" /
    # <save_to> (tau2.runner.batch.run_domain), and an absolute right-hand
    # side replaces the whole left side, so tau2 writes
    # <out_dir>/results.json and never touches its vendored data dir.
    # --auto-resume: continue a cell whose results file already exists
    # (--resume) instead of stopping at tau2's interactive prompt.
    cmd = [
        "uv", "run", spec["bin"], "run",
        "--domain", domain,
        "--task-split-name", "base",
        "--num-trials", str(num_trials),
        "--agent-llm", agent_llm,
        "--user-llm", user_llm,
        "--agent-llm-args", json.dumps({}),
        "--user-llm-args", json.dumps({}),
        "--max-concurrency", str(concurrency),
        "--save-to", str(out_dir),
        "--auto-resume",
    ]
    if task_ids:
        cmd += ["--task-ids", *task_ids]

    env: dict[str, str] = dict(spec["env"])
    for name in (model, args.user_model):
        for k, v in MODEL_ENV.get(name, {}).items():
            env.setdefault(k, os.environ.get(k, v))
    env.update(_admin_cert_env())
    if args.capture_latency:
        env["SASY_LATENCY_LOG_FILE"] = str(out_dir / "latency.jsonl")
    if policy is not None:
        env["SASY_POLICY_PATH"] = str(policy)
    if args.evaluation_type:
        env["TAU2_EVALUATION_TYPE"] = args.evaluation_type
    # Retail scores most tasks partly on natural-language assertions,
    # which tau2 hands to an LLM judge (DEFAULT_LLM_NL_ASSERTIONS).
    if domain == "retail" or args.evaluation_type:
        env["TAU2_NL_JUDGE_MODEL"] = args.nl_judge_model
    models = (agent_llm, user_llm)
    if "TAU2_NL_JUDGE_MODEL" in env:
        models += (env["TAU2_NL_JUDGE_MODEL"],)
    # The assertion judge is a provider call too. In GPT-only retail cells
    # it can be the sole Vertex model; Opus must still use the global endpoint.
    if any(_model_uses_vertex_oauth(m) for m in models):
        env.setdefault("VERTEXAI_LOCATION", os.environ.get("VERTEXAI_LOCATION", "global"))
    unset = tuple(k for k in _TAU2_SCRUBBED_ENV if k not in env)

    return Step(
        id=f"{'warmup/' if warmup else ''}tau2/{cell_name}",
        target="tau2",
        command=cmd,
        out_dir=out_dir,
        env=env,
        unset=unset,
        models=models,
        warms=str(policy.relative_to(REPO_ROOT)) if (warmup and policy) else None,
        result_path=out_dir / "results.json",
        meta={
            "domain": domain, "mode": mode, "model": model,
            "agent_llm": agent_llm, "user_llm": user_llm,
            "policy": str(policy.relative_to(REPO_ROOT)) if policy else None,
            "num_trials": num_trials, "concurrency": concurrency,
            "task_ids": task_ids or "all",
        },
        warmup=warmup,
    )


def _agentdojo_pipeline_dirs(preset: Preset, repeats: int) -> str:
    """The run directories one preset's sweeps write, as a pattern.

    Mirrors ``ArmConfig.pipeline_name`` in sasy_agentdojo/runner.py.
    """
    label = AGENTDOJO_SETTINGS["model_label"]
    suffix = "sasy" if preset.arm == "sasy" else (preset.defense or "baseline")
    tag = f"{preset.tag}-r<1..{repeats}>" if repeats > 1 else f"{preset.tag}-r1"
    return f"{label}-{tag}-{suffix}"


def _agentdojo_step(
    *, args: argparse.Namespace, run_root: Path, preset: Preset, warmup: bool = False,
) -> Step:
    """One AgentDojo table cell: ``run.py --preset <name>`` (or its warm-up)."""
    if warmup:
        logdir = run_root / "warmup" / "agentdojo"
        # One clean task on one thread: enough to make the engine compile
        # the policy once before a cell opens dozens of sessions at once.
        cmd = [
            args.agentdojo_python, str(AGENTDOJO_DIR / "run.py"),
            "--preset", preset.name,
            "--attack", "none",
            "--suite", "workspace",
            "--user-task", "user_task_0",
            "--repeats", "1",
            "--threads", "1",
            "--logdir", str(logdir),
        ]
        result = logdir
    else:
        logdir = run_root / "agentdojo" / "runs"
        cmd = [
            args.agentdojo_python, str(AGENTDOJO_DIR / "run.py"),
            "--preset", preset.name,
            "--repeats", str(args.agentdojo_repeats),
            "--threads", str(args.agentdojo_threads),
            "--logdir", str(logdir),
        ]
        for option, values in (
            ("--suite", args.agentdojo_suite),
            ("--user-task", args.agentdojo_user_task),
            ("--injection-task", args.agentdojo_injection_task),
        ):
            for value in values:
                cmd += [option, value]
        result = logdir / _agentdojo_pipeline_dirs(preset, args.agentdojo_repeats)
    return Step(
        id=f"{'warmup/' if warmup else ''}agentdojo/{preset.name}",
        target="agentdojo",
        command=cmd,
        out_dir=logdir,
        # The presets pin AgentDojo's shared policy per session; a stray
        # override would replace it.
        unset=("SASY_POLICY_PATH",),
        warms="experiments/agentdojo/policies/agentdojo_policy.dl" if warmup else None,
        result_path=result,
        meta={
            "preset": preset.name, "tag": preset.tag, "row": preset.row,
            "arm": preset.arm, "defense": preset.defense,
            "switches": sorted(preset.switches),
            "repeat_user_prompt": preset.repeat_user_prompt,
            "repeats": 1 if warmup else args.agentdojo_repeats,
            "logdir": str(logdir),
            "suites": ["workspace"] if warmup else args.agentdojo_suite or "all",
            "user_tasks": ["user_task_0"] if warmup else args.agentdojo_user_task or "all",
            "injection_tasks": [] if warmup else args.agentdojo_injection_task or "all",
        },
        warmup=warmup,
    )


def _infoflow_step(*, args: argparse.Namespace, run_root: Path, spec: dict,
                   instrumented: bool) -> Step:
    name = spec["name"]
    mode_tag = "instr" if instrumented else "nl"
    cell_name = f"infoflow_{name}_{mode_tag}"
    out_dir = run_root / cell_name
    cmd = [
        "uv", "run", "python",
        "experiments/information-flow/information_flow/main.py",
        "--model", "gpt-4.1-mini",
        "--azure", "True",
        "--instrumented", str(instrumented),
        "--nl_policy", str(not instrumented),
        "--trials", str(args.example_num_trials),
        "--max_concurrency", str(args.example_max_concurrency),
        "--concurrency_mode", "processes",
        "--policy_label", spec.get("policy_label", name),
        "--clearance_label", spec.get("clearance_label", ""),
        "--output_dir", str(out_dir),
        "--suffix", cell_name,
        "--env_credentials", "True",
    ]
    cert = spec["cert"]
    return Step(
        id=f"infoflow/{name}_{mode_tag}",
        target="infoflow",
        command=cmd,
        out_dir=out_dir,
        env={
            "TLS_CERT_PATH": str(REPO_ROOT / "certs" / f"{cert}.crt"),
            "TLS_KEY_PATH": str(REPO_ROOT / "certs" / f"{cert}.key"),
        },
        # main.py pins the policy its --policy_label names unless this is set.
        unset=("SASY_POLICY_PATH",),
        # The example writes results-<model>-<label>-<suffix>.json here.
        result_path=out_dir,
        meta={"name": name, "instrumented": instrumented,
              "policy": spec["policy"], "cert": cert},
    )


def _malade_step(*, args: argparse.Namespace, run_root: Path, instrumented: bool) -> Step:
    mode_tag = "instr" if instrumented else "nl"
    out_dir = run_root / f"malade_{mode_tag}"
    results_path = out_dir / "results.json"
    cmd = [
        "uv", "run", "python",
        "experiments/malade/malade/omop_observability.py",
        "--trials", str(args.example_num_trials),
        "--max_concurrency", str(args.example_max_concurrency),
        "--concurrency_mode", "processes",
        "--instrumented", str(instrumented),
        "--nl_policy", str(not instrumented),
        "--cache_path", str(results_path),
        "--env_credentials", "True",
    ]
    return Step(
        id=f"malade/{mode_tag}",
        target="malade",
        command=cmd,
        out_dir=out_dir,
        env={
            "TLS_CERT_PATH": str(REPO_ROOT / "certs" / f"{MALADE_CERT}.crt"),
            "TLS_KEY_PATH": str(REPO_ROOT / "certs" / f"{MALADE_CERT}.key"),
        },
        # omop_observability.py pins malade_policy.dl unless this is set.
        unset=("SASY_POLICY_PATH",),
        result_path=results_path,
        meta={"instrumented": instrumented, "policy": MALADE_POLICY, "cert": MALADE_CERT},
    )


def build_plan(args: argparse.Namespace, run_root: Path, done: set[str]) -> list[Step]:
    """Every step of the run, in order, with a warm-up before the first
    pending cell on each policy. Cells whose id is in ``done`` are left out."""
    cells: list[Step] = []
    warmups: dict[str, Step] = {}  # cell id → warm-up to run before it
    warmed: set[str] = set()

    def add(cell: Step, warmup: Step | None) -> None:
        if cell.id in done:
            return
        if warmup is not None and args.warmup:
            key = warmup.warms or warmup.id
            if key not in warmed:
                warmed.add(key)
                warmups[cell.id] = warmup
        cells.append(cell)

    if "agentdojo" in args.targets:
        for preset in args.agentdojo_presets:
            cell = _agentdojo_step(args=args, run_root=run_root, preset=preset)
            warm = (
                _agentdojo_step(args=args, run_root=run_root, preset=preset, warmup=True)
                if preset.arm == "sasy" else None
            )
            add(cell, warm)

    if "tau2" in args.targets:
        # Arms are adjacent within each (domain, agent), so provider-side
        # drift over a long run lands on both arms of a pair alike.
        for domain in args.domains:
            for model in args.models:
                for mode in args.modes:
                    cell = _tau2_step(args=args, run_root=run_root,
                                      domain=domain, mode=mode, model=model)
                    warm = (
                        _tau2_step(args=args, run_root=run_root, domain=domain,
                                   mode=mode, model=model, warmup=True)
                        if TAU2_MODES[mode]["policy"] else None
                    )
                    add(cell, warm)

    if "infoflow" in args.targets:
        for spec in INFOFLOW_CELLS:
            add(_infoflow_step(args=args, run_root=run_root, spec=spec, instrumented=True), None)
        add(_infoflow_step(args=args, run_root=run_root, spec=_INFOFLOW_NL_SPEC,
                           instrumented=False), None)

    if "malade" in args.targets:
        for instrumented in (True, False):
            add(_malade_step(args=args, run_root=run_root, instrumented=instrumented), None)

    plan: list[Step] = []
    for cell in cells:
        if cell.id in warmups:
            plan.append(warmups[cell.id])
        plan.append(cell)
    return plan


# ── Running a step ─────────────────────────────────────────────────


def _step_env(step: Step) -> dict[str, str]:
    env = os.environ.copy()
    for k in step.unset:
        env.pop(k, None)
    env.update(step.env)
    env.update(step.private_env)
    return env


def _print_step(step: Step) -> None:
    kind = "warm-up" if step.warmup else step.target
    print("\n══════════════════════════════════════════════════════════════════")
    print(f" [{kind}] {step.id}")
    print(f"   output: {_disp(step.result_path or step.out_dir)}")
    print(f"   command: {' '.join(shlex.quote(c) for c in step.command)}")
    if step.env:
        shown = {k: (_disp(Path(v)) if k.endswith(("_PATH", "_FILE")) and v else v)
                 for k, v in step.env.items()}
        print(f"   env: {shown}")
    if step.unset:
        print(f"   unset: {', '.join(step.unset)}")
    if step.warms:
        print(f"   primes the compile cache for: {step.warms}")
    print("══════════════════════════════════════════════════════════════════")


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    except ProcessLookupError:
        pass


def run_step(step: Step, timeout_s: int) -> tuple[int, float]:
    """Run one step in its own process group; return (returncode, seconds).

    The child gets its own process group so Ctrl+C / SIGTERM / a timeout
    can take down the whole tree: tau2 issues blocking LLM calls from a
    thread pool whose threads ignore SIGINT, and a stalled call can park
    a cell indefinitely. A timed-out step returns 124; whatever trials it
    finished are already on disk, and ``--resume`` continues it.
    """
    env = _step_env(step)
    if step.target == "tau2":
        _inject_vertex_oauth_token(env, *step.models)
    step.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    proc = subprocess.Popen(
        step.command, env=env, cwd=REPO_ROOT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        rc = proc.wait(timeout=timeout_s or None)
    except subprocess.TimeoutExpired:
        print(f"  ! timeout after {timeout_s}s; killing process group {proc.pid}")
        _kill_group(proc)
        rc = 124
    except BaseException:
        print(f"  ! interrupted; killing process group {proc.pid}")
        _kill_group(proc)
        raise
    elapsed = time.monotonic() - started
    print(f"  [{elapsed:.0f}s] returncode={rc}")
    return rc, elapsed


def _manifest_entry(step: Step, run_root: Path, rc: int | None, elapsed: float | None) -> dict:
    """Create a manifest entry for a completed step.

    Internal artifacts (results inside run_root) are stored as relative paths.
    External paths and operational values (command argv, env) stay absolute.
    Capture paths are rebased from cell-relative to run-manifest-relative.
    """
    entry: dict = {"id": step.id, "target": step.target, **step.meta}
    entry["command"] = step.command
    entry["env"] = step.env
    entry["unset"] = list(step.unset)
    if step.capture is not None:
        entry["latency_capture"] = rebase_capture_paths(step.capture, step.out_dir, run_root)
    if step.warms:
        entry["warms"] = step.warms
    entry["returncode"] = rc
    if elapsed is not None:
        entry["wall_s"] = round(elapsed, 1)
    if step.result_path is not None:
        path = step.result_path
        if step.target in ("infoflow",) and path.is_dir():
            matches = sorted(path.glob("results-*.json"), key=lambda p: p.stat().st_mtime)
            path = matches[-1] if matches else path
        key = "sim_path" if step.target in ("tau2", "infoflow") else "results_path"
        try:
            entry[key] = str(path.relative_to(run_root))
        except ValueError:
            entry[key] = str(path)
    return entry


# ── Optional per-run SASY server ─────────────────────────────────────
#
# By default the benchmark starts its own engine for the run, on its own
# port (11089) with a fresh data directory under the run directory, so
# the graph holds only this run and the compile cache starts empty (which
# is what the warm-ups are for). ``--no-start-server`` uses whichever
# engine ``SASY_URL`` points at instead.
#
# The spawned server is torn down at the end via the same signal
# handler that catches Ctrl+C / SIGTERM, so a kill -9 of the
# benchmark process is the only failure mode that orphans it.

_SERVER_BIN = REPO_ROOT / "sasy" / "target" / "debug" / "sasy"

_managed_servers: list[subprocess.Popen] = []


def _stop_managed_servers() -> None:
    """Terminate any sasy server subprocesses we started."""
    for proc in _managed_servers:
        if proc.poll() is not None:
            continue
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        except ProcessLookupError:
            pass
    _managed_servers.clear()


def _wait_for_port(host: str, port: int, timeout_s: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            try:
                s.connect((host, port))
                return True
            except (ConnectionRefusedError, OSError):
                time.sleep(0.5)
    return False


def _start_sasy_server(
    *,
    port: int,
    data_dir: Path,
    initial_policy: Path | None,
    log_path: Path,
    dry_run: bool = False,
    latency_log: Path | None = None,
    server_bin: Path | None = None,
) -> subprocess.Popen | None:
    """Spawn a fresh ``sasy serve`` instance for this run.

    Returns the Popen handle once the port is listening. The handle
    is registered for teardown via the SIGTERM/SIGINT handler.
    Caller is responsible for setting ``SASY_URL`` so cell
    subprocesses target this instance.
    """
    bin_path = server_bin if server_bin is not None else _SERVER_BIN
    if not bin_path.exists():
        message = (
            f"sasy binary not found at {_disp(bin_path)}; "
            "run `make build-rust-dev` first."
        )
        if not dry_run:
            raise RuntimeError(message)
        print(f"[dry-run] warning: {message}")
    if initial_policy is not None and not initial_policy.exists():
        raise RuntimeError(f"initial policy not found: {initial_policy}")

    cmd = [
        str(bin_path), "serve",
        "--evaluator", "souffle",
        "--auth-provider", str(REPO_ROOT / "config" / "auth" / "all.json"),
        "--auth-config", str(REPO_ROOT / "config" / "auth_config.yaml"),
        "--transforms", str(REPO_ROOT / "config" / "transforms.json"),
        "--tls-cert", str(REPO_ROOT / "certs" / "reference-monitor.crt"),
        "--tls-key", str(REPO_ROOT / "certs" / "reference-monitor.key"),
        "--tls-ca", str(REPO_ROOT / "certs" / "ca.crt"),
        "--addr", f"0.0.0.0:{port}",
        "--data-dir", str(data_dir),
        # The memory backend hands the deployment keys in .env-credentials to
        # every tenant; the MALADE cell runs as tenant `malade`, which a
        # sqlite store seeded under tenant `default` would not serve.
        "--credential-backend", "memory",
        "--credentials-env-file", str(REPO_ROOT / ".env-credentials"),
        # Development host only. Each trial binds its domain policy
        # per session, and the Python SDK attaches the companion
        # `<policy>_functors.cpp` next to it (airline_functors.cpp,
        # retail_functors.cpp). The cell credential is the .env client
        # cert, which is not an admin, so without this opt-in every
        # bind is refused with PERMISSION_DENIED. The unsandboxed
        # value is used because a benchmark host is typically a Mac,
        # which has no bubblewrap. Do not copy into a deployment: it
        # lets a non-admin caller compile and run C++ unconfined.
        "--allow-user-functors=unsandboxed",
    ]

    print(f"[server] starting sasy serve on :{port}")
    print(f"[server]   data-dir: {_disp(data_dir)}")
    # Without --policy the engine's tenant default is its built-in
    # deny-all: a session that pins no policy is refused, not let through.
    if initial_policy is not None:
        cmd += ["--policy", str(initial_policy)]
    print(f"[server]   tenant default: "
          f"{_disp(initial_policy) if initial_policy else 'built-in deny-all'}")
    print(f"[server]   log:      {_disp(log_path)}")

    if dry_run:
        print(f"[dry-run] {' '.join(shlex.quote(c) for c in cmd)}")
        return None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"server port {port} is already in use; choose --server-port")

    data_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Inherit env so .env-credentials values (LLM keys, etc.) flow
    # through to the server process.
    env = os.environ.copy()
    env.setdefault("LLM_ENABLED", "true")
    if latency_log is not None:
        latency_log.parent.mkdir(parents=True, exist_ok=True)
        env["SASY_SERVER_LATENCY_LOG_FILE"] = str(latency_log)

    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            cmd, env=env, cwd=REPO_ROOT,
            stdout=log_fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _managed_servers.append(proc)

    if not _wait_for_port("127.0.0.1", port, timeout_s=60) or proc.poll() is not None:
        # Surface the tail of the server log so the operator can
        # diagnose without hunting through filesystem.
        try:
            tail = "\n".join(log_path.read_text().splitlines()[-30:])
        except OSError:
            tail = "(log unreadable)"
        raise RuntimeError(
            f"sasy server failed to bind :{port} within 60s — last log lines:\n{tail}"
        )

    print(f"[server] ready on :{port} (pid {proc.pid})")
    return proc


def _install_termination_handler() -> None:
    """Translate SIGTERM into KeyboardInterrupt so the same cleanup
    path runs on ``kill <pid>`` as on Ctrl+C. Without this, SIGTERM
    just exits Python without unwinding the ``except KeyboardInterrupt``
    block in ``_run_cell``, and tau2 child processes (which are in
    their own process group) survive as orphans.

    Also tears down any sasy server we started via ``--start-server``
    (registered in ``_managed_servers``), so the run-scoped instance
    doesn't outlive the benchmark process.
    """
    def _on_term(signum, frame):  # noqa: ARG001
        _stop_managed_servers()
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    import atexit
    atexit.register(_stop_managed_servers)


def _expand(values: list[str], everything: tuple[str, ...], paper: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Expand ``all`` (and ``paper``) in a multi-value option, keeping order."""
    out: list[str] = []
    for v in values:
        picked = everything if v == "all" else (paper if v == "paper" and paper else (v,))
        out += [p for p in picked if p not in out]
    return tuple(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "output_root", nargs="?", type=Path, default=None,
        help="Directory the run's <timestamp>/ directory is created in"
        " (default: results). Same as --output-root.",
    )
    parser.add_argument(
        "--output-root", dest="output_root_opt", type=Path, default=None,
        help="Same as the positional output root.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None, metavar="RUN_DIR",
        help="Continue the run in RUN_DIR (an results/<timestamp>"
        " directory) instead of starting a new one. Cells its manifest"
        " records as finished (return code 0) are skipped; a tau2 cell"
        " that was cut short resumes from its results file and"
        " AgentDojo skips runs whose result file exists. Pass the same"
        " selection flags as the original run.",
    )
    parser.add_argument(
        "--retry-infrastructure-errors", action="store_true",
        help="With --resume, requeue rc=0 tau2 cells that still contain infrastructure"
        " failures. Tau2 retries only those task/trial/seed keys; evaluated"
        " outcomes are preserved. Original checkpoints are archived before retry.",
    )
    parser.add_argument(
        "--targets", nargs="+", default=["paper"],
        choices=("agentdojo", "tau2", "infoflow", "malade", "paper", "all"),
        help="Benchmarks to run (default: paper = agentdojo tau2 infoflow malade;"
        " all selects the same targets).",
    )

    tau2 = parser.add_argument_group("tau2")
    tau2.add_argument(
        "--domain", dest="domain", nargs="+", default=["all"],
        choices=(*TAU2_DOMAINS, "all"), help="Domains (default: all).",
    )
    tau2.add_argument(
        "--mode", dest="mode", nargs="+", default=["paper"],
        choices=(*TAU2_MODES, "paper", "all"),
        help="Arms (default: paper = enforce-off enforce-on). sasy hides"
        " the policy text and enforces the domain policy.",
    )
    tau2.add_argument(
        "--model", dest="model", nargs="+", default=["all"],
        choices=(*MODELS, "all"),
        help="Agent models (default: all = " + ", ".join(MODELS) + ").",
    )
    tau2.add_argument(
        "--user-model", default=DEFAULT_USER_MODEL,
        help="Simulated user: a short model name (" + ", ".join(MODELS) + "),"
        " a litellm model string, or 'match' for the agent's own model"
        f" (default: {DEFAULT_USER_MODEL}, the same user for every agent).",
    )
    tau2.add_argument(
        "--num-trials", type=int, default=4, help="Trials per task (default: 4).",
    )
    tau2.add_argument(
        "--task-subset", choices=("all", "curated"), default="all",
        help="all: every task of the domain's base split (airline 50,"
        " retail 114; the default). curated: the three hand-picked tasks"
        " per domain.",
    )
    tau2.add_argument(
        "--task-ids", nargs="+", default=None, metavar="ID",
        help="Explicit task IDs to run (overrides --task-subset). Must be"
        " in the domain's base split. Only supported for single-domain runs.",
    )
    tau2.add_argument(
        "--max-concurrency", type=int, default=32,
        help="Concurrent trials within a tau2 cell (default: 32).",
    )
    tau2.add_argument(
        "--nl-judge-model", default=DEFAULT_NL_JUDGE_MODEL,
        help="Model tau2 uses to judge natural-language assertions; set"
        f" for retail cells (default: {DEFAULT_NL_JUDGE_MODEL}).",
    )
    tau2.add_argument(
        "--evaluation-type", default=None,
        help="Set TAU2_EVALUATION_TYPE (for example all_with_nl_assertions)."
        " By default it is cleared and tau2 scores each task on its own"
        " reward basis.",
    )

    adj = parser.add_argument_group("AgentDojo")
    adj.add_argument(
        "--agentdojo-preset", dest="agentdojo_preset", action="append", default=None,
        metavar="NAME",
        help="A run.py preset, group or table tag; repeatable (default: table,"
        " the eight cells).",
    )
    adj.add_argument(
        "--agentdojo-repeats", type=int, default=AGENTDOJO_SETTINGS["repeats"],
        help=f"Sweeps per cell (default: {AGENTDOJO_SETTINGS['repeats']}).",
    )
    adj.add_argument(
        "--agentdojo-threads", type=int, default=32,
        help="Concurrent AgentDojo runs within a cell (default: 32).",
    )
    adj.add_argument(
        "--agentdojo-python", default=None,
        help="Python with AgentDojo and the sasy SDK installed (default:"
        " $AGENTDOJO_PYTHON, else experiments/agentdojo/.venv if it exists,"
        " else this interpreter). Run experiments/agentdojo/setup.sh to create"
        " the dedicated environment.",
    )

    adj.add_argument("--agentdojo-suite", action="append", default=[],
                     choices=("banking", "slack", "travel", "workspace"),
                     help="Restrict AgentDojo suites; repeatable (default: all four).")
    adj.add_argument("--agentdojo-user-task", action="append", default=[],
                     help="Restrict AgentDojo user task ids; repeatable.")
    adj.add_argument("--agentdojo-injection-task", action="append", default=[],
                     help="Restrict AgentDojo injection task ids; repeatable.")

    ex = parser.add_argument_group("info-flow and MALADE (--targets infoflow malade)")
    ex.add_argument(
        "--example-num-trials", type=int, default=5,
        help="Trials per info-flow / MALADE cell (default: 5).",
    )
    ex.add_argument(
        "--example-max-concurrency", type=int, default=5,
        help="Worker processes per info-flow / MALADE cell (default: 5).",
    )

    run = parser.add_argument_group("running")
    run.add_argument(
        "--dry-run", action="store_true",
        help="Print every step (cell or warm-up), its command, env and output"
        " path; run nothing and write nothing.",
    )
    run.add_argument(
        "--warmup", action=argparse.BooleanOptionalAction, default=True,
        help="Before the first cell on each policy, run one trial of one task"
        " on one worker so the engine compiles the policy once (default: on).",
    )
    run.add_argument(
        "--continue-on-error", action="store_true",
        help="Run the remaining cells after one fails.",
    )
    run.add_argument(
        "--cell-timeout", type=int, default=21600,
        help="Wall-clock limit per cell in seconds (default: 21600 = 6 h;"
        " 0 = none). A full-domain tau2 cell at 32 workers takes one to"
        " three hours; the limit is for a stalled LLM call.",
    )

    run.add_argument(
        "--capture-latency", action=argparse.BooleanOptionalAction, default=True,
        help="Capture tau2 SDK timing and managed-engine authorization requests"
        " for replay without new model calls (default: on).",
    )
    srv = parser.add_argument_group("engine")
    # ``BENCH_START_SERVER`` is honoured for explicit overrides ("0"
    # disables, "1" enables); when unset, start a run-scoped engine.
    _bench_start_default = os.environ.get("BENCH_START_SERVER")
    srv.add_argument(
        "--start-server", action=argparse.BooleanOptionalAction,
        default=(_bench_start_default != "0" if _bench_start_default is not None else True),
        help="Start a fresh engine on --server-port for this run, with its"
        " data under the run directory, and point every cell at it"
        " (default: on). --no-start-server (or BENCH_START_SERVER=0) uses"
        " the engine SASY_URL names.",
    )
    srv.add_argument(
        "--server-port", type=int, default=int(os.environ.get("BENCH_SERVER_PORT", "11089")),
        help="Port of the run-scoped engine (default: 11089, or BENCH_SERVER_PORT).",
    )
    srv.add_argument(
        "--server-data-dir", type=Path, default=None,
        help="Data directory of the run-scoped engine (default:"
        " <run dir>/server-data).",
    )
    srv.add_argument(
        "--server-policy", type=Path, default=None,
        help="Tenant default policy of the run-scoped engine. Every cell"
        " pins its own policy, so this only decides what a session that"
        " pins nothing gets (default: the engine's built-in deny-all).",
    )
    srv.add_argument(
        "--server-latency-log", type=Path, default=None,
        help="Server request log to capture. Managed default: <run>/server_latency.jsonl."
        " For an external engine, configure its SASY_SERVER_LATENCY_LOG_FILE to this path.",
    )
    srv.add_argument(
        "--server-bin", type=Path, default=None,
        help="Path to the sasy binary. Default: sasy/target/debug/sasy."
        " Used in startup, preflight, provenance and hash computation.",
    )
    return parser


def _resolve_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.retry_infrastructure_errors and args.resume is None:
        parser.error("--retry-infrastructure-errors requires --resume")
    args.targets = _expand(args.targets, ("agentdojo", "tau2", "infoflow", "malade"),
                           paper=("agentdojo", "tau2", "infoflow", "malade"))
    args.domains = _expand(args.domain, TAU2_DOMAINS)
    args.modes = _expand(args.mode, tuple(TAU2_MODES), paper=PAPER_TAU2_MODES)
    args.models = _expand(args.model, tuple(MODELS))
    try:
        args.agentdojo_presets = resolve_presets(args.agentdojo_preset or ["table"])
    except ValueError as e:
        parser.error(str(e))
    if args.output_root and args.output_root_opt and args.output_root != args.output_root_opt:
        parser.error("give the output root once, positionally or with --output-root")
    args.output_root = args.output_root or args.output_root_opt or REPO_ROOT / "results"
    for name in ("num_trials", "max_concurrency", "agentdojo_repeats", "agentdojo_threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.agentdojo_python is None:
        args.agentdojo_python = resolve_agentdojo_python()
    # Resolve server binary path
    args.resolved_server_bin = (args.server_bin or _SERVER_BIN).expanduser().resolve()
    # Validate --task-ids for tau2 targets
    if args.task_ids is not None:
        # --task-ids requires tau2 target
        if "tau2" not in args.targets:
            parser.error("--task-ids requires --targets tau2")
        # Reject duplicate task IDs
        seen = set()
        for tid in args.task_ids:
            if tid in seen:
                parser.error(f"--task-ids: duplicate ID '{tid}'")
            seen.add(tid)
        # Validate single domain
        if len(args.domains) != 1:
            parser.error("--task-ids requires a single domain (use --domain airline or --domain retail)")
        domain = args.domains[0]
        valid_ids = set(tau2_task_ids(domain))
        for tid in args.task_ids:
            if tid not in valid_ids:
                parser.error(
                    f"--task-ids: '{tid}' not in {domain} base split. "
                    f"Valid: {', '.join(sorted(valid_ids, key=int)[:10])}..."
                )


def _preflight(args: argparse.Namespace, plan: list[Step]) -> list[str]:
    """Problems that would make a cell fail at once, found before anything runs."""
    problems: list[str] = []
    if any(s.target == "agentdojo" for s in plan):
        setup_cmd = "./experiments/agentdojo/setup.sh"
        try:
            probe = subprocess.run(
                [args.agentdojo_python, "-c", "import agentdojo, sasy"],
                capture_output=True, text=True,
            )
        except OSError:
            problems.append(
                f"AgentDojo interpreter not found or not executable: {args.agentdojo_python}\n"
                f"  Run: {setup_cmd}\n"
                f"  Or pass --agentdojo-python / set AGENTDOJO_PYTHON to a Python\n"
                f"  environment with agentdojo==0.1.35 and the sasy SDK installed."
            )
            probe = None
        if probe is not None and probe.returncode != 0:
            problems.append(
                f"{args.agentdojo_python} cannot import agentdojo and sasy.\n"
                f"  Run: {setup_cmd}\n"
                f"  Or pass --agentdojo-python / set AGENTDOJO_PYTHON to a Python\n"
                f"  environment with agentdojo==0.1.35 and the sasy SDK installed."
            )
        has_sasy_arm = any(
            s.target == "agentdojo" and s.meta.get("arm") == "sasy" for s in plan
        )
        if has_sasy_arm:
            resolved_key = resolve_sasy_api_key(args.start_server)
            if resolved_key is None:
                if args.start_server:
                    problems.append(
                        "SASY_API_KEY or SASY_API_KEY_SUFFIX must be set for the"
                        " AgentDojo policy arm. With a managed engine, set"
                        " SASY_API_KEY_SUFFIX in .env (make serve writes one)."
                    )
                else:
                    problems.append(
                        "SASY_API_KEY is not set. The AgentDojo policy arm needs it."
                        " For an external engine (--no-start-server), the API key"
                        " must match the target engine's configuration."
                    )
    return problems


def main() -> int:
    _load_dotenv(os.environ, REPO_ROOT / ".env")
    _load_dotenv(os.environ, REPO_ROOT / ".env-credentials")
    _install_termination_handler()
    parser = build_parser()
    args = parser.parse_args()
    _resolve_args(parser, args)

    previous: dict = {}
    if args.resume is not None:
        run_root = args.resume.resolve()
        manifest_path = run_root / "manifest.json"
        if not manifest_path.is_file():
            parser.error(f"--resume: no manifest.json in {run_root}")
        previous = json.loads(manifest_path.read_text())
        timestamp = previous.get("timestamp", run_root.name)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # Absolute, so the paths in the manifest and in the commands work
        # from any directory.
        run_root = (args.output_root / timestamp).resolve()
        manifest_path = run_root / "manifest.json"

    selected = [s for s in build_plan(args, run_root, done=set()) if not s.warmup]
    selected_ids = {s.id for s in selected}
    checkpoint_paths = {s.id: s.result_path for s in selected if s.target == "tau2"}
    finished = finished_cells(
        previous, run_root, args.retry_infrastructure_errors, selected_ids, checkpoint_paths,
    )
    plan = build_plan(args, run_root, done=set(finished))
    cells = [s for s in plan if not s.warmup]

    print(f"Run directory: {_disp(run_root)}")
    if finished:
        print(f"Skipping {len(finished)} finished cell(s) recorded in the manifest.")
    print(f"{len(cells)} cell(s), {len(plan) - len(cells)} warm-up(s):")
    for step in plan:
        print(f"  {step.id:<46} → {_disp(step.result_path or step.out_dir)}")

    server_log = None
    if args.capture_latency:
        server_log = (args.server_latency_log.resolve() if args.server_latency_log else
                      run_root / "server_latency.jsonl" if args.start_server else None)
    data_dir = (args.server_data_dir or run_root / "server-data").resolve()
    manifest: dict = {
        "timestamp": timestamp,
        "run_dir": ".",
        "targets": list(args.targets),
        # tau2 settings (top-level names kept for older readers).
        "domains": list(args.domains),
        "modes": list(args.modes),
        "models": list(args.models),
        "model_strings": {m: MODELS[m] for m in args.models},
        "user_llm": (
            {m: MODELS[m] for m in args.models} if args.user_model == "match"
            else model_string(args.user_model)
        ),
        "num_trials": args.num_trials,
        "max_concurrency": args.max_concurrency,
        "task_ids": (
            args.task_ids if args.task_ids is not None
            else "all" if args.task_subset == "all"
            else {d: CURATED_TASK_IDS[d] for d in args.domains}
        ),
        "nl_judge_model": args.nl_judge_model,
        "agentdojo": {
            **{k: v for k, v in AGENTDOJO_SETTINGS.items() if k not in ("repeats", "threads")},
            "presets": [p.name for p in args.agentdojo_presets],
            "repeats": args.agentdojo_repeats,
            "threads": args.agentdojo_threads,
            "suites": args.agentdojo_suite or "all",
            "user_tasks": args.agentdojo_user_task or "all",
            "injection_tasks": args.agentdojo_injection_task or "all",
            "logdir": "agentdojo/runs",
        },
        "example_num_trials": args.example_num_trials,
        "example_max_concurrency": args.example_max_concurrency,
        "retry_infrastructure_errors": args.retry_infrastructure_errors,
        "start_server": args.start_server,
        "backend": {
            "data_dir": portable_path(data_dir, run_root) if args.start_server else None,
            "url": f"localhost:{args.server_port}" if args.start_server else os.environ.get("SASY_URL"),
            "binary": str(args.resolved_server_bin) if args.start_server else None,
        },
        "latency_capture": {
            "version": 1, "enabled": args.capture_latency,
            "server_log": portable_path(server_log, run_root) if server_log else None,
            "graph_state": "persisted-at-stop",
            "state": "running",
        },
        "warmups": list(previous.get("warmups", [])),
        # Keep pending/requeued entries until an attempt records a replacement.
        # Normalize paths for moved runs when resuming.
        "cells": pending_cell_entries(
            previous, {s.id for s in cells}, new_manifest_dir=run_root,
        ),
    }

    if args.dry_run:
        if args.start_server:
            _start_sasy_server(
                port=args.server_port,
                data_dir=args.server_data_dir or (run_root / "server-data"),
                initial_policy=args.server_policy,
                log_path=run_root / "server.log",
                dry_run=True,
                server_bin=args.resolved_server_bin,
            )
        for step in plan:
            _print_step(step)
        print(f"\n[dry-run] {len(cells)} cell(s), {len(plan) - len(cells)} warm-up(s);"
              f" manifest would be {_disp(manifest_path)}")
        return 0

    problems = _preflight(args, plan)
    if problems:
        for p in problems:
            print(f"ERROR: {p}")
        return 2

    # Inject the resolved API key ONLY into AgentDojo policy-arm steps.
    # The derived key must never leak to other targets (infoflow, malade,
    # tau2), which use mTLS cert auth: an inherited SASY_API_KEY would
    # replace the cert identity with the developer identity.
    # - private_env is excluded from repr, logs, and manifest.
    # - Explicit operator SASY_API_KEY flows via os.environ to ALL cells.
    agentdojo_sasy_steps = [
        s for s in plan
        if s.target == "agentdojo" and s.meta.get("arm") == "sasy"
    ]
    if agentdojo_sasy_steps and args.start_server:
        # Only derive when the operator hasn't set an explicit full key.
        if not os.environ.get("SASY_API_KEY"):
            resolved_key = resolve_sasy_api_key(start_server=True)
            if resolved_key:
                for step in agentdojo_sasy_steps:
                    step.private_env["SASY_API_KEY"] = resolved_key
                print("[auth] Using developer-key-<suffix> for AgentDojo policy arm")

    run_root.mkdir(parents=True, exist_ok=True)

    def save_manifest() -> None:
        write_json(manifest_path, manifest)

    save_manifest()

    if args.start_server:
        _start_sasy_server(
            port=args.server_port,
            data_dir=args.server_data_dir or (run_root / "server-data"),
            initial_policy=args.server_policy,
            log_path=run_root / "server.log",
            latency_log=server_log,
            server_bin=args.resolved_server_bin,
        )
        # Every step inherits os.environ.
        os.environ["SASY_URL"] = f"localhost:{args.server_port}"
        print(f"[server] cells will use SASY_URL=localhost:{args.server_port}")

    if args.capture_latency and args.start_server:
        manifest["latency_capture"]["engine_sha256"] = sha256(args.resolved_server_bin)
        save_manifest()
    failures: list[tuple[str, int]] = []
    try:
        for step in plan:
            _print_step(step)
            prior_entry = next((c for c in manifest["cells"] if c["id"] == step.id), None)
            if prior_entry is not None and "previous_attempt" in prior_entry:
                step.meta["previous_attempt"] = prior_entry["previous_attempt"]
            if step.target == "tau2" and step.result_path is not None:
                retry = archive_infrastructure_failures(step.result_path)
                if retry is not None:
                    step.meta["infrastructure_retry"] = retry
                    print(f"  Retry {retry['failure_count']} infrastructure failure(s);"
                          f" original checkpoint: {retry['snapshot']}")
            attempt = None
            if args.capture_latency and step.target == "tau2":
                step.out_dir.mkdir(parents=True, exist_ok=True)
                attempt = CaptureAttempt(
                    step.out_dir, server_log, Path(step.env["SASY_POLICY_PATH"]),
                    engine_sha256=manifest["latency_capture"].get("engine_sha256"),
                    concurrency=step.meta["concurrency"],
                )
            started = time.monotonic()
            rc = 1
            elapsed = 0.0
            try:
                rc, elapsed = run_step(step, args.cell_timeout)
            except KeyboardInterrupt:
                rc = 130
                raise
            finally:
                if attempt is not None:
                    elapsed = time.monotonic() - started
                    step.capture = attempt.finish(rc, elapsed)
                entry = _manifest_entry(step, run_root, rc, elapsed)
                if step.warmup:
                    manifest["warmups"].append(entry)
                else:
                    manifest["cells"] = [c for c in manifest["cells"] if c.get("id") != step.id] + [entry]
                save_manifest()
            if step.warmup:
                if rc != 0:
                    print(f"  ! warm-up {step.id} failed (rc={rc}); the cell runs anyway")
                continue
            if rc != 0:
                failures.append((step.id, rc))
                if not args.continue_on_error:
                    break
    finally:
        _stop_managed_servers()
        manifest["latency_capture"]["state"] = "stopped"
        save_manifest()

    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for cell_id, rc in failures:
            print(f"  - {cell_id}: returncode {rc}")
        print(f"Continue with: --resume {_disp(run_root)}")
        return 1
    print(f"All {len(cells)} cell(s) finished. Manifest: {_disp(manifest_path)}")
    if "tau2" in args.targets:
        print(f"  tau2:      uv run python scripts/evaluation/summarize_tau2.py {_disp(manifest_path)}")
    if "agentdojo" in args.targets:
        print(f"  AgentDojo: uv run python scripts/evaluation/summarize_agentdojo.py"
              f" {_disp(manifest_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
