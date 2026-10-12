"""Shared harness installer for every ``forge_tau2`` entry point.

The ``tau2-instrumented`` CLI calls :func:`install_harness` before
``tau2.cli.main``. It opens a pinned SASY session per trial and installs
optional evaluation overrides in the v1.0.1 batch runner.

Targets the tau2-bench v1.0.1 batch runner. ``tau2 run`` goes
``run_domain`` -> ``run_tasks`` -> ``run_with_retry`` ->
``tau2.runner.batch.run_single_task``. ``run_tasks`` looks
``run_single_task`` up as a module global each time a trial runs, so
replacing ``tau2.runner.batch.run_single_task`` wraps every trial
attempt (retries included). The old ``tau2.run.run_task`` is a
deprecated shim the batch path never calls, so wrapping it does
nothing.

What v1.0.1 already does, and this module therefore no longer adds:
``run_with_retry`` (``tau2/runner/progress.py``) retries any failed
trial and, once retries run out, returns an ``INFRASTRUCTURE_ERROR``
``SimulationRun`` instead of raising, so one crashing trial cannot
take down the whole thread pool.

Every patch here checks its target first and raises
:class:`HarnessPatchError` if the target is missing or has a
different shape, rather than quietly doing nothing.

Importing this module sets ``GRPC_ENABLE_FORK_SUPPORT=false``. SASY
entry points import it before anything that loads gRPC, because gRPC
reads the variable once, at import time. (It turns off gRPC's fork
handlers, which otherwise warn "Other threads are currently calling
into gRPC" under tau2's thread pool.)
"""

from __future__ import annotations

import argparse
import functools
import inspect
import logging
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

logger = logging.getLogger(__name__)

# Attribute set on each wrapper so a second install is a no-op.
_MARKER_ATTR = "__sasy_harness_patch__"


class HarnessPatchError(RuntimeError):
    """A harness patch target is missing or changed shape in tau2."""


class MissingDomainPolicyError(SystemExit):
    """A SASY-enforcing ``tau2 run`` has no domain policy to pin.

    A ``SystemExit`` subclass, so the entry point stops with the
    message on stderr and exit status 1 instead of a traceback.
    """


# Opt-out for MissingDomainPolicyError: run trials on the tenant's
# default policy when no domain policy is found.
ALLOW_TENANT_DEFAULT_ENV = "SASY_TAU2_ALLOW_TENANT_DEFAULT"


class _ProbeParserError(Exception):
    """The probe parser rejected argv, or argv asked for help."""


class _ProbeParser(argparse.ArgumentParser):
    """An ``ArgumentParser`` that never prints and never exits.

    ``-h`` / ``--help`` and parse errors raise
    :class:`_ProbeParserError` instead, so probing argv has no
    visible effect and leaves the real ``tau2`` parser to report
    them.
    """

    def _print_message(self, message: str, file: Any = None) -> None:
        return None

    def exit(self, status: int = 0, message: str | None = None) -> Any:
        raise _ProbeParserError(message or "")

    def error(self, message: str) -> Any:
        raise _ProbeParserError(message)


def _parse_run_command(argv: list[str] | None = None) -> argparse.Namespace | None:
    """Parse argv the way ``tau2``'s CLI parses a ``run`` command.

    ``tau2.cli.main`` builds its parser inside the function, so it
    cannot be imported. This builds the same ``run`` sub-parser with
    tau2's own :func:`tau2.cli.add_run_args` under a top-level parser
    with a ``command`` sub-command, as ``main`` does, so every
    spelling tau2 accepts for an option (``-d X``, ``-dX``,
    ``--domain=X``, abbreviations, repeated options where the last
    wins) is read exactly as tau2 reads it.

    Returns the parsed ``run`` arguments, or ``None`` when argv is not
    a ``tau2 run`` that would start trials: another sub-command, no
    sub-command, a help request, or arguments tau2 would reject.
    """
    from tau2.cli import add_run_args

    argv = sys.argv if argv is None else argv
    parser = _ProbeParser(add_help=True)
    subparsers = parser.add_subparsers(dest="command")
    add_run_args(subparsers.add_parser("run"))
    try:
        # parse_args, as tau2.cli.main does: an option tau2 does not
        # know makes it exit with a usage error, so it is left to tau2
        # to report rather than read as a run with no domain.
        parsed = parser.parse_args(argv[1:])
    except _ProbeParserError:
        return None
    if getattr(parsed, "command", None) != "run":
        return None
    return parsed


def detect_domain(argv: list[str] | None = None) -> str | None:
    """Return the ``tau2 run`` domain from argv, as tau2 would read it.

    ``argv`` is a full command line including the program name
    (default ``sys.argv``). Returns ``None`` when argv is not a
    ``tau2 run`` command or names no domain. See
    :func:`_parse_run_command`.
    """
    parsed = _parse_run_command(argv)
    return getattr(parsed, "domain", None) if parsed is not None else None


def resolve_domain_policy(argv: list[str] | None = None) -> Path | None:
    """Choose the policy file each trial's session is pinned to.

    Called by ``tau2-instrumented``. Resolution order:
      1. ``SASY_POLICY_PATH`` env var (path to a .dl file)
      2. ``experiments/tau2/{domain}_policy.dl`` for the
         ``--domain`` of the ``tau2 run`` command
      3. no policy found for a ``tau2 run``: raise
         :class:`MissingDomainPolicyError`, unless
         ``SASY_TAU2_ALLOW_TENANT_DEFAULT=1`` is set, in which case
         return ``None`` and sessions use the tenant's default policy

    Returns ``None`` without checking anything when argv is not a
    ``tau2 run`` that starts trials (``--help``, ``view``, ...).

    Returns a ``Path`` so ``sasy.session(policy=...)`` reads it as a
    file and resolves ``#include`` lines and companion functors.
    """
    override = os.environ.get("SASY_POLICY_PATH")
    if override:
        return Path(override)
    parsed = _parse_run_command(argv)
    if parsed is None:
        return None
    domain = getattr(parsed, "domain", None)
    policy_dir = Path(__file__).resolve().parent.parent
    if domain is not None:
        candidate = policy_dir / f"{domain}_policy.dl"
        if candidate.exists():
            return candidate
        problem = (
            f"no SASY policy for domain {domain!r}: expected {candidate}"
        )
    else:
        problem = (
            "no --domain on the tau2 run command line, so no domain policy"
            f" can be chosen (expected {policy_dir}/<domain>_policy.dl)"
        )
    if os.environ.get(ALLOW_TENANT_DEFAULT_ENV) == "1":
        logger.warning(
            "%s; %s=1 is set, so trial sessions run on the tenant's"
            " default policy.",
            problem,
            ALLOW_TENANT_DEFAULT_ENV,
        )
        return None
    raise MissingDomainPolicyError(
        f"error: {problem}. Set SASY_POLICY_PATH to a policy file, or set"
        f" {ALLOW_TENANT_DEFAULT_ENV}=1 to run trials on the tenant's"
        " default policy."
    )


def _batch_module() -> Any:
    """Return ``tau2.runner.batch`` after checking the patch target.

    Checks that ``run_single_task`` exists, takes ``config`` and
    ``task``, and takes ``evaluation_type`` as a keyword-only
    argument, and that ``run_tasks`` exists.
    """
    try:
        from tau2.runner import batch
    except ImportError as exc:
        raise HarnessPatchError(
            f"cannot import tau2.runner.batch ({exc}); the tau2 harness"
            " targets the tau2-bench v1.0.1 runner layout"
        ) from exc
    for name in ("run_single_task", "run_tasks"):
        if not callable(getattr(batch, name, None)):
            raise HarnessPatchError(
                f"tau2.runner.batch.{name} is missing; the tau2 harness"
                " targets the tau2-bench v1.0.1 runner layout"
            )
    params = inspect.signature(
        inspect.unwrap(batch.run_single_task)
    ).parameters
    names = list(params)
    if names[:2] != ["config", "task"]:
        raise HarnessPatchError(
            "tau2.runner.batch.run_single_task should start with"
            f" (config, task, ...); found {tuple(names)}"
        )
    eval_param = params.get("evaluation_type")
    if eval_param is None or eval_param.kind is not inspect.Parameter.KEYWORD_ONLY:
        raise HarnessPatchError(
            "tau2.runner.batch.run_single_task should take a keyword-only"
            f" evaluation_type argument; found {tuple(names)}"
        )
    return batch


def _is_installed(fn: Any, marker: str) -> bool:
    """True if ``fn`` or anything it wraps carries ``marker``."""
    seen: set[int] = set()
    while fn is not None and id(fn) not in seen:
        if getattr(fn, _MARKER_ATTR, None) == marker:
            return True
        seen.add(id(fn))
        fn = getattr(fn, "__wrapped__", None)
    return False


def _wrap_run_single_task(
    marker: str,
    make_wrapper: Callable[[Callable[..., Any]], Callable[..., Any]],
) -> None:
    """Replace ``tau2.runner.batch.run_single_task`` with a wrapper."""
    batch = _batch_module()
    original = batch.run_single_task
    if _is_installed(original, marker):
        return
    wrapper = make_wrapper(original)
    functools.update_wrapper(wrapper, original)
    setattr(wrapper, _MARKER_ATTR, marker)
    batch.run_single_task = wrapper


def _task_from_call(args: tuple, kwargs: dict) -> Any:
    if "task" in kwargs:
        return kwargs["task"]
    return args[1] if len(args) > 1 else None


def install_evaluation_type_override() -> None:
    """Honor ``TAU2_EVALUATION_TYPE`` (e.g. ``all_with_nl_assertions``).

    The tau2 CLI always evaluates with ``EvaluationType.ALL``;
    ``run_tasks`` passes it to ``run_single_task`` as the keyword-only
    ``evaluation_type`` argument, which this wrapper replaces. An
    unknown value raises rather than being ignored.
    """
    raw = os.environ.get("TAU2_EVALUATION_TYPE")
    if not raw:
        return
    try:
        from tau2.evaluator.evaluator import EvaluationType
    except ImportError as exc:
        raise HarnessPatchError(
            f"cannot import tau2.evaluator.evaluator.EvaluationType ({exc})"
        ) from exc
    try:
        target = EvaluationType(raw)
    except ValueError as exc:
        valid = ", ".join(e.value for e in EvaluationType)
        raise HarnessPatchError(
            f"TAU2_EVALUATION_TYPE={raw!r} is not a valid tau2"
            f" EvaluationType (valid: {valid})"
        ) from exc

    def make(original: Callable[..., Any]) -> Callable[..., Any]:
        def evaluation_type_run_single_task(*args: Any, **kwargs: Any) -> Any:
            kwargs["evaluation_type"] = target
            return original(*args, **kwargs)

        return evaluation_type_run_single_task

    _wrap_run_single_task("evaluation_type", make)
    logger.info("evaluation_type override: %s", target.value)


def install_nl_judge_model_override() -> None:
    """Honor ``TAU2_NL_JUDGE_MODEL=azure/gpt-5.5`` (or similar).

    Sets ``DEFAULT_LLM_NL_ASSERTIONS`` in ``tau2.config`` and in
    ``tau2.evaluator.evaluator_nl_assertions``, which copies the
    constant by name when it is imported and reads its own copy when
    it calls the judge.
    """
    override = os.environ.get("TAU2_NL_JUDGE_MODEL")
    if not override:
        return
    from tau2 import config as tau2_config
    from tau2.evaluator import evaluator_nl_assertions

    for module in (tau2_config, evaluator_nl_assertions):
        if not hasattr(module, "DEFAULT_LLM_NL_ASSERTIONS"):
            raise HarnessPatchError(
                f"{module.__name__}.DEFAULT_LLM_NL_ASSERTIONS is missing;"
                " cannot apply TAU2_NL_JUDGE_MODEL"
            )
        setattr(module, "DEFAULT_LLM_NL_ASSERTIONS", override)
    logger.info("NL judge model override: %s", override)

    # tau2 parses the judge's reply with json.loads directly. The default
    # judge returns bare JSON; other models (e.g. Claude) may wrap it in a
    # ```json fence, which fails to parse and makes tau2 rerun the whole
    # simulation and then record infrastructure_error. Strip a fence that
    # surrounds the reply, for the judge's calls only.
    if not hasattr(evaluator_nl_assertions, "generate"):
        raise HarnessPatchError(
            "tau2.evaluator.evaluator_nl_assertions.generate is missing;"
            " cannot unwrap fenced judge replies"
        )
    original_generate = evaluator_nl_assertions.generate
    if getattr(original_generate, "_sasy_unfenced", False):
        return

    def unfenced_generate(*args: Any, **kwargs: Any) -> Any:
        message = original_generate(*args, **kwargs)
        content = getattr(message, "content", None)
        if kwargs.get("call_name") == "nl_assertions_eval" and isinstance(content, str):
            match = re.fullmatch(r"\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*", content, re.S)
            if match:
                message.content = match.group(1)
        return message

    unfenced_generate._sasy_unfenced = True  # type: ignore[attr-defined]
    evaluator_nl_assertions.generate = unfenced_generate


def install_per_trial_session(policy: object | None) -> None:
    """Run each trial attempt inside its own ``sasy.session``.

    ``run_tasks`` runs trials on a ``ThreadPoolExecutor``; a worker
    thread starts with default ContextVars, so opening the session
    inside ``run_single_task`` gives each trial its own graph
    partition and policy evaluator on the SASY server. A retry is a
    new call, so it gets a fresh session.

    ``policy`` (a path or Soufflé source) pins every trial's session
    to that policy; ``None`` leaves sessions on the tenant default.
    The server dedupes identical uploads by content hash, so the
    per-trial cost is the session's evaluator start, not a recompile.
    """
    import sasy

    def make(original: Callable[..., Any]) -> Callable[..., Any]:
        def session_scoped_run_single_task(*args: Any, **kwargs: Any) -> Any:
            # A fresh UUID per attempt by default, so concurrent trials
            # and reruns never share a partition. SASY_SESSION_READABLE=1
            # adds the task id to the name, for filtering logs.
            if os.environ.get("SASY_SESSION_READABLE") == "1":
                from uuid import uuid4

                task = _task_from_call(args, kwargs)
                task_id = str(getattr(task, "id", "unknown"))
                sid: str | None = f"tau2-task-{task_id}-{uuid4().hex[:8]}"
            else:
                sid = None
            with sasy.session(sid, policy=policy):
                return original(*args, **kwargs)

        return session_scoped_run_single_task

    _wrap_run_single_task("per_trial_session", make)


def install_harness(
    *,
    sasy_sessions: bool,
    policy: object | None = None,
) -> None:
    """The single installer every ``forge_tau2`` entry point calls.

    Always checks the batch-runner patch target (so a tau2 layout
    change fails at startup), then applies the optional
    ``TAU2_EVALUATION_TYPE`` / ``TAU2_NL_JUDGE_MODEL`` overrides. With
    ``sasy_sessions=True`` it also opens a SASY session per trial,
    pinned to ``policy`` (``None`` = tenant default). Offline harness
    callers can pass ``sasy_sessions=False`` to install only the
    evaluation overrides without importing ``sasy``.
    """
    _batch_module()
    install_evaluation_type_override()
    install_nl_judge_model_override()
    if sasy_sessions:
        install_per_trial_session(policy)
