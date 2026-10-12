"""Shared access to the public implementation's Soufflé tooling.

The setup script exposes the pinned implementation's preprocessing and functor
sources through local compatibility paths. This module resolves those paths,
expands policy includes, desugars policy syntax, and records source hashes.
`resolve_includes` uses an explicitly selected common policy; `desugar` uses
the preprocessor's default. `SASY_REPO_ROOT` can override the repository anchor,
and `SASY_HARNESS_CACHE_DIR` selects the compiled-functor cache directory.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    """The checkout this package was installed from.

    `SASY_REPO_ROOT` wins when set, so a caller running against a different
    checkout says so explicitly. Otherwise walk up from this file looking for
    the tooling itself, which is what every path below is relative to, and fall
    back to the layout `packages/<distribution>/<package>/` implies.
    """

    override = os.environ.get("SASY_REPO_ROOT")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "sasy-services" / "souffle" / "sugar.py").is_file():
            return candidate
    return here.parents[3]


REPO_ROOT = _repo_root()
SOUFFLE_DIR = REPO_ROOT / "sasy-services" / "souffle"
SUGAR_SCRIPT = SOUFFLE_DIR / "sugar.py"
FUNCTORS_SRC = SOUFFLE_DIR / "functors_common.cpp"
# Per-policy functor libraries are content-hashed and cached here, so a
# policy with a companion <name>_functors.cpp gets its own .so rather
# than silently running against the shared functors_common.cpp alone.
HARNESS_CACHE_DIR = Path(os.environ.get("SASY_HARNESS_CACHE_DIR", SOUFFLE_DIR / ".harness-cache"))
# The policy every other policy inherits its relations from: what an offline
# run composes a rule fragment against when the fragment includes nothing.
DEFAULT_COMMON_POLICY = REPO_ROOT / "policies" / "common_policy.dl"

# The pass has no reason to take a minute on a single policy; a hang is a
# defect, not a slow machine. The value is the one `resolve_includes`'s caller
# used before the call moved here, and that caller catches the expiry itself,
# so `resolve_includes` still lets it out. `desugar`'s callers never had a
# timeout and read only the return code, so `desugar` turns an expiry into one.
SUGAR_TIMEOUT_SECONDS = 60
# What `desugar` returns when the pass ran past that bound: the return code a
# shell reports for a command its `timeout` stopped.
SUGAR_TIMEOUT_RETURNCODE = 124


def resolve_includes(
    source: Path | str,
    common_policy: Path | str,
    timeout: int = SUGAR_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run the pinned sugar pass over `source` and return the finished process.

    `common_policy` is passed through the environment as `COMMON_POLICY_DL`,
    which is how `sugar.py` decides which common policy an unqualified
    `#include` resolves to. Nothing is raised on a non-zero exit: the caller
    reads `returncode`, `stdout` and `stderr` and phrases its own refusal.
    """

    environment = dict(os.environ)
    environment["COMMON_POLICY_DL"] = str(common_policy)
    return subprocess.run(
        [sys.executable, str(SUGAR_SCRIPT), "--resolve-includes", str(source)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _captured_text(captured: object) -> str:
    """What a stopped child had written so far, as text.

    `subprocess.run` decodes its output under `text=True`, but a run that hits
    its timeout does not go through that decoding: CPython raises
    `TimeoutExpired` from inside the communication loop, carrying the raw byte
    buffers it had read. So `TimeoutExpired.stdout` and `.stderr` are `bytes`
    where `CompletedProcess.stdout` and `.stderr` are `str`, and a child that
    printed one diagnostic line before it hung would otherwise turn the seam's
    refusal into a `TypeError`. Undecodable bytes are replaced rather than
    raising: this text is a reason string a caller prints, not data.
    """

    if isinstance(captured, bytes):
        return captured.decode("utf-8", "replace")
    if isinstance(captured, str):
        return captured
    return ""


def desugar(
    source: Path | str,
    sugar_script: Path | str = SUGAR_SCRIPT,
    timeout: int = SUGAR_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Run the sugar pass over `source` with no common policy chosen for it.

    `sugar.py` picks its own default common policy when `COMMON_POLICY_DL` is
    unset, and that default is what the offline harness and the validator
    resolved their `#include` directives against before this call moved here;
    unlike `resolve_includes`, this entry leaves the environment alone so that
    stays true. Nothing is raised, on a non-zero exit or on a pass that runs
    past `timeout`: the caller reads `returncode`, `stdout` and `stderr` and
    phrases its own refusal. The two callers this seam serves had no timeout
    before the move and handle a failure by reading `returncode`, so an expiry
    is reported the same way -- return code 124, the shell convention for a
    command a timeout stopped, with the reason in `stderr` -- rather than
    travelling out as a `TimeoutExpired` traceback past their refusal.

    `sugar_script` is a parameter because the harness's `desugar_policy` takes
    the script from its caller; the default is the pinned one.
    """

    argv = [sys.executable, str(sugar_script), "--resolve-includes", str(source)]
    try:
        return subprocess.run(argv, text=True, capture_output=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as expiry:
        return subprocess.CompletedProcess(
            argv,
            returncode=SUGAR_TIMEOUT_RETURNCODE,
            stdout=_captured_text(expiry.stdout),
            stderr=_captured_text(expiry.stderr)
            + f"sugar.py did not finish within {timeout} seconds and was stopped\n",
        )


def sugar_pin() -> dict[str, Any]:
    """Pin the bytes that ran: the sugar pass and the interpreter running it.

    A provenance record naming a text pass has to say *which* text pass, or a
    later reader cannot tell whether the recorded output is reproducible. Both
    entries are digests of files on this host, read at call time.
    """

    interpreter = Path(sys.executable).resolve()
    return {
        "path": str(SUGAR_SCRIPT.resolve()),
        "sha256": hashlib.sha256(SUGAR_SCRIPT.read_bytes()).hexdigest(),
        "python": {
            "path": str(interpreter),
            "sha256": hashlib.sha256(interpreter.read_bytes()).hexdigest(),
            "bytes": interpreter.stat().st_size,
        },
    }
