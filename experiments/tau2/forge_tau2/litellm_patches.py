"""litellm monkey-patches for the tau2-bench harness.

**Vertex OAuth token from env (+ in-process serialization)** — at high
concurrency, every fresh worker thread refreshing its own OAuth token
causes ``oauth2.googleapis.com`` to drop TLS handshakes
(``SSLEOFError: UNEXPECTED_EOF_WHILE_READING``), not 429s. The parent
process pre-fetches a token and exports ``SASY_VERTEX_ACCESS_TOKEN`` /
``SASY_VERTEX_ACCESS_TOKEN_EXPIRY``; our patched ``refresh_auth`` reads
from those env vars instead of calling ``credentials.refresh(Request())``.
As a backstop, when the env vars are absent or stale, a process-wide
``threading.Lock`` + token cache serializes the network refresh so at
most one worker thread opens an OAuth handshake at a time, even at
``--max-concurrency 600``.

The patch targets ``VertexBase.refresh_auth(self, credentials) -> None``
in ``litellm.llms.vertex_ai.vertex_llm_base`` (unchanged across the
``litellm>=1.80.15,<1.82.7`` range that tau2 v1.0.1 requires).

Gemini-3 ``thought_signature`` handling for multi-turn function calling
is native in that litellm range (``convert_to_gemini_tool_call_invoke``
reads the signature from ``provider_specific_fields`` or the
``__thought__``-encoded tool_call_id and falls back to Google's dummy
signature for gemini-3 models), so no backport is needed any more.

Apply once at process startup with :func:`apply_litellm_patches`.
Idempotent (safe to call from multiple entry points).
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Vertex OAuth token reuse
# ────────────────────────────────────────────────────────────────────

# Env vars the parent process exports after a successful OAuth refresh.
# Workers pick these up in the patched ``refresh_auth`` below to skip
# their own ``oauth2.googleapis.com`` handshake.
_VERTEX_TOKEN_ENV = "SASY_VERTEX_ACCESS_TOKEN"
_VERTEX_TOKEN_EXPIRY_ENV = "SASY_VERTEX_ACCESS_TOKEN_EXPIRY"

# Safety margin: if the parent-supplied token has less than this much
# validity left, fall back to a fresh refresh in the worker. The
# parent should refresh well before this triggers (it checks ~15 min
# margin); this is just a backstop for very long-running cells.
_VERTEX_TOKEN_MIN_MARGIN = timedelta(minutes=2)

# Process-wide lock + cache for the worker-side OAuth refresh fallback.
# litellm builds a fresh ``VertexBase`` instance per completion, so its
# per-instance ``_credentials_project_mapping`` cache doesn't help across
# concurrent calls. At ``--max-concurrency 600``, hundreds of threads can
# enter ``refresh_auth`` simultaneously and each open its own TLS
# handshake to ``oauth2.googleapis.com``, which then drops a fraction of
# them with ``SSLEOFError: UNEXPECTED_EOF_WHILE_READING``. The lock
# serializes the network refresh; the cache lets the second-through-Nth
# caller reuse the result without re-opening the handshake.
_VERTEX_TOKEN_LOCK = threading.Lock()
_VERTEX_TOKEN_PROCESS_CACHE: Optional[tuple[str, datetime]] = None


def _parent_supplied_vertex_token() -> Optional[tuple[str, datetime]]:
    """Return ``(token, naive_utc_expiry)`` if env vars are valid, else None.

    Returns None when the env vars aren't set, can't be parsed, or the
    token is already within ``_VERTEX_TOKEN_MIN_MARGIN`` of expiry.
    """
    token = os.environ.get(_VERTEX_TOKEN_ENV)
    expiry_iso = os.environ.get(_VERTEX_TOKEN_EXPIRY_ENV)
    if not token or not expiry_iso:
        return None
    try:
        expiry = datetime.fromisoformat(expiry_iso)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to per-worker OAuth refresh",
            _VERTEX_TOKEN_EXPIRY_ENV, expiry_iso,
        )
        return None
    # google-auth uses naive UTC for ``Credentials.expiry``. Normalize
    # tz-aware ISO strings to naive UTC so the comparison and the
    # value we set on the credentials object both match google-auth's
    # convention.
    if expiry.tzinfo is not None:
        expiry = expiry.astimezone(timezone.utc).replace(tzinfo=None)
    if expiry - datetime.utcnow() < _VERTEX_TOKEN_MIN_MARGIN:
        return None
    return token, expiry


def _process_cached_vertex_token() -> Optional[tuple[str, datetime]]:
    """Return the cached (token, naive-UTC expiry) if still fresh."""
    cached = _VERTEX_TOKEN_PROCESS_CACHE
    if cached is None:
        return None
    _, expiry = cached
    if expiry - datetime.utcnow() < _VERTEX_TOKEN_MIN_MARGIN:
        return None
    return cached


def _patched_vertex_refresh_auth(self, credentials: Any) -> None:
    """Drop-in replacement for ``VertexBase.refresh_auth``.

    Three paths, in order:

    1. Parent-supplied env-var token (no network, no lock).
    2. Process-wide cached token from a prior in-process refresh
       (no network, lock-checked under the cache check).
    3. One real ``credentials.refresh(Request())`` per process at a
       time, serialized by ``_VERTEX_TOKEN_LOCK`` so concurrent threads
       don't stampede ``oauth2.googleapis.com``. The result is cached
       in ``_VERTEX_TOKEN_PROCESS_CACHE`` so the next thread waiting on
       the lock skips the network call entirely.

    All paths leave ``credentials.token`` / ``credentials.expiry``
    populated, matching upstream's contract.
    """
    parent = _parent_supplied_vertex_token()
    if parent is not None:
        token, expiry = parent
        credentials.token = token
        credentials.expiry = expiry
        return

    cached = _process_cached_vertex_token()
    if cached is not None:
        token, expiry = cached
        credentials.token = token
        credentials.expiry = expiry
        return

    from google.auth.transport.requests import Request  # type: ignore[import-untyped]
    global _VERTEX_TOKEN_PROCESS_CACHE
    with _VERTEX_TOKEN_LOCK:
        # Double-checked: another thread may have refreshed while we
        # were waiting on the lock. Re-read both caches before issuing
        # a real network call.
        parent = _parent_supplied_vertex_token()
        if parent is not None:
            token, expiry = parent
            credentials.token = token
            credentials.expiry = expiry
            return
        cached = _process_cached_vertex_token()
        if cached is not None:
            token, expiry = cached
            credentials.token = token
            credentials.expiry = expiry
            return

        credentials.refresh(Request())
        if isinstance(credentials.token, str) and credentials.expiry is not None:
            _VERTEX_TOKEN_PROCESS_CACHE = (credentials.token, credentials.expiry)


_PATCHES_APPLIED = False


def apply_litellm_patches() -> None:
    """Install monkey-patches for litellm.

    Idempotent. Safe to call multiple times — only the first call
    actually swaps function pointers; subsequent calls are no-ops.

    Patches:
      - ``litellm.llms.vertex_ai.vertex_llm_base.VertexBase.refresh_auth``
        replaced with a version that consumes a parent-supplied OAuth
        token from ``SASY_VERTEX_ACCESS_TOKEN`` /
        ``SASY_VERTEX_ACCESS_TOKEN_EXPIRY`` env vars, falling back to
        a process-wide-cached + lock-serialized network refresh when
        those are absent. Avoids TLS-handshake stampedes on
        ``oauth2.googleapis.com`` at high tau2 ``--max-concurrency``.
    """
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return

    from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

    VertexBase.refresh_auth = _patched_vertex_refresh_auth

    _PATCHES_APPLIED = True
    logger.info(
        "litellm patches applied: vertex OAuth token reuse (parent-supplied: %s)",
        "yes" if _parent_supplied_vertex_token() is not None else "no",
    )

    # Optional: enable litellm verbose debug logging. Useful for
    # diagnosing slow trials (rate-limit retries, network hangs,
    # provider-side delays). Default off — only on when an operator
    # explicitly asks for it via TAU2_LITELLM_DEBUG=1, since the log
    # volume is significant.
    if os.environ.get("TAU2_LITELLM_DEBUG", "").strip().lower() in (
        "1", "true", "yes", "on"
    ):
        import litellm
        try:
            litellm._turn_on_debug()
            logger.warning(
                "litellm debug mode enabled — log volume will be high"
            )
        except Exception as e:
            logger.warning("Failed to enable litellm debug: %s", e)
