"""Point the SDK at a local SASY binary.

The SDK defaults to the cloud deployment, so a local run has to say so. The
developer certificates and the API key here are the ones the repository's test
fixtures use against `make serve`; a real deployment supplies its own.
"""

from __future__ import annotations

import os
from pathlib import Path

from sasy.auth.hooks import APIKeyAuthHook
from sasy.config import configure

REPO_ROOT = Path(__file__).resolve().parents[3]


def connect_local(url: str | None = None, api_key: str | None = None) -> None:
    api_key = api_key or os.environ.get("SASY_API_KEY")
    if not api_key:
        raise ValueError("Set SASY_API_KEY to the full API key or pass api_key explicitly")
    certs = REPO_ROOT / "certs"
    configure(
        url=url or os.environ.get("SASY_URL", "localhost:10089"),
        auth_hook=APIKeyAuthHook(
            api_key=api_key
        ),
        ca_path=str(certs / "ca.crt") if (certs / "ca.crt").exists() else None,
        cert_path=str(certs / "developer.crt") if (certs / "developer.crt").exists() else None,
        key_path=str(certs / "developer.key") if (certs / "developer.key").exists() else None,
    )
