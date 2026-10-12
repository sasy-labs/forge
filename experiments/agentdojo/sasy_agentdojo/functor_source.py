"""Read the functor source as a single self-contained translation unit.

The pure helpers live in `agentdojo_functors_impl.h` so that the unit test and
the shipping functors compile the *same* code — a copy-pasted second version is
how a test ends up green while a different, weaker rule runs in production.

The engine, though, receives the functor source as one string and compiles it in
a scratch directory that contains nothing else, so a local `#include` of that
header cannot resolve there. This inlines it at load time: one source of truth on
disk, one self-contained payload on the wire.
"""

from __future__ import annotations

import re
from pathlib import Path

_LOCAL_INCLUDE = re.compile(r'^[ \t]*#include[ \t]*"([^"]+)"[ \t]*$', re.MULTILINE)


def load(path: Path, _seen: frozenset[Path] = frozenset()) -> str:
    """`path`'s text with every local `#include "..."` replaced by its contents.

    Angle-bracket includes are system headers and are left alone. A header
    already inlined on this path is replaced with nothing, which is what its
    include guard would have done anyway and which stops a cycle from looping.
    """
    path = path.resolve()
    text = path.read_text()

    def substitute(match: re.Match[str]) -> str:
        target = (path.parent / match.group(1)).resolve()
        if target in _seen or target == path:
            return ""
        if not target.exists():
            raise FileNotFoundError(f"{path}: cannot resolve #include \"{match.group(1)}\"")
        return load(target, _seen | {path})

    return _LOCAL_INCLUDE.sub(substitute, text)
