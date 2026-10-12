"""Validate (and optionally analyze) a Soufflé policy file.

Usage:
    python scripts/validate_policy.py path/to/policy.dl
    python scripts/validate_policy.py --analyze path/to/policy.dl
    python scripts/validate_policy.py --analyze --json path/to/policy.dl
    python scripts/validate_policy.py --no-rpc path/to/policy.dl   # force local

Examples are also reachable via Make:
    make souffle-validate FILE=experiments/malade/malade_policy.dl
    make souffle-analyze  FILE=experiments/malade/malade_policy.dl
"""

from __future__ import annotations

import sys
from pathlib import Path

# `make souffle-validate` and `make policy-test` run these files with a
# bare `python3`, which need not be the interpreter the package is
# installed into, so name the distribution's directory as well. It is
# appended rather than inserted, so an install still wins.
_DISTRIBUTION = Path(__file__).resolve().parents[1] / "packages" / "policy-translation"
if str(_DISTRIBUTION) not in sys.path:
    sys.path.append(str(_DISTRIBUTION))

from policy_translation.evaluation.validate import *
from policy_translation.evaluation.validate import main

if __name__ == "__main__":
    raise SystemExit(main())
