"""Run Souffle policies against test scenarios in interpreted mode.

Usage:
    python3 scripts/policy_test_harness.py \\
        --policy path/to/policy.dl \\
        --scenarios path/to/tests.json

Exit code 0 = all pass, 1 = any failure.
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

from policy_translation.evaluation.harness import *
from policy_translation.evaluation.harness import main

if __name__ == "__main__":
    raise SystemExit(main())
