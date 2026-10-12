"""Compare a candidate policy with labeled authorization scenarios.

Usage:
    uv run python experiments/translation/policy_convergence.py \\
        --policy path/to/policy.dl \\
        --scenarios path/to/labeled_tests.json \\
        [--policy-nl path/to/policy.md] \\
        [--report path/to/report.json] [--fail-on-disputed]

Exit code 0 = converged, 1 = not converged (or errors).
"""

from __future__ import annotations

import sys
from pathlib import Path

# `make souffle-validate` and `make policy-test` run these files with a
# bare `python3`, which need not be the interpreter the package is
# installed into, so name the distribution's directory as well. It is
# appended rather than inserted, so an install still wins.
_DISTRIBUTION = Path(__file__).resolve().parents[2] / "packages" / "policy-translation"
if str(_DISTRIBUTION) not in sys.path:
    sys.path.append(str(_DISTRIBUTION))

from policy_translation.evaluation.convergence import *
from policy_translation.evaluation.convergence import main

if __name__ == "__main__":
    raise SystemExit(main())
