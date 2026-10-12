"""Measure a generated policy against labeled captured decision points.

Usage:
    uv run python experiments/translation/generated_policy_fidelity.py \\
        --policy path/to/generated.dl \\
        --scenarios path/to/labeled_tests.json \\
        [--policy-nl path/to/policy.md] [--report path/to/out.json] \\
        [--min-fidelity 0.9]

Exit code 0 = the measurement completed; 1 = it did not, or
`--min-fidelity` was given and not met.
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

from policy_translation.evaluation.fidelity import *
from policy_translation.evaluation.fidelity import main

if __name__ == "__main__":
    raise SystemExit(main())
