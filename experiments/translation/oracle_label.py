"""Label recorded authorization decisions with a model-based policy oracle.

Usage:
    uv run python experiments/translation/oracle_label.py \\
        --scenarios path/to/scenarios.json \\
        --policy-nl path/to/policy.md \\
        [--policy-class general|taint] [--provider anthropic|litellm]
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

from policy_translation.evaluation.oracle_label import *
from policy_translation.evaluation.oracle_label import main

if __name__ == "__main__":
    raise SystemExit(main())
