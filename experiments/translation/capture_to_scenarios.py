"""Convert captured decision points into policy test scenarios.

Usage:
    python3 experiments/translation/capture_to_scenarios.py \\
        --capture path/to/capture.jsonl [--out path/to/scenarios.json]
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

from policy_translation.inputs.capture_to_scenarios import *
from policy_translation.inputs.capture_to_scenarios import main

if __name__ == "__main__":
    raise SystemExit(main())
