#!/usr/bin/env python3
"""Finalize gate for a translated policy.

Bundles every verification the write-policy workflow requires into
one command, so non-interactive (`claude -p`) runs can't skip steps:

  1. Validate     — `make souffle-validate` (sugar + lint + Soufflé)
  2. Config sync  — ApplyTransform ids must exist in
                    config/transforms.json (FAIL); HasRole roles
                    should appear in config/auth_config.yaml (WARN)
  3. Scenarios    — policy_test_harness must pass (all scenarios)
  4. Convergence  — when labeled/captured scenarios exist (the
                    scenarios file carries labels, or --traces names
                    another labeled file), policy_convergence must
                    report converged (zero non-disputed mismatches,
                    all disputes acked)
  5. Round-trip   — WARN if the policy header lacks a
                    `// Round-trip audit: PASS` line

On success writes `.sasy-policy-run/finalized/<sha256(policy)>.ok`
— the marker the plugin's Stop hook checks before letting a session
that wrote a policy finish. Content-hash keying means any later
edit to the policy re-arms the gate.

Usage:
    make policy-finalize POLICY=path.dl SCENARIOS=tests.json \\
        [TRACES=labeled_captured.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Per-session when SASY_POLICY_RUN_DIR is set, so concurrent
# translation sessions in one checkout do not satisfy each
# other's finalize gate. Must match the hooks' resolution.
MARKER_DIR = Path(
    os.environ.get("SASY_POLICY_RUN_DIR", str(REPO_ROOT / ".sasy-policy-run"))
) / "finalized"


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True
    )
    return proc.returncode, (proc.stdout + proc.stderr)


def step_validate(policy: Path) -> list[str]:
    rc, out = _run(
        ["make", "souffle-validate", f"FILE={policy}"]
    )
    if rc != 0:
        return [f"validate failed:\n{out.strip()}"]
    print("  [1/5] validate: OK")
    return []


def step_config_sync(policy: Path) -> tuple[list[str], list[str]]:
    """Returns (failures, warnings)."""
    source = policy.read_text()
    failures: list[str] = []
    warnings: list[str] = []

    transforms_file = REPO_ROOT / "config" / "transforms.json"
    known_transforms: set[str] = set()
    if transforms_file.exists():
        known_transforms = set(
            json.loads(transforms_file.read_text())
            .get("transforms", {})
            .keys()
        )
    for tid in re.findall(
        r'ApplyTransform\s*\([^,]+,\s*"([^"]+)"', source
    ):
        if tid not in known_transforms:
            failures.append(
                f"transform id {tid!r} not in config/transforms.json"
            )

    auth_file = REPO_ROOT / "config" / "auth_config.yaml"
    auth_text = auth_file.read_text() if auth_file.exists() else ""
    for role in re.findall(r'HasRole\s*\(\s*"([^"]+)"', source):
        if role not in auth_text:
            warnings.append(
                f"role {role!r} not found in config/auth_config.yaml"
            )

    if not failures:
        print(
            "  [2/5] config-sync: OK"
            + (f" ({len(warnings)} warnings)" if warnings else "")
        )
    return failures, warnings


def step_scenarios(policy: Path, scenarios: Path) -> list[str]:
    rc, out = _run(
        [
            "python3",
            "scripts/policy_test_harness.py",
            "--policy", str(policy),
            "--scenarios", str(scenarios),
        ]
    )
    if rc != 0:
        return [f"scenario harness failed:\n{out.strip()[-2000:]}"]
    # The harness treats an empty scenario list as success (it warns and
    # returns 0), and _has_labels then skips convergence, so an empty
    # placeholder file would satisfy the whole gate without the policy
    # ever being evaluated against a single decision.
    try:
        count = len(json.loads(scenarios.read_text()).get("scenarios", []))
    except (json.JSONDecodeError, OSError) as e:
        return [f"could not read scenarios file {scenarios}: {e}"]
    if count == 0:
        return [
            f"{scenarios} contains no scenarios; finalization requires "
            f"at least one (>=1 positive and >=1 negative per clause)"
        ]
    print(f"  [3/5] scenarios: OK ({count})")
    return []


def _has_labels(path: Path) -> bool:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return any(
        s.get("label") for s in data.get("scenarios", [])
    )


def step_convergence(
    policy: Path, scenarios: Path, traces: Path | None
) -> tuple[list[str], bool]:
    """Returns (failures, ran)."""
    targets = [
        p
        for p in (traces, scenarios)
        if p is not None and _has_labels(p)
    ]
    if not targets:
        print(
            "  [4/5] convergence: skipped (no labeled scenarios "
            "— offline-only workflow)"
        )
        return [], False
    failures = []
    for target in targets:
        rc, out = _run(
            [
                "python3",
                "experiments/translation/policy_convergence.py",
                "--policy", str(policy),
                "--scenarios", str(target),
            ]
        )
        if rc != 0:
            failures.append(
                f"convergence not reached on {target}:\n"
                f"{out.strip()[-2000:]}"
            )
    if not failures:
        print(f"  [4/5] convergence: OK ({len(targets)} file(s))")
    return failures, True


def step_round_trip(policy: Path) -> list[str]:
    header = "\n".join(policy.read_text().splitlines()[:30])
    if "Round-trip audit: PASS" in header:
        print("  [5/5] round-trip header: OK")
        return []
    return [
        "no '// Round-trip audit: PASS' line in the policy header "
        "(add it after performing the blind round-trip audit)"
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize a translated policy",
    )
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--scenarios", required=True, type=Path)
    parser.add_argument(
        "--traces",
        type=Path,
        default=None,
        help="Labeled captured-scenario file for convergence",
    )
    args = parser.parse_args()

    required = [(args.policy, "policy"), (args.scenarios, "scenarios")]
    # An explicit --traces path must exist. _has_labels swallows a read
    # error into False, so a typo would drop the requested convergence
    # target and still let finalize write a success marker — the gate
    # would report PASS precisely because its input was unreadable.
    if args.traces is not None:
        required.append((args.traces, "traces"))
    for p, what in required:
        if not p.exists():
            print(f"POLICY-FINALIZE FAIL {args.policy} "
                  f"({what} file not found: {p})")
            sys.exit(1)

    print(f"Finalizing {args.policy} ...")
    failures: list[str] = []
    warnings: list[str] = []

    failures += step_validate(args.policy)
    if not failures:
        f, w = step_config_sync(args.policy)
        failures += f
        warnings += w
    if not failures:
        failures += step_scenarios(args.policy, args.scenarios)
    converged_ran = False
    if not failures:
        f, converged_ran = step_convergence(
            args.policy, args.scenarios, args.traces
        )
        failures += f
    warnings += step_round_trip(args.policy)

    for w in warnings:
        print(f"  WARN: {w}")

    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        print(f"POLICY-FINALIZE FAIL {args.policy}")
        sys.exit(1)

    sha = hashlib.sha256(args.policy.read_bytes()).hexdigest()
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    marker = MARKER_DIR / f"{sha}.ok"
    marker.write_text(
        json.dumps(
            {
                "policy": str(args.policy),
                "scenarios": str(args.scenarios),
                "traces": str(args.traces) if args.traces else None,
                "convergence_ran": converged_ran,
                "warnings": warnings,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=1,
        )
        + "\n"
    )
    print(
        f"POLICY-FINALIZE PASS {args.policy} "
        f"scenarios={args.scenarios} "
        f"convergence={'yes' if converged_ran else 'n/a'} "
        f"warnings={len(warnings)}"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
