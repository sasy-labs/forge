"""Measure a generated policy against labeled captured decision points.

The behavioral arm of the generation pipeline. The blind Datalog
authors write rules over a **projected universe** — a vocabulary
derived from the substrate the engine actually compiles, so its
relations ARE the engine's relations. A rule written against it is
therefore deployable as it stands, and the question this script
answers is the one the offline gates cannot: on the decisions a real
run produced, does the generated rule decide what the labels say?

What it does, in order:

1. **Composes.** A generated rule is a fragment: it uses relations
   `common_policy.dl` declares but does not declare them itself. If
   the policy names no ``#include``, the common policy's text is put
   in front of it and the result is written to a scratch directory —
   textual composition, valid precisely because the universe's
   relations are the engine's. A policy that already includes
   something is run as it is.
2. **Runs the convergence gate.** `policy_convergence.py` is imported
   and invoked, not reimplemented: the same classification, the same
   advisory rules, the same report. The gate writes into this run's
   own scratch directory, and its report is copied out to the
   deterministic path only once this run has built its own report from
   it. A gate that refused its inputs wrote nothing there, so nothing
   can be read in its place — in particular not the file an earlier
   run left at that same path.
3. **Reports.** The convergence report is carried through whole,
   `oracle_accuracy` included, under a fidelity summary that says how
   many decision points were measurable and on how many the policy
   agreed, and an `evidence` block naming the policy, the digest of
   the exact program bytes that ran, and the scenarios file the gate
   says it ran on.

What "fidelity" counts. A scenario is **measured** when it carries a
label (or author-written `expected` assertions) and evaluation
succeeded. It **agrees** when the policy's decision matches. Scenarios
with no label are counted separately and left out of the ratio — they
are not agreement, and folding them in either way would move the
number for a reason that has nothing to do with the policy.

What it does not establish: the recording's coverage. A rule that is
wrong on a branch no capture exercised scores the same as one that is
right there. See `README.md`, "What convergence does and does not
establish".

Usage:
    uv run python experiments/translation/generated_policy_fidelity.py \\
        --policy path/to/generated.dl \\
        --scenarios path/to/labeled_tests.json \\
        [--policy-nl path/to/policy.md] [--report path/to/out.json] \\
        [--min-fidelity 0.9]

Exit code 0 = the measurement completed; 1 = it did not (a scenario
failed to evaluate, or the policy or scenarios could not be read), or
`--min-fidelity` was given and not met.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from policy_translation.evaluation import convergence as policy_convergence
from policy_translation.evaluation.harness import discover_companion_functors
from policy_translation.souffle_text import DEFAULT_COMMON_POLICY

# A line that opens with `#include`, which is what makes a policy a
# whole program rather than a fragment. Matching the directive rather
# than the file it names keeps a policy that includes something OTHER
# than the common policy out of the composition path, where prepending
# a second copy of the declarations would break the compile.
INCLUDE = re.compile(r"^[ \t]*#include\b", re.MULTILINE)


class CompositionError(Exception):
    """The policy could not be made into a program to run."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Publish complete text or leave the destination absent/unchanged."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy to a sibling temporary file before publishing by rename."""
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def needs_composition(policy_text: str) -> bool:
    """Whether this policy is a fragment that needs the common policy."""
    return INCLUDE.search(policy_text) is None


def compose(policy: Path, common: Path, into: Path) -> Path:
    """Write the common policy followed by the generated rule.

    Returns the path of the composed program. Any companion functor
    source the original policy would have linked is copied in beside
    it as ``functors.cpp``, which is the first candidate the harness
    looks for, so a generated rule calling a policy-local primitive
    still links the same C++ it would have linked in place.
    """
    if not common.exists():
        raise CompositionError(f"common policy not found: {common}")
    composed = into / f"{policy.stem}_composed.dl"
    composed.write_text(f"{common.read_text()}\n// ── generated rule ──\n{policy.read_text()}")
    companion = discover_companion_functors(policy)
    if companion is not None:
        shutil.copyfile(companion, into / "functors.cpp")
    return composed


def run_convergence(
    policy: Path,
    scenarios: Path,
    report: Path,
    policy_nl: Path | None,
) -> dict[str, Any]:
    """Run the convergence gate and hand back the report it wrote.

    `policy_convergence.main()` is the loop this measurement is built
    on — same classification, same advisory rules, same report shape —
    so it is invoked rather than copied. It reads `sys.argv` and ends
    in `sys.exit`, so both are handled here.

    The exit status alone is not the answer: "did not converge" is a
    finding this report states rather than an error, and the gate ends
    in `sys.exit(1)` for it after writing a complete report. What
    separates that from a refusal is the report itself. The gate
    refuses its inputs — an unresolvable `#include`, a scenarios file
    with no scenarios — by exiting before it writes anything, so an
    absent report is a refusal and is raised as one, with the exit
    status named.

    `report` must be a path this run owns and nothing has written yet
    (`measure` gives it one inside the run's scratch directory).
    Reading a deterministic path an earlier run also wrote would let a
    refusal here be reported as this run's measurement. The two names
    the gate records are checked against the two this run passed, so a
    report that measured something else is refused rather than
    summarized.
    """
    argv = [
        str(Path(policy_convergence.__file__)),
        "--policy",
        str(policy),
        "--scenarios",
        str(scenarios),
        "--report",
        str(report),
    ]
    if policy_nl is not None:
        argv += ["--policy-nl", str(policy_nl)]
    saved = sys.argv
    sys.argv = argv
    status: Any = 0
    try:
        policy_convergence.main()
    except SystemExit as e:
        status = e.code
    finally:
        sys.argv = saved
    if not report.exists():
        raise CompositionError(
            f"the convergence gate wrote no report at {report} (it exited "
            f"{status!r}); it refused the inputs before it ran"
        )
    parsed: dict[str, Any] = json.loads(report.read_text())
    for field, expected in (
        ("policy", str(policy)),
        ("scenarios_file", str(scenarios)),
    ):
        found = parsed.get(field)
        if found != expected:
            raise CompositionError(
                f"the convergence report names {field} {found!r}, not {expected!r}; it is not this run's measurement"
            )
    return parsed


def fidelity_summary(convergence: dict[str, Any]) -> dict[str, Any]:
    """How many decisions were measurable, and how many agreed."""
    counts = convergence.get("summary", {})
    total = int(counts.get("total", 0))
    agree = int(counts.get("match", 0))
    mismatch = int(counts.get("mismatch", 0))
    disputed = int(counts.get("disputed_advisory", 0))
    unlabeled = int(counts.get("unlabeled", 0))
    errors = int(counts.get("errors", 0))
    measured = total - unlabeled - errors
    return {
        "agree": agree,
        "behavioral_fidelity": (round(agree / measured, 4) if measured > 0 else None),
        "converged": bool(convergence.get("converged", False)),
        "disagree": mismatch + disputed,
        "disagree_advisory": disputed,
        "disagree_blocking": mismatch,
        "errors": errors,
        "measured": measured,
        "scenarios": total,
        "unlabeled": unlabeled,
    }


def measure(
    policy: Path,
    scenarios: Path,
    report_path: Path,
    common: Path,
    policy_nl: Path | None = None,
) -> dict[str, Any]:
    """Compose, run the convergence gate, and build the report.

    Both output paths are derived from `report_path`, so an earlier
    run's files sit exactly where this run's go. They are removed
    before the gate runs and written only from what this run measured:
    a run that fails leaves no file at either path, rather than
    leaving the previous run's numbers standing under this run's name.

    That holds because neither path is written until this run has a
    report to put there. The gate writes into a scratch directory, its
    report is copied out only once the fidelity report has been built
    from it, and a failure to write that report takes the copy away
    again.
    """
    policy_text = policy.read_text()
    composing = needs_composition(policy_text)
    convergence_path = report_path.parent / f"{report_path.stem}.convergence.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.unlink(missing_ok=True)
    convergence_path.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="generated_fidelity_") as scratch:
        program = compose(policy, common, Path(scratch)) if composing else policy
        # The gate writes here, where only this run can have written:
        # a refusal leaves the path empty, and there is no earlier
        # file to mistake for this run's measurement.
        fresh = Path(scratch) / "convergence.json"
        convergence = run_convergence(program, scenarios, fresh, policy_nl)
        composition = {
            "composed": composing,
            "common_policy": str(common) if composing else None,
            "common_policy_sha256": sha256(common) if composing else None,
            "program_sha256": sha256(program),
            "reason": (
                "the policy names no #include, so the common policy's declarations were put in front of it"
                if composing
                else "the policy names an #include, so it was run as it is"
            ),
        }

        report = {
            "schemaVersion": "generated-policy-fidelity-v1",
            "policy": str(policy),
            "policy_sha256": sha256(policy),
            "scenarios_file": str(scenarios),
            "scenarios_sha256": sha256(scenarios),
            "composition": composition,
            # What the number was computed over. The scenarios file is
            # the name the gate recorded, checked against the one this
            # run passed it; the program the gate ran is named by the
            # policy it came from and the digest of the exact bytes,
            # since a composed program lives in a scratch directory
            # that is gone by the time anyone reads this.
            "evidence": {
                "generated": convergence.get("generated"),
                "policy": str(policy),
                "program_sha256": composition["program_sha256"],
                "scenarios_file": convergence.get("scenarios_file"),
            },
            "fidelity": fidelity_summary(convergence),
            "convergence": convergence,
            "convergence_report": str(convergence_path),
        }
        if "oracle_accuracy" in convergence:
            report["oracle_accuracy"] = convergence["oracle_accuracy"]
        # The gate's report is copied out only now, with this run's
        # report ready to sit beside it, and taken away again if that
        # report cannot be written. Either both files are this run's or
        # neither is there.
        atomic_copy(fresh, convergence_path)
        try:
            atomic_write_text(report_path, json.dumps(report, indent=2) + "\n")
        except BaseException:
            report_path.unlink(missing_ok=True)
            convergence_path.unlink(missing_ok=True)
            raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure a generated policy against labeled scenarios",
    )
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--scenarios", required=True, type=Path)
    parser.add_argument(
        "--policy-nl",
        type=Path,
        default=None,
        help="Natural-language policy the judge was shown; passed "
        "through to the convergence gate, which recomputes each "
        "oracle label's prompt signature against it",
    )
    parser.add_argument(
        "--common-policy",
        type=Path,
        default=DEFAULT_COMMON_POLICY,
        help=f"The declarations a fragment is composed with (default: {DEFAULT_COMMON_POLICY})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Report JSON path (default: <scenarios>.fidelity.json)",
    )
    parser.add_argument(
        "--min-fidelity",
        type=float,
        default=None,
        help="Fail unless the measured fidelity is at least this. "
        "Without it the exit code says only whether the measurement "
        "completed",
    )
    args = parser.parse_args()

    for path, what in [(args.policy, "policy"), (args.scenarios, "scenarios")]:
        if not path.exists():
            print(f"ERROR: {what} file not found: {path}")
            sys.exit(1)
    if args.policy_nl is not None and not args.policy_nl.exists():
        print(f"ERROR: policy-nl file not found: {args.policy_nl}")
        sys.exit(1)

    out = args.report or args.scenarios.with_suffix(".fidelity.json")
    try:
        report = measure(
            args.policy,
            args.scenarios,
            out,
            args.common_policy,
            args.policy_nl,
        )
    except CompositionError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    fidelity = report["fidelity"]
    composition = report["composition"]
    print(f"Composed: {'yes' if composition['composed'] else 'no'} — {composition['reason']}")
    measured = fidelity["behavioral_fidelity"]
    print(
        f"Behavioral fidelity: "
        f"{'unmeasured' if measured is None else measured} "
        f"({fidelity['agree']}/{fidelity['measured']} measured decisions "
        f"agree, {fidelity['unlabeled']} unlabeled, "
        f"{fidelity['errors']} errors) → {out}"
    )
    # The convergence gate has already printed its own line, including
    # whether it converged and its oracle-accuracy count, so neither is
    # repeated here; both are in the report this line names.

    if fidelity["errors"] > 0:
        sys.exit(1)
    if args.min_fidelity is not None and (measured is None or measured < args.min_fidelity):
        print(f"FAIL: fidelity {measured} is below the required {args.min_fidelity}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
