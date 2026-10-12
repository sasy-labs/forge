"""Diff a candidate policy's decisions against labeled scenarios.

The convergence step of the translation loop: runs the offline
harness (Soufflé interpreted mode) over every scenario and compares
the policy's decision against the scenario's label, classifying
each as:

  match              decision == label
  mismatch           decision != TRUSTED label — blocks convergence
  disputed-advisory  decision != ADVISORY label — surfaced for
                     review, non-blocking unless --fail-on-disputed;
                     must be ACKED to converge
  unlabeled          no label and no expected assertions
  error              Soufflé evaluation failed

A label is ADVISORY when its recorded provenance says the judge was
working outside its reliable regime: `context_mode: "graph"` (graph-
only rendering measurably degrades judges) or `policy_class: "taint"`.
Everything else is trusted, including a label that carries
`oracle_weak: true` without such provenance — that combination is
reported under `warnings` and still blocks. Author and ground-truth
labels are trusted whatever they record.

`warnings` also names any advisory label that records no
`prompt_sig`. The labeler writes a prompt signature — a hash of the
exact question the judge was asked — on every label it produces, so
advisory provenance with no signature at all was not written by a
labeler run. That signal catches a hand-written label; it does NOT
catch an edit to `context_mode` or `policy_class` on a label the
labeler really wrote, because the signature it left survives such an
edit untouched.

Catching that needs the signature recomputed, which needs the
natural-language policy the judge read: pass `--policy-nl`. Each
oracle label's signature is then rebuilt from the scenario and the
label's own recorded `context_mode`, through the same prompt builder
the labeler used, and a label whose recorded provenance does not
reproduce its signature is named under `warnings`. `policy_class` is
not part of the judge's prompt, so no signature can speak to it —
only the measurement harness's snapshot covers that field. The
report says what it found; it does not act on it.

Converged = zero mismatches AND every disputed entry carries an ack.
To ack a disputed scenario, add to the scenario in the scenarios
file:

    "ack": {"resolution": "label-wrong" | "policy-correct-by-design",
            "note": "why"}

The report JSON gives the translating model everything needed to fix
the policy: which deny rules fired (DenialReason), the label's
reasoning, and the rendered linear context — exactly what the oracle
saw when it judged.

Usage:
    uv run python experiments/translation/policy_convergence.py \\
        --policy path/to/policy.dl \\
        --scenarios path/to/labeled_tests.json \\
        [--policy-nl path/to/policy.md] \\
        [--report path/to/report.json] [--fail-on-disputed]

Exit code 0 = converged, 1 = not converged (or errors).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

from policy_translation.bundle.environment_conformance import producer_value
from policy_translation.evaluation.harness import (
    check_scenario,
    policy_uses_llm_check,
    prepare_policy,
    run_one_scenario,
    three_valued_verdict,
)
from policy_translation.evaluation.labels import (
    is_advisory,
    label_source,
    oracle_accuracy,
    outcome_authorized,
    provenance_warning,
    verdict_outcome,
)
from policy_translation.inputs.capture import (
    CONTEXT_HEADERS,
    blind_scenario_fixture_identifiers,
    build_user_prompt,
    render_scenario_context,
    scenario_actor,
)

CLAUSE_ID = re.compile(r"\b[A-Z][A-Z0-9_]*-\d+(?:\.\d+)+\b")


def disagreement_localization(
    scenario: dict[str, Any],
    verdict: dict[str, Any],
    denial_reasons: list[tuple[int, str, str, str]],
) -> dict[str, Any]:
    """Route a mismatch to environment, clause, or composition review."""
    environment = []
    unknown = []
    for row in verdict.get("predicate_labels", []):
        if not isinstance(row, dict) or not isinstance(row.get("predicate"), str):
            continue
        actual = producer_value(scenario, row["predicate"])
        expected = row.get("value")
        item = {
            "predicate": row["predicate"],
            "judge": expected,
            "producer": actual,
            "evidence": row.get("evidence", []),
        }
        if actual in {"true", "false"} and expected in {"true", "false"}:
            if actual != expected:
                environment.append(item)
        else:
            unknown.append(item)
    candidate_clauses = sorted(
        {
            match.group(0)
            for _, _, reason, suggestion in denial_reasons
            for match in CLAUSE_ID.finditer(f"{reason} {suggestion}")
        }
    )
    judge_clauses = sorted(set(verdict.get("applicable_clauses", [])))
    label = scenario.get("label") or {}
    observations_withheld = (
        isinstance(label, dict)
        and label.get("label_schema") == "evidence"
        and label.get("evidence_observations") == "none"
    )
    unresolved_environment = [
        row for row in unknown if row["judge"] in {"true", "false"} and row["producer"] not in {"true", "false"}
    ]
    if observations_withheld:
        route = "environment-input-insufficient"
    elif environment:
        route = "environment"
    elif unresolved_environment:
        route = "environment-input-unresolved"
    elif candidate_clauses != judge_clauses and (candidate_clauses or judge_clauses):
        route = "clause-applicability"
    else:
        route = "composition-or-source"
    return {
        "route": route,
        "environment_predicate_disagreements": environment,
        "unknown_predicates": unknown,
        "unresolved_environment_predicates": unresolved_environment,
        "candidate_fired_clauses": candidate_clauses,
        "judge_applicable_clauses": judge_clauses,
        "clause_symmetric_difference": sorted(set(candidate_clauses) ^ set(judge_clauses)),
        "producer_observations_withheld": observations_withheld,
        "automatic_policy_repair": False,
    }


class Classification(NamedTuple):
    """What one scenario's comparison came to.

    A tuple so existing three-way unpacking still reads naturally;
    `warning` carries a provenance complaint that belongs in the
    report even when the scenario otherwise matches.
    """

    kind: str
    label_authorized: bool | None
    source: str
    warning: str | None = None


def classify(
    scenario: dict[str, Any],
    harness_authorized: bool,
    full_check_failures: list[str] | None,
    harness_verdict: str | None = None,
) -> Classification:
    """Compare one scenario's harness decision against its label."""
    label = scenario.get("label") or {}
    if label:
        verdict = label.get("verdict") or {}
        ref_outcome = verdict_outcome(verdict, strict=True)
        ref = outcome_authorized(ref_outcome) if ref_outcome is not None else None
        source = label_source(label)
        # Trust is a property of what the label RECORDS about how it
        # was produced, not of a flag on it. A translator that sets
        # `oracle_weak: true` on a linear-context label demotes
        # nothing; the disagreement still blocks, and the warning
        # names the scenario.
        advisory = is_advisory(label)
        warning = provenance_warning(str(scenario.get("name", "unnamed")), label)
        if ref_outcome is None:
            return Classification("unlabeled", None, source, warning)
        actual_outcome = harness_verdict or ("ALLOW" if harness_authorized else "DENY")
        agrees = ref_outcome == actual_outcome if "outcome" in verdict else ref == harness_authorized
        if agrees and not full_check_failures:
            return Classification("match", ref, source, warning)
        return Classification(
            "disputed-advisory" if advisory else "mismatch",
            ref,
            source,
            warning,
        )
    # No label — fall back to author-written expected assertions.
    expected = scenario.get("expected") or {}
    if expected:
        if full_check_failures:
            return Classification("mismatch", expected.get("authorized"), "author")
        return Classification("match", expected.get("authorized"), "author")
    return Classification("unlabeled", None, "")


def prompt_signature_warning(scenario: dict[str, Any], policy_nl: str) -> str | None:
    """Rebuild an oracle label's prompt signature, or None if it fits.

    `oracle_label.py` records `prompt_sig`: the first twelve hex
    digits of the sha256 of the exact prompt it sent the judge. That
    prompt is built from the natural-language policy, the context
    RENDERED IN THE MODE THE LABEL RECORDS, the tool call and its
    arguments, and the acting agent. Rebuilding it here from the same
    inputs turns the recorded `context_mode` into a claim that can be
    checked: edit it from `linear` to `graph` to make a blocking
    mismatch into an ackable dispute, and the rendering changes, so
    the signature no longer comes out the same.

    What it cannot check:

      * `policy_class`, which never reaches the judge's prompt;
      * a label with no signature at all — `provenance_warning`
        already reports that one;
      * anything, unless `--policy-nl` names the same policy text the
        judge read. A different spec changes every signature, so the
        warning then says the provenance does not reproduce, which is
        true but not about the label.
    """
    label = scenario.get("label") or {}
    name = str(scenario.get("name", "unnamed"))
    if label_source(label) != "oracle":
        return None
    recorded = label.get("prompt_sig")
    if not isinstance(recorded, str) or not recorded:
        return None
    mode = label.get("context_mode")
    if mode not in CONTEXT_HEADERS:
        return (
            f"provenance does not reproduce prompt_sig: {name} "
            f"(context_mode={mode!r} is not a rendering the labeler "
            f"can produce, so its recorded signature "
            f"{recorded} cannot have come from it)"
        )
    prompt_scenario = (
        blind_scenario_fixture_identifiers(scenario) if label.get("blind_scenario_identifiers", False) else scenario
    )
    action = prompt_scenario.get("action") or {}
    action_args = action.get("args", "{}")
    try:
        prompt = build_user_prompt(
            policy_nl,
            mode,
            render_scenario_context(prompt_scenario, mode),
            action.get("fn_name", ""),
            action_args,
            scenario_actor(prompt_scenario),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        return f"prompt_sig could not be recomputed for {name}: {e}"
    rebuilt = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    # V2 extends the base question with a schema instruction and, in evidence
    # mode, a declared vocabulary. The base signature still binds the policy,
    # rendering, action and actor here; vocabulary_sig binds the ABI in the
    # snapshotted label. Full prompt reproduction additionally needs the ABI
    # file used by the labeler.
    expected = label.get("base_prompt_sig", recorded)
    if rebuilt == expected:
        return None
    return (
        f"provenance does not reproduce prompt_sig: {name} "
        f"(recorded {expected}, recomputed {rebuilt} from "
        f"context_mode={mode!r} and this scenario) — either the "
        f"recorded context mode is not the one the judge was shown, "
        f"or --policy-nl is not the policy text it read"
    )


def has_ack(scenario: dict[str, Any]) -> bool:
    ack = scenario.get("ack") or (scenario.get("label") or {}).get("ack")
    return bool(
        isinstance(ack, dict)
        and ack.get("resolution")
        in {
            "label-wrong",
            "policy-correct-by-design",
        }
        and isinstance(ack.get("note"), str)
        and ack["note"].strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diff policy decisions against labeled scenarios",
    )
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--scenarios", required=True, type=Path)
    parser.add_argument(
        "--policy-nl",
        type=Path,
        default=None,
        help="Natural-language policy the judge was shown. Given it, "
        "every oracle label's prompt signature is recomputed from the "
        "scenario and the label's own recorded context_mode, and a "
        "mismatch is reported under warnings",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Report JSON path (default: <scenarios>.convergence.json)",
    )
    parser.add_argument(
        "--fail-on-disputed",
        action="store_true",
        help="Strict mode: fail if ANY disputed-advisory entry "
        "remains, even acked ones (un-acked disputes already block "
        "convergence by default)",
    )
    args = parser.parse_args()

    checked = [(args.policy, "policy"), (args.scenarios, "scenarios")]
    if args.policy_nl is not None:
        checked.append((args.policy_nl, "policy-nl"))
    for p, what in checked:
        if not p.exists():
            print(f"ERROR: {what} file not found: {p}")
            sys.exit(1)
    policy_nl = args.policy_nl.read_text() if args.policy_nl is not None else None

    data = json.loads(args.scenarios.read_text())
    scenarios = data.get("scenarios", [])
    if not scenarios:
        print("ERROR: no scenarios in file")
        sys.exit(1)

    desugared, functor_lib = prepare_policy(args.policy)
    llm_check = policy_uses_llm_check(desugared)
    llm_stub = "cache" if os.environ.get("SASY_LLM_STUB_FILE") else "default-0"

    counts = {
        "total": len(scenarios),
        "match": 0,
        "mismatch": 0,
        "disputed_advisory": 0,
        "unlabeled": 0,
        "errors": 0,
    }
    mismatches: list[dict[str, Any]] = []
    disputed: list[dict[str, Any]] = []
    unlabeled_names: list[str] = []
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    unacked_disputes = 0
    roles_unknown_count = 0

    for scenario in scenarios:
        name = scenario.get("name", "unnamed")
        prov = scenario.get("provenance") or {}
        if prov.get("roles_unknown"):
            roles_unknown_count += 1

        # Before the policy is run, so an evaluation error further
        # down does not swallow a provenance complaint.
        if policy_nl is not None:
            sig_warning = prompt_signature_warning(scenario, policy_nl)
            if sig_warning:
                warnings.append(sig_warning)

        try:
            results = run_one_scenario(desugared, scenario, functor_lib)
        except RuntimeError as e:
            counts["errors"] += 1
            errors.append({"name": name, "error": str(e)})
            continue

        harness_authorized = 0 in results.authorized
        harness_verdict = three_valued_verdict(results)
        # Author-written expected blocks get the full field
        # comparison (transforms, deny_unauthorized, ...).
        full_failures: list[str] | None = None
        if not scenario.get("label") and scenario.get("expected") or scenario.get("label") and scenario.get("expected"):
            _, full_failures = check_scenario(scenario, results)

        try:
            cls, label_auth, source, warning = classify(
                scenario,
                harness_authorized,
                full_failures,
                harness_verdict=harness_verdict,
            )
        except ValueError as e:
            counts["errors"] += 1
            errors.append({"name": name, "error": str(e)})
            continue

        # Recorded whether or not the scenario agrees: a flag that
        # claims a trust level the provenance does not support is
        # worth seeing even when the verdicts happen to match.
        if warning:
            warnings.append(warning)

        if cls == "match":
            counts["match"] += 1
            continue
        if cls == "unlabeled":
            counts["unlabeled"] += 1
            unlabeled_names.append(name)
            continue

        label = scenario.get("label") or {}
        verdict = label.get("verdict") or {}
        label_outcome = verdict_outcome(verdict)
        entry: dict[str, Any] = {
            "name": name,
            "clause": scenario.get("clause", "?"),
            "harness": {
                "verdict": harness_verdict,
                "authorized": harness_authorized,
                "fired_denials": [
                    {
                        "idx": i,
                        "kind": k,
                        "reason": r,
                        "suggestion": sg,
                    }
                    for i, k, r, sg in results.denial_reasons
                ],
                "transforms": results.apply_transform.get(0, []),
            },
            "label": {
                "verdict": label_outcome,
                "authorized": label_auth,
                "source": source,
                "advisory": cls == "disputed-advisory",
                "context_mode": label.get("context_mode", ""),
                "policy_class": label.get("policy_class", ""),
                "oracle_weak": label.get("oracle_weak", False),
                "reason": verdict.get("reason", ""),
                "suggested_fix": verdict.get("suggested_fix", ""),
                "applicable_clauses": verdict.get("applicable_clauses", []),
                "predicate_labels": verdict.get("predicate_labels", []),
                "missing_preconditions": verdict.get("missing_preconditions", []),
            },
            "live_decision": prov.get("live_decision"),
            "linear_context": render_scenario_context(scenario, "linear"),
            "provenance": {k: prov.get(k) for k in ("run_id", "tool_id", "capture_file") if k in prov},
            "localization": disagreement_localization(scenario, verdict, results.denial_reasons),
        }
        if full_failures:
            entry["failed_assertions"] = full_failures
        if llm_check:
            entry["llm_stub"] = llm_stub
            if llm_stub == "default-0":
                entry["llm_stub_suspect"] = True

        if cls == "mismatch":
            counts["mismatch"] += 1
            mismatches.append(entry)
        else:
            counts["disputed_advisory"] += 1
            entry["ack"] = scenario.get("ack") or (scenario.get("label") or {}).get("ack")
            if not has_ack(scenario):
                unacked_disputes += 1
            disputed.append(entry)

    # Unlabeled scenarios are not agreement. Without this, a file
    # labeled with `--limit 1` converges on the strength of one
    # decision and finalization writes a success marker for a policy
    # checked against a single point of the capture.
    converged = counts["mismatch"] == 0 and counts["errors"] == 0 and unacked_disputes == 0 and counts["unlabeled"] == 0

    report = {
        "policy": str(args.policy),
        "scenarios_file": str(args.scenarios),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": counts,
        "converged": converged,
        "unacked_disputes": unacked_disputes,
        "mismatches": mismatches,
        "disputed": disputed,
        "unlabeled": unlabeled_names,
        "errors": errors,
        "warnings": warnings,
    }
    accuracy = oracle_accuracy(scenarios)
    if accuracy["n_both"]:
        report["oracle_accuracy"] = accuracy
    if llm_check and llm_stub == "default-0":
        report["note_llm_check"] = (
            "Policy uses @llm_check_fn with no SASY_LLM_STUB_FILE — "
            "offline answers default to 0; mismatches may be stub "
            "artifacts."
        )
    if roles_unknown_count:
        report["note_roles"] = (
            f"{roles_unknown_count} scenarios carry no roles "
            "(captured traces can't see server-side identity) — "
            "role-gated clauses are not exercised by them."
        )

    out = args.report or args.scenarios.with_suffix(".convergence.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"Convergence: {counts['match']}/{counts['total']} match, "
        f"{counts['mismatch']} mismatch, "
        f"{counts['disputed_advisory']} disputed "
        f"({unacked_disputes} un-acked), "
        f"{counts['unlabeled']} unlabeled, "
        f"{counts['errors']} errors → {out}"
    )
    for w in warnings:
        print(f"  WARN: {w}")
    if accuracy["n_both"]:
        print(f"Oracle vs ground truth: {accuracy['n_agree']}/{accuracy['n_both']} agree")
    print(f"Converged: {converged}")

    # `converged` already blocks on un-acked disputes. The flag is a
    # stricter mode: refuse to pass while ANY oracle disagreement
    # remains, even an acked one (zero-disagreement gate).
    if not converged or (args.fail_on_disputed and counts["disputed_advisory"] > 0):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
