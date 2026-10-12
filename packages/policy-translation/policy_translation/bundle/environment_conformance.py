#!/usr/bin/env python3
"""Compare advisory judge predicate labels with executable fact producers.

This is the environment -> policy separation point.  The judge labels what a
trace appears to mean; deterministic producers project the replayable EDB.  A
mismatch is assigned to the environment boundary and is not repaired by
editing policy clauses.  Absence is UNKNOWN unless ``--closed-world`` is used
for a fixture known to enumerate the complete relation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MISSING = object()


def _metadata(scenario: dict[str, Any]) -> list[list[Any]]:
    return [row for row in scenario.get("action_metadata", []) if isinstance(row, list) and len(row) >= 3]


def _current_depends(scenario: dict[str, Any]) -> set[str]:
    """Mirror the graph ancestor relation consumed by HistoryFact."""
    graph = scenario.get("graph") or {}
    current = {str(node) for node in graph.get("current_nodes", []) if isinstance(node, str)}
    predecessors: dict[str, set[str]] = {}
    for edge in graph.get("edges", []):
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        source, destination = map(str, edge)
        predecessors.setdefault(destination, set()).add(source)
    reachable = set(current)
    pending = list(current)
    while pending:
        for source in predecessors.get(pending.pop(), set()):
            if source not in reachable:
                reachable.add(source)
                pending.append(source)
    return reachable


def _path_value(value: Any, path: list[str]) -> Any:
    current = value
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return MISSING
        current = current[part]
    return current


def _binding_value(scenario: dict[str, Any], binding: dict[str, Any]) -> str:
    extractor = binding["extractor"]
    value = _path_value(scenario, extractor["path"])
    closed_world = bool(binding["closed_world"])
    if extractor["kind"] == "json-present":
        if value is MISSING:
            return "false" if closed_world else "unknown"
        return "true"
    if extractor["kind"] == "json-nonempty":
        if value is MISSING:
            return "false" if closed_world else "unknown"
        if isinstance(value, (str, list, dict)):
            return "true" if len(value) > 0 else "false"
        return "unknown"
    if extractor["kind"] == "json-equals":
        if value is MISSING:
            return "false" if closed_world else "unknown"
        return "true" if value == extractor["value"] else "false"
    if value is MISSING or not isinstance(value, list):
        return "false" if closed_world else "unknown"
    matched = any(
        isinstance(row, list)
        and all(
            selector["position"] < len(row)
            and row[selector["position"]] == selector["value"]
            for selector in extractor["selectors"]
        )
        for row in value
    )
    return "true" if matched else ("false" if closed_world else "unknown")


def producer_value(
    scenario: dict[str, Any],
    predicate: str,
    *,
    closed_world: bool = False,
    bindings: dict[str, dict[str, Any]] | None = None,
) -> str:
    if bindings is not None:
        binding = bindings.get(predicate)
        return _binding_value(scenario, binding) if binding is not None else "unknown"
    rows = _metadata(scenario)
    if predicate.startswith("operation:"):
        wanted = predicate.split(":", 1)[1]
        values = {str(row[1]) for row in rows if row[0] == "operation"}
        return "true" if wanted in values else ("false" if values or closed_world else "unknown")
    if predicate.startswith("attest:"):
        atom = predicate.split(":", 1)[1]
        if "=" in atom:
            key, wanted = atom.split("=", 1)
            values = {str(row[2]) for row in rows if row[:2] == ["attest", key]}
            return "true" if wanted in values else ("false" if values or closed_world else "unknown")
        values = {str(row[2]).lower() for row in rows if row[:2] == ["attest", atom]}
        if "true" in values:
            return "true"
        if "false" in values:
            return "false"
        return "false" if closed_world else "unknown"
    if predicate.startswith("history:"):
        wanted = predicate.split(":", 1)[1]
        matching_nodes = {str(row[1]) for row in rows if row[0] == "history_fact" and str(row[2]) == wanted}
        if matching_nodes:
            return "true" if matching_nodes & _current_depends(scenario) else "false"
        return "false" if closed_world else "unknown"
    if predicate.startswith("prior_evidence:"):
        wanted = predicate.split(":", 1)[1]
        values = {str(row[1]) for row in rows if row[0] == "evidence"}
        # Presence alone cannot prove reachability and durable recording.
        return "unknown" if wanted in values else ("false" if closed_world else "unknown")
    if predicate.startswith("policy:") and predicate.endswith(":configured"):
        key = predicate[len("policy:") : -len(":configured")]
        configured = any(
            isinstance(row, list) and len(row) >= 1 and row[0] == key for row in scenario.get("policy_metadata", [])
        )
        return "true" if configured else ("false" if closed_world else "unknown")
    # Approval validity is a join over reachability, action binding, logging,
    # and role. It must come from the approval verifier, not metadata presence.
    return "unknown"


def conformance_report(
    data: dict[str, Any],
    *,
    closed_world: bool = False,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bindings = (
        {
            str(row["predicate"]): row
            for row in environment.get("predicate_observations", [])
            if isinstance(row, dict) and isinstance(row.get("predicate"), str)
        }
        if environment is not None
        else None
    )
    comparable = agree = 0
    true_positive = true_negative = false_positive = false_negative = 0
    mismatches: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    coverage: dict[str, dict[str, Any]] = {
        name: {
            "coverage_mode": str(binding.get("coverage_mode", "variable")),
            "labeled": 0,
            "comparable": 0,
            "true": 0,
            "false": 0,
            "unknown": 0,
            "not_applicable": 0,
        }
        for name, binding in sorted((bindings or {}).items())
    }
    for scenario in data.get("scenarios", []):
        if not isinstance(scenario, dict):
            continue
        label = scenario.get("label") or {}
        verdict = label.get("verdict") if isinstance(label, dict) else {}
        rows = verdict.get("predicate_labels", []) if isinstance(verdict, dict) else []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or not isinstance(row.get("predicate"), str):
                continue
            predicate = str(row["predicate"])
            if predicate in coverage:
                coverage[predicate]["labeled"] += 1
                label_value = row.get("value")
                if label_value == "true":
                    coverage[predicate]["true"] += 1
                elif label_value == "false":
                    coverage[predicate]["false"] += 1
                elif label_value == "not-applicable":
                    coverage[predicate]["not_applicable"] += 1
                else:
                    coverage[predicate]["unknown"] += 1
            actual = producer_value(
                scenario,
                predicate,
                closed_world=closed_world,
                bindings=bindings,
            )
            expected = row.get("value")
            result = {
                "scenario": scenario.get("name", "unnamed"),
                "predicate": row["predicate"],
                "judge": expected,
                "producer": actual,
                "evidence": row.get("evidence", []),
            }
            if actual not in {"true", "false"} or expected not in {"true", "false"}:
                unknown.append(result)
                continue
            if predicate in coverage:
                coverage[predicate]["comparable"] += 1
            comparable += 1
            if actual == expected:
                agree += 1
            if expected == "true" and actual == "true":
                true_positive += 1
            elif expected == "false" and actual == "false":
                true_negative += 1
            elif expected == "true" and actual == "false":
                false_positive += 1
            elif expected == "false" and actual == "true":
                false_negative += 1
            else:
                # The preceding cases are exhaustive for comparable Boolean
                # labels; only disagreements belong in the mismatch list.
                pass
            if actual != expected:
                mismatches.append(result)
    producer_true = true_positive + false_negative
    producer_false = true_negative + false_positive
    judge_true = true_positive + false_positive
    true_recall = true_positive / producer_true if producer_true else None
    true_specificity = true_negative / producer_false if producer_false else None
    return {
        "schema": "environment-conformance/v1",
        "environment_contract": environment.get("path") if environment else None,
        "declared_predicates": sorted(bindings) if bindings is not None else None,
        "closed_world": closed_world,
        "comparable": comparable,
        "agree": agree,
        "accuracy": round(agree / comparable, 4) if comparable else None,
        "confusion": {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "producer_true_recall": round(true_recall, 4) if true_recall is not None else None,
        "judge_true_precision": (round(true_positive / judge_true, 4) if judge_true else None),
        "balanced_accuracy": (
            round((true_recall + true_specificity) / 2, 4)
            if true_recall is not None and true_specificity is not None
            else None
        ),
        "mismatches": mismatches,
        "unknown": unknown,
        "predicate_coverage": coverage,
        "unlabeled_predicates": sorted(name for name, counts in coverage.items() if counts["labeled"] == 0),
        "policy_repair_authorized": False,
        "routing": (
            "Resolve mismatches in instrumentation/classifiers or labels; do not edit policy clauses at this boundary."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--closed-world", action="store_true")
    args = parser.parse_args()
    report = conformance_report(
        json.loads(args.scenarios.read_text()),
        closed_world=args.closed_world,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Environment conformance: {report['agree']}/{report['comparable']} "
        f"comparable labels agree; {len(report['unknown'])} unknown"
    )


if __name__ == "__main__":
    main()
