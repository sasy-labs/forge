#!/usr/bin/env python3
"""Import human ground-truth labels and score the oracle against them.

A GROUND-TRUTH LABEL is a decision a person made about one scenario:
this call should have been allowed, or it should not have been. It is
trusted unconditionally — `policy_convergence.py` blocks on any
disagreement with it — and it is the only thing in this loop that can
say whether the LLM judge (the "oracle") is right.

Without ground truth, agreement with the oracle measures whether the
generated Datalog matches the judge's reading of the English, and
nothing more. `report` turns imported ground truth into the missing
number: how often the judge agreed with a person.

Importing preserves the oracle's own verdict under `label.oracle`, so
the comparison survives the label being replaced.

Usage:
    uv run python experiments/translation/ground_truth_labels.py import \\
        --scenarios path/to/tests.json \\
        --labels path/to/ground_truth.json [--force]

    uv run python experiments/translation/ground_truth_labels.py report \\
        --scenarios path/to/tests.json [--json path/to/out.json]

The labels file is a JSON list. ``verdict`` is canonical and can be
``ALLOW``, ``ASK``, or ``DENY``; legacy Boolean ``authorized`` rows remain
accepted for migration:

    [{"name": "scenario-name", "verdict": "DENY",
      "reason": "cancellation outside the 24h window",
      "ref": "spec.md#cancellation"}]

`reason` and `ref` are optional; `ref` is where the decision came
from (a spec section, a ticket, a reviewer's initials).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from policy_translation.evaluation.labels import (
    VERDICTS,
    label_source,
    oracle_accuracy,
    outcome_authorized,
)

GROUND_TRUTH = "ground-truth"


# ── Import ────────────────────────────────────────────────────


def parse_entries(raw: Any) -> list[dict[str, Any]]:
    """Validate the labels file without coercing anything.

    `bool("false")` is true in Python, so a string in the authorized
    field would silently reverse a human's denial — the one label in
    the system nothing else double-checks.
    """
    if not isinstance(raw, list):
        raise ValueError("labels file must be a JSON list of objects")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"labels[{i}] is not an object")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"labels[{i}] has no name")
        explicit = item.get("verdict", item.get("outcome"))
        legacy = item.get("authorized")
        if explicit is None:
            if type(legacy) is not bool:
                raise ValueError(
                    f"labels[{i}] ({name}): verdict must be ALLOW, ASK, "
                    "or DENY (legacy authorized must be true or false)"
                )
            explicit = "ALLOW" if legacy else "DENY"
        if explicit not in VERDICTS:
            raise ValueError(f"labels[{i}] ({name}): verdict must be ALLOW, ASK, or DENY")
        projected = outcome_authorized(str(explicit))
        if legacy is not None:
            if type(legacy) is not bool:
                raise ValueError(f"labels[{i}] ({name}): authorized must be true or false")
            if projected is None or legacy != projected:
                raise ValueError(f"labels[{i}] ({name}): authorized conflicts with verdict")
        for field in ("reason", "ref"):
            if field in item and not isinstance(item[field], str):
                raise ValueError(f"labels[{i}] ({name}): {field} must be a string")
        if name in seen:
            raise ValueError(f"labels file names {name!r} twice")
        seen.add(name)
        entries.append({**item, "verdict": explicit})
    return entries


def preserved_oracle(existing: dict[str, Any]) -> dict[str, Any] | None:
    """The oracle record to carry forward under `label.oracle`.

    Either the oracle label being replaced, or the block a previous
    import already preserved — re-importing must not erase the
    judge's verdict and with it the accuracy number.

    "Is this an oracle label" is asked through `label_source`, the
    one place that answers it, which reads a label with no `source`
    field as an oracle label. Matching on the literal string instead
    dropped exactly those: an unmarked judge verdict was overwritten
    with no `oracle` block kept, so the scenario silently left the
    accuracy report.
    """
    if existing and label_source(existing) == "oracle":
        return {
            "verdict": existing.get("verdict") or {},
            "schema": existing.get("schema", ""),
            "label_schema": existing.get("label_schema", ""),
            "model": existing.get("model", ""),
            "context_mode": existing.get("context_mode", ""),
            "policy_class": existing.get("policy_class", ""),
            "base_prompt_sig": existing.get("base_prompt_sig", ""),
            "prompt_sig": existing.get("prompt_sig", ""),
            "vocabulary_sig": existing.get("vocabulary_sig", ""),
        }
    prior = existing.get("oracle")
    return prior if isinstance(prior, dict) else None


def import_labels(
    data: dict[str, Any],
    entries: list[dict[str, Any]],
    force: bool = False,
) -> int:
    """Apply ground-truth entries to a loaded scenarios file.

    All-or-nothing: every name is checked before anything is written,
    so a typo in one entry cannot leave the file half-imported.
    """
    # Public callers use this function directly as well as through the CLI.
    # Normalize them here so the safety validation cannot be bypassed.
    entries = parse_entries(entries)
    by_name: dict[str, list[dict[str, Any]]] = {}
    for scenario in data.get("scenarios", []):
        if isinstance(scenario, dict) and isinstance(scenario.get("name"), str):
            by_name.setdefault(scenario["name"], []).append(scenario)

    unknown = [e["name"] for e in entries if e["name"] not in by_name]
    if unknown:
        raise ValueError(f"no scenario named {sorted(unknown)} in the scenarios file")
    ambiguous = [e["name"] for e in entries if len(by_name[e["name"]]) > 1]
    if ambiguous:
        raise ValueError(
            f"scenario name(s) {sorted(set(ambiguous))} appear more than once; ground truth must name one scenario"
        )
    if not force:
        taken = [e["name"] for e in entries if (by_name[e["name"]][0].get("label") or {}).get("source") == GROUND_TRUTH]
        if taken:
            raise ValueError(
                f"{sorted(set(taken))} already carry a ground-truth label; pass --force to replace a human decision"
            )

    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for entry in entries:
        scenario = by_name[entry["name"]][0]
        existing = scenario.get("label") or {}
        label: dict[str, Any] = {
            "verdict": {
                "outcome": entry["verdict"],
                "reason": entry.get("reason", ""),
                "suggested_fix": "",
            },
            "source": GROUND_TRUTH,
            "ts": stamp,
        }
        if entry.get("ref"):
            label["ref"] = entry["ref"]
        authorized = outcome_authorized(entry["verdict"])
        if authorized is not None:
            # Kept for consumers of pre-v2 human-label files. Outcome is the
            # canonical field and ASK intentionally has no boolean projection.
            label["verdict"]["authorized"] = authorized
        oracle = preserved_oracle(existing)
        if oracle:
            label["oracle"] = oracle
        scenario["label"] = label
        # Ground truth is trusted, so it drives the plain harness too.
        # Only the authorized field is touched: an author-written
        # `expected` may also assert transforms or denial fields, and
        # replacing the whole block would drop those assertions.
        expected = scenario.get("expected")
        if not isinstance(expected, dict):
            expected = {}
        expected["verdict"] = entry["verdict"]
        if authorized is None:
            expected.pop("authorized", None)
        else:
            expected["authorized"] = authorized
        scenario["expected"] = expected
    return len(entries)


def cmd_import(args: argparse.Namespace) -> int:
    data = json.loads(args.scenarios.read_text())
    if not isinstance(data, dict):
        print("ERROR: scenarios file must be a JSON object")
        return 1
    try:
        entries = parse_entries(json.loads(args.labels.read_text()))
        count = import_labels(data, entries, force=args.force)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    args.scenarios.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Imported {count} ground-truth label(s) → {args.scenarios}")
    return 0


# ── Report ────────────────────────────────────────────────────


def cmd_report(args: argparse.Namespace) -> int:
    data = json.loads(args.scenarios.read_text())
    if not isinstance(data, dict):
        print("ERROR: scenarios file must be a JSON object")
        return 1
    stats = oracle_accuracy(data.get("scenarios", []))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(stats, indent=2) + "\n")

    if not stats["n_both"]:
        print(
            "No scenario carries both an oracle verdict and a "
            "ground-truth label — judge accuracy is unmeasured. "
            "Import ground truth first."
        )
        return 0
    accuracy = stats["accuracy"]
    print(f"Oracle vs ground truth: {stats['n_agree']}/{stats['n_both']} agree (accuracy {accuracy:.3f})")
    for row in stats["disagreements"]:
        print(f"  {row['name']}: oracle={row['oracle']} ground-truth={row['ground_truth']}")
    if args.json:
        print(f"  → {args.json}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import ground-truth labels and score the oracle",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    imp = sub.add_parser("import", help="apply human labels to a scenarios file")
    imp.add_argument("--scenarios", required=True, type=Path)
    imp.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="JSON list of {name, authorized, reason?, ref?}",
    )
    imp.add_argument(
        "--force",
        action="store_true",
        help="replace an existing ground-truth label",
    )
    imp.set_defaults(func=cmd_import, needs=("scenarios", "labels"))

    rep = sub.add_parser("report", help="oracle-vs-ground-truth agreement")
    rep.add_argument("--scenarios", required=True, type=Path)
    rep.add_argument("--json", type=Path, default=None)
    rep.set_defaults(func=cmd_report, needs=("scenarios",))

    args = parser.parse_args()
    for name in args.needs:
        path = getattr(args, name)
        if not path.exists():
            print(f"ERROR: {name} file not found: {path}")
            sys.exit(1)
    try:
        sys.exit(args.func(args))
    except json.JSONDecodeError as e:
        print(f"ERROR: malformed JSON: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
