#!/usr/bin/env python3
"""Label trust, the evidence digest, and oracle-vs-ground-truth accuracy.

Three questions about a scenario live here, because four scripts have
to answer them the same way:

  1. Is this label TRUSTED (a disagreement blocks convergence) or
     ADVISORY (a disagreement is surfaced for review)? Answered from
     RECORDED PROVENANCE — the context the judge was shown and the
     policy class it judged under — never from a flag that anything
     downstream can flip. `oracle_label.py` also writes an
     `oracle_weak` boolean for readability; it is a summary of the
     provenance, not the source of truth. A label carrying
     `oracle_weak: true` without advisory provenance stays trusted
     and is reported as a warning.

  2. What evidence did a policy converge against? Answered by the
     SCENARIO SNAPSHOT: every scenario's name mapped to its SCENARIO
     IDENTITY — canonical JSON of the whole scenario with only the
     fields the convergence protocol asks a translator to write
     removed (an `ack`, and a label's prose, `oracle_weak` and
     timestamp). A label is only half of a decision point: the
     verdict answers a question posed by the action, the graph, the
     roles and the entity, so all of those are inside the identity
     too. `snapshot_problem` compares two snapshots and reports the
     first scenario that went missing or changed; `evidence_digest`
     hashes one snapshot into the single string the finalize marker
     records. A SNAPSHOT RECORD pairs those identities with a census
     taken at the same moment — how many of the scenarios carried a
     label — so a later check can ask whether this run was labeled
     without re-reading the file it is checking.

  3. How often does the oracle agree with a human? Answered by
     `oracle_accuracy`, over scenarios that carry both a ground-truth
     label and the oracle verdict it replaced.

WHAT THIS CANNOT DO. The translating agent that `run_variant_set.py`
launches runs as the SAME OPERATING-SYSTEM USER as the harness, with
permission prompts disabled, write access to the whole checkout, and
the ability to leave background processes behind after it exits.
Nothing on a disk that user can write is tamper-proof, and these
functions are themselves part of the TRUSTED BASE — the code the gate
believes without checking. That base is: `scripts/policy_finalize.py`,
`scripts/policy_test_harness.py`,
`experiments/translation/policy_convergence.py`, this file, the Makefile
targets they call, `sasy-services/souffle/sugar.py`, the functor
sources, the Python environment and the compiled-functor cache. An
agent that rewrites any of them defeats the gate, and no check here
would notice.

What the measurement harness does guarantee, within one invocation, is
narrower and is stated in `run_variant_set.verify_run`: it copies the
artifacts it is going to measure into a directory it created, hashes
and gates and grades the copies, and holds its record of the labels in
the parent process's memory. Real isolation would need a different
operating-system user or a sandbox; this harness does not provide one.

Stdlib-only, no imports from the other loop scripts, so
`scripts/policy_finalize.py` can import it without pulling in the
harness.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

# Sources that are trusted whatever else the label records: a human
# wrote them, or they came from a scripted run with a known outcome.
TRUSTED_SOURCES = ("author", "ground-truth")

# Judge regimes whose labels are advisory. Graph-only rendering
# measurably degrades judges, and taint-class policies need reasoning
# over graph topology whichever way the context is rendered.
ADVISORY_CONTEXT_MODES = ("graph",)
ADVISORY_POLICY_CLASSES = ("taint",)

# Canonical decision lattice used by Guard and plan-policy monitors. Older
# translation artifacts carry only ``authorized: bool``; readers accept those
# as ALLOW/DENY, while new artifacts preserve ASK instead of collapsing it.
VERDICTS = ("ALLOW", "ASK", "DENY")
FIX_DISPOSITIONS = ("none", "optional", "required-precondition", "repair")


def label_quarantine_reasons(label: Any) -> list[str]:
    """Return internal contradictions that make a decision label unusable.

    The prose is retained for audit, but an ALLOW label cannot calibrate or
    select a policy while simultaneously saying that a required precondition
    or repair is still outstanding.
    """

    if not isinstance(label, dict):
        return []
    verdict = label.get("verdict")
    outcome = verdict_outcome(verdict)
    if outcome != "ALLOW" or not isinstance(verdict, dict):
        return []
    reasons: list[str] = []
    missing = verdict.get("missing_preconditions", [])
    if isinstance(missing, list) and any(isinstance(item, str) and item.strip() for item in missing):
        reasons.append("ALLOW verdict declares missing_preconditions")
    disposition = verdict.get("suggested_fix_disposition")
    if disposition in {"required-precondition", "repair"}:
        reasons.append(f"ALLOW verdict marks suggested_fix as {disposition}")
    elif disposition is None:
        suggestion = str(verdict.get("suggested_fix", "")).strip()
        optional = re.match(r"(?i)^(none\b|no (?:change|action|fix)\b|optional\b)", suggestion)
        required = re.search(
            r"(?i)(?:\bmust\b|\brequired\b|\bneeds? to\b|\bmake sure\b|"
            r"\bensure\b|\bbefore (?:proceeding|retrying|executing)\b|"
            r"if (?:it has|this has) not already been (?:done|confirmed|provided))",
            suggestion,
        )
        if suggestion and not optional and required:
            reasons.append("ALLOW verdict has a required-sounding legacy suggested_fix")
    return reasons


def label_is_quarantined(label: Any) -> bool:
    return bool(
        isinstance(label, dict)
        and (
            label.get("quarantine", {}).get("status") == "quarantined"
            if isinstance(label.get("quarantine"), dict)
            else False
        )
    ) or bool(label_quarantine_reasons(label))


def scenario_label_tier(scenario: dict[str, Any]) -> str:
    """Classify evidence as hard truth, judge measurement, or regression."""

    label = scenario.get("label")
    if label_is_quarantined(label):
        return "quarantined"
    if isinstance(label, dict) and label:
        source = label_source(label)
        if source in TRUSTED_SOURCES or source in {
            "construction-ground-truth",
            "human-ground-truth",
            "spec-derived-ground-truth",
        }:
            return "hard"
        return "judge"
    expected = scenario.get("expected")
    if isinstance(expected, dict):
        provenance = scenario.get("provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        if provenance.get("source") == "capture" and provenance.get("label_authority") not in {
            "construction-ground-truth",
            "human-ground-truth",
            "spec-derived-ground-truth",
        }:
            return "regression"
        return "hard"
    return "unlabeled"


def verdict_outcome(verdict: Any, *, strict: bool = False) -> str | None:
    """Return ALLOW/ASK/DENY from a v2 or legacy verdict object.

    V2 stores ``outcome``. V1 stores ``authorized`` and therefore cannot
    represent ASK. No truthiness coercion is permitted: a JSON string such as
    ``"false"`` is an error in strict mode, not an accidental ALLOW.
    """
    if not isinstance(verdict, dict):
        if strict:
            raise ValueError("verdict must be a JSON object")
        return None
    if "outcome" in verdict:
        outcome = verdict.get("outcome")
        if outcome not in VERDICTS:
            if strict:
                raise ValueError("verdict outcome must be ALLOW, ASK, or DENY")
            return None
        if "authorized" in verdict:
            authorized = verdict.get("authorized")
            projected = outcome_authorized(str(outcome))
            if type(authorized) is not bool or projected is None or (authorized != projected):
                if strict:
                    raise ValueError("verdict authorized conflicts with outcome")
                return None
        return str(outcome)
    if "authorized" in verdict:
        authorized = verdict.get("authorized")
        if type(authorized) is not bool:
            if strict:
                raise ValueError("verdict authorized must be a boolean")
            return None
        return "ALLOW" if authorized else "DENY"
    return None


def outcome_authorized(outcome: str) -> bool | None:
    """Legacy projection, undefined for ASK rather than silently collapsing."""
    if outcome == "ALLOW":
        return True
    if outcome == "DENY":
        return False
    return None


# ── Trust ─────────────────────────────────────────────────────


def label_source(label: dict[str, Any]) -> str:
    """The recorded source; an unmarked label is an oracle label."""
    source = label.get("source")
    return source if isinstance(source, str) and source else "oracle"


def is_advisory(label: dict[str, Any]) -> bool:
    """True when a disagreement with this label is non-blocking.

    Derived from provenance only. A label that records no provenance
    at all is trusted — the gate fails closed, so an unrecognised or
    hand-edited label blocks convergence instead of waving it through.
    """
    if label_source(label) in TRUSTED_SOURCES:
        return False
    return label.get("context_mode") in ADVISORY_CONTEXT_MODES or label.get("policy_class") in ADVISORY_POLICY_CLASSES


def provenance_warning(name: str, label: dict[str, Any]) -> str | None:
    """A complaint about how this label's provenance reads, or None.

    Two cheap signals that a label's provenance was written by hand
    rather than by the labeler. Neither changes what the label
    decides — both are reported so a reader can look:

      * an `oracle_weak` that no provenance justifies. The flag alone
        no longer demotes anything, so the mismatch is reported
        rather than obeyed: either the label came from an older
        labeler run, or something tried to demote a trusted label by
        editing it.
      * advisory provenance on a label that records no `prompt_sig`.
        `oracle_label.py` writes a prompt signature — a hash of the
        exact question the judge was asked — on every label it
        produces, so advisory provenance with no signature at all did
        not come from a labeler run.

    Read the second signal for exactly what it says. It catches a
    label whose advisory provenance never came from a labeler run: a
    hand-written one, or one whose signature was dropped. It does NOT
    catch an edit to `context_mode` or `policy_class` on a label the
    labeler really did write, because the signature the labeler left
    survives that edit untouched. Recomputing the signature is what
    catches that, and needs the natural-language policy the judge was
    shown: `policy_convergence.py --policy-nl`.

    The two cases are mutually exclusive (the first needs a trusted
    label, the second an advisory one), so at most one applies.
    """
    if label_source(label) in TRUSTED_SOURCES:
        return None
    if is_advisory(label):
        if label.get("prompt_sig"):
            return None
        return (
            f"advisory provenance with no prompt_sig: {name} "
            f"(context_mode={label.get('context_mode', '<absent>')!r}, "
            f"policy_class={label.get('policy_class', '<absent>')!r}) "
            f"— the labeler records one on every label it writes, so "
            f"this provenance was not written by a labeler run"
        )
    if not label.get("oracle_weak"):
        return None
    return (
        f"oracle_weak without advisory provenance: {name} "
        f"(context_mode={label.get('context_mode', '<absent>')!r}, "
        f"policy_class={label.get('policy_class', '<absent>')!r}) "
        f"— treated as TRUSTED"
    )


# ── Scenario snapshot and evidence digest ─────────────────────

# The only edits the convergence protocol asks a translator to make
# to an already-labeled scenario, and therefore the only fields
# outside its identity.
#
# `ack` records a disagreement the loop could not resolve — a
# required step, so it must not invalidate the evidence a policy was
# finalized against. The label's prose and timestamp are the judge's
# own writing about a verdict that itself stays inside the identity.
# `oracle_weak` decides nothing (see `is_advisory`); leaving it in
# would give an inert flag the power to invalidate a finalize marker.
#
# EVERYTHING ELSE IS EVIDENCE, including the parts of a scenario that
# pose the question the verdict answers: the action and its
# arguments, the graph, the roles, the entity, the principal, the
# action metadata, the provenance block, the cache signatures, and
# the `oracle` block a ground-truth import preserves (written once by
# the import, never legitimately edited afterwards). Renaming a
# scenario's tool call, for instance, changes which decision the
# label is a verdict about, so a label kept beside a rewritten action
# is not the same evidence at all.
MUTABLE_SCENARIO_FIELDS = ("ack",)
MUTABLE_LABEL_FIELDS = (
    "reason",
    "suggested_fix",
    "oracle_weak",
    "ts",
    "ack",
)
# The judge's prose lives inside the verdict object as written by
# oracle_label.py; `verdict.authorized` stays.
MUTABLE_VERDICT_FIELDS = ("reason", "suggested_fix")


def scenario_identity(scenario: Any) -> str:
    """Canonical JSON of everything about a scenario that is evidence.

    Sorted keys and compact separators, so re-serializing a file with
    different indentation or key order is not a change, while any
    difference in a value is.
    """
    if not isinstance(scenario, dict):
        return _canonical(scenario)
    reduced = {k: v for k, v in scenario.items() if k not in MUTABLE_SCENARIO_FIELDS}
    label = reduced.get("label")
    if isinstance(label, dict):
        kept = {k: v for k, v in label.items() if k not in MUTABLE_LABEL_FIELDS}
        verdict = kept.get("verdict")
        if isinstance(verdict, dict):
            kept["verdict"] = {k: v for k, v in verdict.items() if k not in MUTABLE_VERDICT_FIELDS}
        reduced["label"] = kept
    return _canonical(reduced)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def scenario_snapshot(data: dict[str, Any]) -> dict[str, str]:
    """Every scenario's name mapped to its scenario identity.

    Raises on a repeated scenario name. Names are how the loop
    addresses a scenario — the convergence report lists them, an ack
    names one — so two scenarios sharing a name make both the report
    and this check ambiguous, and one of the two could be dropped
    unnoticed. Failing is the safe reading.
    """
    if not isinstance(data, dict):
        raise ValueError("scenarios file must be a JSON object")
    snapshot: dict[str, str] = {}
    for scenario in data.get("scenarios", []):
        if not isinstance(scenario, dict):
            continue
        name = str(scenario.get("name", ""))
        if name in snapshot:
            raise ValueError(f"two scenarios are named {name!r}; names must be unique for scenarios to be tracked")
        snapshot[name] = scenario_identity(scenario)
    return snapshot


def snapshot_problem(
    snapshot: dict[str, str],
    current: dict[str, str],
) -> str | None:
    """First scenario that vanished or changed since `snapshot`, or None.

    The invariant, stated once: every scenario that existed when
    labeling finished is still there and still says the same thing.
    Names that appear only in `current` are fine — author-written
    top-up scenarios are added during translation on purpose, and
    adding evidence cannot weaken the evidence already recorded.
    """
    for name in sorted(snapshot):
        was = snapshot[name]
        if name not in current:
            return (
                f"scenario {name!r} was labeled before translation "
                f"and is no longer in the file; scenarios may be "
                f"added, never removed"
            )
        now = current[name]
        if now != was:
            return f"scenario {name!r} changed since it was labeled ({identity_diff(was, now)})"
    return None


MAX_DIFF_PARTS = 6


def identity_diff(was: str, now: str) -> str:
    """Name the fields that differ between two identities.

    Both sides are canonical JSON, so this parses them back and walks
    the two trees together. The error message is the only place a
    refusal says WHAT moved, which is the difference between a person
    finding the edit and re-running the whole pipeline blind.
    """
    try:
        parts = _diff_paths(json.loads(was), json.loads(now), "")
    except (TypeError, ValueError):
        return "the scenario is no longer readable as JSON"
    if not parts:
        return "identity changed"
    shown = parts[:MAX_DIFF_PARTS]
    if len(parts) > MAX_DIFF_PARTS:
        shown.append(f"... and {len(parts) - MAX_DIFF_PARTS} more")
    return "; ".join(shown)


def _diff_paths(was: Any, now: Any, prefix: str) -> list[str]:
    """Dotted paths at which two decoded identities differ."""
    if isinstance(was, dict) or isinstance(now, dict):
        # A field that changed from an object to something else is
        # still reported per key, so `expected: null` names
        # `expected.authorized` rather than only `expected`.
        left = was if isinstance(was, dict) else {}
        right = now if isinstance(now, dict) else {}
        out: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left:
                out.append(f"{path}: <absent> -> {_short(right[key])}")
            elif key not in right:
                out.append(f"{path}: {_short(left[key])} -> <absent>")
            else:
                out.extend(_diff_paths(left[key], right[key], path))
        return out
    if was != now:
        where = prefix or "scenario"
        return [f"{where}: {_short(was)} -> {_short(now)}"]
    return []


def _short(value: Any, limit: int = 60) -> str:
    """A value's repr, clipped — a rewritten graph is not a message."""
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def snapshot_digest(snapshot: dict[str, str]) -> str:
    """sha256 over the sorted (name, identity) rows of a snapshot."""
    rows = sorted(json.dumps([name, identity], sort_keys=True, default=str) for name, identity in snapshot.items())
    payload = json.dumps(rows, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evidence_digest(data: dict[str, Any]) -> str:
    """The one-string form of a scenarios file's snapshot.

    This is what `scripts/policy_finalize.py` records in the marker
    for each file convergence ran on, so a marker can be checked
    against that file without carrying the whole snapshot. It is a
    digest of the same identities `snapshot_problem` compares, so the
    two can never disagree about what counts as a change.
    """
    return snapshot_digest(scenario_snapshot(data))


def evidence_digest_of_file(path: Path) -> str:
    """Evidence digest of a scenarios file on disk.

    Raises rather than returning a sentinel: a digest that silently
    means "unreadable" would compare equal to another unreadable file
    and bind a marker to nothing.
    """
    return evidence_digest(json.loads(Path(path).read_text()))


# ── Snapshot records ──────────────────────────────────────────

# Stamped into every record so a file written by an older layout is
# refused rather than read as if it meant the same thing.
SNAPSHOT_VERSION = "scenario-snapshot/v2"


def labeled_scenario_count(data: dict[str, Any]) -> int:
    """How many scenarios in this file carry a label."""
    if not isinstance(data, dict):
        return 0
    scenarios = data.get("scenarios", [])
    if not isinstance(scenarios, list):
        return 0
    return sum(1 for s in scenarios if isinstance(s, dict) and s.get("label"))


def scenario_snapshot_record(data: dict[str, Any]) -> dict[str, Any]:
    """The identities, the label census, and a digest over both.

    The census is the part that cannot be re-derived later. Whether a
    run was labeled decides whether its gate has to show a convergence
    run, and reading that off the scenarios file at gate time would
    ask the file being checked to say whether it needs checking:
    deleting every label would then answer "no". Recording the count
    when the snapshot is taken makes the answer independent of what
    the file says afterwards.
    """
    identities = scenario_snapshot(data)
    return {
        "version": SNAPSHOT_VERSION,
        "identities": identities,
        "labeled": labeled_scenario_count(data),
        "digest": snapshot_digest(identities),
    }


def scenario_snapshot_of_file(path: Path) -> dict[str, Any]:
    """Snapshot record of a scenarios file on disk. Raises like above."""
    return scenario_snapshot_record(json.loads(Path(path).read_text()))


def snapshot_record_problem(record: Any) -> str | None:
    """Why this is not a usable snapshot record, or None.

    Checks the shape and the record's agreement with its own digest.
    That last check catches an edited identity map; it does not make
    the record trustworthy, since whoever edited it could recompute
    the digest too.
    """
    if not isinstance(record, dict):
        return "the snapshot is not a JSON object"
    if record.get("version") != SNAPSHOT_VERSION:
        return f"the snapshot records version {record.get('version')!r}, not {SNAPSHOT_VERSION!r}"
    identities = record.get("identities")
    if not isinstance(identities, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in identities.items()
    ):
        return "the snapshot's identities are not a map of strings"
    labeled = record.get("labeled")
    if not isinstance(labeled, int) or isinstance(labeled, bool):
        return "the snapshot records no label count"
    if labeled < 0 or labeled > len(identities):
        return f"the snapshot's label count ({labeled}) does not fit its {len(identities)} scenarios"
    if snapshot_digest(identities) != record.get("digest"):
        return "the snapshot's contents do not match its own digest"
    return None


# ── Oracle accuracy ───────────────────────────────────────────


def oracle_outcome(label: dict[str, Any]) -> str | None:
    """The oracle's canonical call, wherever it now lives.

    On an oracle label that is the verdict itself; on a ground-truth
    label it is the `oracle` block the import preserved.
    """
    if label_source(label) == "oracle":
        return verdict_outcome(label.get("verdict"))
    preserved = label.get("oracle")
    if isinstance(preserved, dict):
        return verdict_outcome(preserved.get("verdict"))
    return None


def ground_truth_outcome(label: dict[str, Any]) -> str | None:
    if label.get("source") != "ground-truth":
        return None
    return verdict_outcome(label.get("verdict"))


def oracle_verdict(label: dict[str, Any]) -> bool | None:
    """Legacy boolean view; ASK deliberately has no boolean projection."""
    outcome = oracle_outcome(label)
    return outcome_authorized(outcome) if outcome else None


def ground_truth_verdict(label: dict[str, Any]) -> bool | None:
    """Legacy boolean view; ASK deliberately has no boolean projection."""
    outcome = ground_truth_outcome(label)
    return outcome_authorized(outcome) if outcome else None


def oracle_accuracy(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    """Agreement between the oracle and ground truth.

    Counts only scenarios carrying both, so it is empty until someone
    imports ground-truth labels — which is the honest reading: without
    them, judge accuracy is unmeasured rather than perfect.
    """
    n_both = n_agree = 0
    disagreements: list[dict[str, Any]] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        label = scenario.get("label") or {}
        truth = ground_truth_outcome(label)
        oracle = oracle_outcome(label)
        if truth is None or oracle is None:
            continue
        n_both += 1
        if truth == oracle:
            n_agree += 1
        else:
            oracle_verdict_obj = (
                label.get("verdict") if label_source(label) == "oracle" else (label.get("oracle") or {}).get("verdict")
            )
            truth_verdict_obj = label.get("verdict")
            legacy_display = (
                isinstance(oracle_verdict_obj, dict)
                and "outcome" not in oracle_verdict_obj
                and isinstance(truth_verdict_obj, dict)
                and "outcome" not in truth_verdict_obj
            )
            disagreements.append(
                {
                    "name": scenario.get("name", "unnamed"),
                    "oracle": (outcome_authorized(oracle) if legacy_display else oracle),
                    "ground_truth": (outcome_authorized(truth) if legacy_display else truth),
                }
            )
    return {
        "n_both": n_both,
        "n_agree": n_agree,
        "accuracy": (n_agree / n_both) if n_both else None,
        "disagreements": disagreements,
    }
