"""Label policy test scenarios with a strong-LLM oracle verdict.

For each recorded authorization scenario, render the agent's context and ask
an oracle model whether the proposed tool call complies with the natural-language
policy. Linear conversation history is the default rendering. Anthropic requests
use structured JSON output; LiteLLM provides the alternative provider route.
These labels support translation evaluation and convergence checks.

Label semantics:
  - label.source: "oracle" | "author" | "ground-truth". Author and
    ground-truth labels are never overwritten without --force.
  - label.context_mode and label.policy_class record how the label
    was produced: which context the judge was shown, and which
    policy class it judged under. Those two fields, and only those,
    decide whether policy_convergence.py treats a disagreement as
    blocking (a mismatch) or as ADVISORY (a disputed-advisory entry
    routed to review). Advisory = context_mode "graph" (graph-only
    rendering measurably degrades judges) or policy_class "taint"
    (needs reasoning over graph topology). A deep slice under the
    default linear rendering is NOT advisory — linear history is the
    judge's reliable regime regardless of conversation depth.
  - label.oracle_weak repeats that decision as one boolean, for
    readers. It is not the source of truth: setting it on a label
    whose provenance says linear/general demotes nothing, and
    convergence reports the combination as a warning.
  - expected.authorized is auto-populated ONLY from trusted
    (non-advisory) labels, so the plain test harness never
    hard-fails on advisory labels.

Verdicts are cached by content hash (provider + model + policy +
system prompt + user prompt) so
re-running after policy edits is free; only new decision points
hit the API.

Usage:
    uv run python experiments/translation/oracle_label.py \\
        --scenarios path/to/captured_tests.json \\
        --policy-nl path/to/policy.md \\
        [--model claude-opus-5] [--context linear|graph] \\
        [--policy-class general|taint] [--cache PATH] \\
        [--limit N] [--force] [--out PATH] [--dry-run]

Env:
    ANTHROPIC_API_KEY    required (unless --dry-run or all cached)
    SASY_ORACLE_MODEL    default model override
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from policy_translation.evaluation.labels import VERDICTS, is_advisory, label_source, verdict_outcome
from policy_translation.inputs.capture import (
    blind_scenario_fixture_identifiers,
    build_user_prompt,
    judge_system_message,
    render_scenario_context,
    scenario_actor,
)

DEFAULT_MODEL = os.environ.get("SASY_ORACLE_MODEL", "claude-opus-5")
DEFAULT_CACHE = Path("outputs/oracle-cache.json")

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "outcome": {
            "type": "string",
            "enum": list(VERDICTS),
            "description": ("ALLOW executes now; ASK requires approval or more evidence; DENY must not execute"),
        },
        "reason": {"type": "string"},
        "suggested_fix": {"type": "string"},
        "suggested_fix_disposition": {
            "type": "string",
            "enum": ["none", "optional", "required-precondition", "repair"],
        },
        "applicable_clauses": {
            "type": "array",
            "items": {"type": "string"},
        },
        "predicate_labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "predicate": {"type": "string"},
                    "value": {
                        "type": "string",
                        "enum": ["true", "false", "unknown", "not-applicable"],
                    },
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
                "required": ["predicate", "value", "evidence", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
        "missing_preconditions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "outcome",
        "reason",
        "suggested_fix",
        "suggested_fix_disposition",
        "applicable_clauses",
        "predicate_labels",
        "missing_preconditions",
    ],
    "additionalProperties": False,
}

LABEL_SCHEMA_VERSION = "policy-judge/v2"
PREDICATE_VALUES = {"true", "false", "unknown", "not-applicable"}


# ── Verdict cache ─────────────────────────────────────────────


def _vocabulary_names(
    vocabulary: dict[str, Any] | None,
) -> tuple[set[str], set[str]]:
    if vocabulary is None:
        return set(), set()

    def names(field: str) -> set[str]:
        raw = vocabulary.get(field, [])
        if not isinstance(raw, list):
            raise ValueError(f"predicate vocabulary {field} must be a list")  # noqa: TRY004
        found: set[str] = set()
        for i, item in enumerate(raw):
            name = item.get("name") if isinstance(item, dict) else item
            if not isinstance(name, str) or not name:
                raise ValueError(f"predicate vocabulary {field}[{i}] has no name")
            if name in found:
                raise ValueError(f"predicate vocabulary {field} names {name!r} twice")
            found.add(name)
        return found

    return names("predicates"), names("clauses")


def validate_oracle_verdict(
    value: Any,
    *,
    vocabulary: dict[str, Any] | None = None,
    evidence_mode: bool | None = None,
) -> dict[str, Any]:
    """Validate untrusted provider/cache JSON without coercion.

    ``bool("false")`` is true in Python, so accepting a string in the
    authorized field silently reverses a denial. The Anthropic path has
    schema enforcement, but LiteLLM responses and persisted caches do not.
    """
    if not isinstance(value, dict):
        raise ValueError("oracle verdict must be a JSON object")  # noqa: TRY004
    outcome = verdict_outcome(value, strict=True)
    for field in ("reason", "suggested_fix"):
        if not isinstance(value.get(field), str):
            raise ValueError(f"oracle verdict {field} must be a string")  # noqa: TRY004

    # V1 cache/provider responses remain readable, but every newly emitted
    # scenario is canonicalized to ``outcome`` by main().
    verdict: dict[str, Any] = {
        ("outcome" if "outcome" in value else "authorized"): (outcome if "outcome" in value else value["authorized"]),
        "reason": value["reason"],
        "suggested_fix": value["suggested_fix"],
    }
    fix_disposition = value.get("suggested_fix_disposition")
    if fix_disposition is not None:
        if fix_disposition not in {"none", "optional", "required-precondition", "repair"}:
            raise ValueError("oracle verdict suggested_fix_disposition is invalid")
        verdict["suggested_fix_disposition"] = fix_disposition
    predicates, clauses = _vocabulary_names(vocabulary)
    for field in ("applicable_clauses", "predicate_labels", "missing_preconditions"):
        raw = value.get(field, [])
        if not isinstance(raw, list):
            raise ValueError(f"oracle verdict {field} must be a list")  # noqa: TRY004
        if field in value or evidence_mode is not None:
            verdict[field] = raw
    if evidence_mode is True and vocabulary is None:
        raise ValueError("evidence labels require a predicate vocabulary")
    if evidence_mode is False and (verdict.get("applicable_clauses") or verdict.get("predicate_labels")):
        raise ValueError("decision-only labels must not contain clause or predicate labels")
    if any(not isinstance(x, str) for x in verdict.get("missing_preconditions", [])):
        raise ValueError("oracle verdict missing_preconditions must be strings")
    if clauses:
        unknown = set(verdict.get("applicable_clauses", [])) - clauses
        if unknown:
            raise ValueError(f"oracle verdict names unknown clause(s): {sorted(unknown)}")
    elif vocabulary is not None and verdict.get("applicable_clauses"):
        raise ValueError("clause labels require a declared clause vocabulary")
    seen_predicates: set[str] = set()
    for i, row in enumerate(verdict.get("predicate_labels", [])):
        if not isinstance(row, dict):
            raise ValueError(f"oracle predicate_labels[{i}] must be an object")  # noqa: TRY004
        predicate = row.get("predicate")
        if vocabulary is not None and predicate not in predicates:
            raise ValueError(f"oracle predicate_labels[{i}] names unknown predicate {predicate!r}")
        if predicate in seen_predicates:
            raise ValueError(f"oracle labels predicate {predicate!r} twice")
        seen_predicates.add(str(predicate))
        if row.get("value") not in PREDICATE_VALUES:
            raise ValueError(f"oracle predicate_labels[{i}] has an invalid value")
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or any(not isinstance(ref, str) or not ref for ref in evidence):
            raise ValueError(f"oracle predicate_labels[{i}].evidence must be strings")
        confidence = row.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            raise ValueError(f"oracle predicate_labels[{i}].confidence must be in [0, 1]")
        if not isinstance(row.get("reason"), str):
            raise ValueError(f"oracle predicate_labels[{i}].reason must be a string")  # noqa: TRY004
    if "usage" in value:
        if not isinstance(value["usage"], dict):
            raise ValueError("oracle verdict usage must be an object")
        verdict["usage"] = value["usage"]
    return verdict


def load_vocabulary(
    path: Path | None,
    *,
    clause_prefixes: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("predicate vocabulary must be a JSON object")  # noqa: TRY004
    if clause_prefixes:
        value = dict(value)
        value["clauses"] = [
            row
            for row in value.get("clauses", [])
            if isinstance(row, (str, dict))
            and (row if isinstance(row, str) else row.get("name", "")).startswith(clause_prefixes)
        ]
    _vocabulary_names(value)
    return value


def expected_after_oracle_label(
    existing_label: dict[str, Any],
    existing_expected: Any,
    *,
    oracle_weak: bool,
    outcome: str,
    provenance_source: str = "",
) -> Any:
    """Preserve independent assertions; update only oracle-owned ones."""
    independent = existing_expected is not None and (
        existing_label.get("expected_owner") == "independent"
        or provenance_source
        in {
            "spec-derived-ground-truth",
            "construction-ground-truth",
            "human-ground-truth",
        }
        or (bool(existing_label) and label_source(existing_label) != "oracle")
        or not existing_label
    )
    if independent:
        return existing_expected
    return None if oracle_weak else {"verdict": outcome}


class VerdictCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, Any] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text())
                if isinstance(loaded, dict):
                    self._data = loaded
                else:
                    print(
                        f"WARNING: invalid cache at {path}; starting fresh",
                        file=sys.stderr,
                    )
            except json.JSONDecodeError:
                print(
                    f"WARNING: corrupt cache at {path}; starting fresh",
                    file=sys.stderr,
                )

    @staticmethod
    def key(
        provider: str,
        model: str,
        policy_nl: str,
        system: str,
        prompt: str,
    ) -> str:
        h = hashlib.sha256()
        for part in (provider, model, policy_nl, system, prompt):
            h.update(part.encode("utf-8"))
            h.update(b"\x1f")
        return h.hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self._data.get(key)
        if not isinstance(entry, dict):
            return None
        try:
            return validate_oracle_verdict(entry.get("verdict"))
        except ValueError:
            return None

    def put(
        self,
        key: str,
        verdict: dict[str, Any],
        provider: str,
        model: str,
    ) -> None:
        self._data[key] = {
            "verdict": validate_oracle_verdict(verdict),
            "provider": provider,
            "model": model,
            "ts": time.time(),
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Merge with any concurrent writer before persisting.
        if self.path.exists():
            try:
                on_disk = json.loads(self.path.read_text())
                if isinstance(on_disk, dict):
                    on_disk.update(self._data)
                    self._data = on_disk
            except json.JSONDecodeError:
                pass
        self.path.write_text(json.dumps(self._data, indent=1) + "\n")


# ── Oracle calls ──────────────────────────────────────────────


def call_oracle_anthropic(model: str, system: str, user_prompt: str) -> dict[str, Any]:
    """One verdict via the Anthropic API with structured output."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": VERDICT_SCHEMA,
            }
        },
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    verdict = json.loads(text)
    verdict["usage"] = {
        "tokens_in": response.usage.input_tokens,
        "tokens_out": response.usage.output_tokens,
    }
    return validate_oracle_verdict(verdict)


def call_oracle_litellm(
    model: str,
    system: str,
    user_prompt: str,
    *,
    vocabulary: dict[str, Any] | None = None,
    evidence_mode: bool | None = None,
) -> dict[str, Any]:
    """Fallback provider routing (Azure/Vertex ablations) via litellm.

    Asks for bare JSON and parses it; no schema enforcement, so a
    retry nudge is included for malformed output.
    """
    import litellm

    suffix = (
        "\n\nRespond with ONLY a JSON object: "
        '{"outcome": "ALLOW|ASK|DENY", "reason": str, '
        '"suggested_fix": str, "suggested_fix_disposition": '
        '"none|optional|required-precondition|repair", "applicable_clauses": [str], '
        "where each applicable clause is an exact artifact-id:Lstart-Lend source citation, "
        '"predicate_labels": [{"predicate": str, "value": '
        '"true|false|unknown|not-applicable", "evidence": [str], '
        '"confidence": number, "reason": str}], '
        '"missing_preconditions": [str]}'
    )
    validation_error = ""
    for attempt in range(2):
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt + suffix},
            ],
        )
        text = resp.choices[0].message.content or ""
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return validate_oracle_verdict(
                    json.loads(text[start : end + 1]),
                    vocabulary=vocabulary,
                    evidence_mode=evidence_mode,
                )
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                validation_error = str(error)
        if attempt == 0:
            suffix += (
                "\nYour previous reply failed validation: "
                f"{validation_error or 'not a JSON object'}. "
                "Correct it using only names from the supplied ABI."
            )
    raise ValueError(f"oracle response failed validation ({validation_error}): {text[:200]}")


# ── Main ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label scenarios with a strong-LLM oracle",
    )
    parser.add_argument("--scenarios", required=True, type=Path)
    parser.add_argument(
        "--policy-nl",
        required=True,
        type=Path,
        help="Natural-language policy text the oracle judges against",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--provider",
        choices=["anthropic", "litellm"],
        default="anthropic",
    )
    parser.add_argument("--context", choices=["linear", "graph", "both"], default="linear")
    parser.add_argument(
        "--policy-class",
        choices=["general", "taint"],
        default="general",
        help="taint = every label from this run is advisory",
    )
    parser.add_argument(
        "--label-schema",
        choices=["decision", "evidence"],
        default="decision",
        help=(
            "decision emits ALLOW/ASK/DENY only; evidence also labels the declared predicates and applicable clauses"
        ),
    )
    parser.add_argument(
        "--predicate-vocabulary",
        type=Path,
        default=None,
        help=("JSON ABI with predicates and clauses; required by --label-schema evidence"),
    )
    parser.add_argument(
        "--clause-prefix",
        action="append",
        default=[],
        help="limit the declared clause ABI to this prefix; repeat as needed",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Label at most N scenarios (0 = all)",
    )
    parser.add_argument(
        "--scenario-name",
        action="append",
        default=[],
        help=("label only this exact scenario name; repeat for a calibration slice (selection happens before --limit)"),
    )
    parser.add_argument(
        "--blind-scenario-identifiers",
        action="store_true",
        help=(
            "redact fixture/result identifiers from tool arguments so a calibration label cannot be inferred from names"
        ),
    )
    parser.add_argument(
        "--evidence-observations",
        choices=["none", "producer"],
        default="producer",
        help=(
            "evidence-mode context: none uses only the trace/action; producer "
            "also includes concrete adapter observations"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Relabel even author/ground-truth/existing labels",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: in place)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print rendered prompts instead of calling the API",
    )
    args = parser.parse_args()

    if not args.scenarios.exists():
        print(f"ERROR: scenarios not found: {args.scenarios}")
        sys.exit(1)
    if not args.policy_nl.exists():
        print(f"ERROR: policy text not found: {args.policy_nl}")
        sys.exit(1)

    data = json.loads(args.scenarios.read_text())
    scenarios = data.get("scenarios", [])
    selected_names = set(args.scenario_name)
    available_names = {
        scenario.get("name")
        for scenario in scenarios
        if isinstance(scenario, dict) and isinstance(scenario.get("name"), str)
    }
    unknown_names = selected_names - available_names
    if unknown_names:
        print(
            f"ERROR: unknown --scenario-name values: {sorted(unknown_names)}",
            file=sys.stderr,
        )
        sys.exit(1)
    policy_nl = args.policy_nl.read_text()
    system = judge_system_message(args.context)
    cache = VerdictCache(args.cache)
    try:
        vocabulary = load_vocabulary(
            args.predicate_vocabulary,
            clause_prefixes=tuple(args.clause_prefix),
        )
        if args.label_schema == "evidence" and vocabulary is None:
            raise ValueError("--label-schema evidence requires --predicate-vocabulary")
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    vocabulary_sig = (
        hashlib.sha256(json.dumps(vocabulary, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        if vocabulary is not None
        else ""
    )

    # Identifies which NL policy text a prior oracle label was judged
    # against, so editing the spec auto-re-judges (no --force needed).
    policy_sig = hashlib.sha256(policy_nl.encode("utf-8")).hexdigest()[:12]
    system_sig = hashlib.sha256(system.encode("utf-8")).hexdigest()[:12]

    labeled = skipped = cache_hits = api_calls = 0
    for scenario in scenarios:
        if selected_names and scenario.get("name") not in selected_names:
            continue
        existing = scenario.get("label") or {}
        existing_expected = scenario.get("expected")
        provenance = scenario.get("provenance") or {}
        provenance_source = str(provenance.get("source", "")) if isinstance(provenance, dict) else ""
        # A label produced by a previous oracle run owns the harness assertion
        # it installed.  A scenario with no prior label (or a human/author
        # label) does not: its expected verdict is independent evaluation
        # evidence and must survive calibration/relabeling.
        if not args.force:
            src = existing.get("source")
            # Hand-authored labels are never clobbered.
            if src in ("author", "ground-truth"):
                skipped += 1
                continue
            # An author scenario may set `expected` with no label
            # dict — that is ground truth too; don't overwrite it.
            if not existing and scenario.get("expected") is not None:
                skipped += 1
                continue

        prompt_scenario = blind_scenario_fixture_identifiers(scenario) if args.blind_scenario_identifiers else scenario
        action = prompt_scenario.get("action", {})
        context_body = render_scenario_context(prompt_scenario, args.context)
        action_args = action.get("args", "{}")
        prompt = build_user_prompt(
            policy_nl,
            args.context,
            context_body,
            action.get("fn_name", ""),
            action_args,
            scenario_actor(prompt_scenario),
        )
        base_prompt_sig = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        if args.label_schema == "evidence":
            if args.evidence_observations == "producer":
                prompt += "\n\nCONCRETE TRACE/ADAPTER OBSERVATIONS (observations, not a verdict):\n" + json.dumps(
                    {
                        "roles": scenario.get("roles", []),
                        "action_metadata": prompt_scenario.get("action_metadata", []),
                        "policy_metadata": prompt_scenario.get("policy_metadata", []),
                    },
                    sort_keys=True,
                    indent=2,
                )
            prompt += (
                "\n\nEVIDENCE LABELING ABI (use only these names):\n"
                + json.dumps(vocabulary, sort_keys=True, indent=2)
                + "\nFor every relevant predicate, return true, false, "
                "unknown, or not-applicable. Cite concrete trace node IDs, "
                "argument paths, or source spans in evidence. Unknown is "
                "different from false. Do not infer facts from fixture or "
                "case names. applicable_clauses must also come from the ABI."
            )
        else:
            prompt += (
                "\n\nReturn the three-valued outcome. Leave "
                "applicable_clauses and predicate_labels empty; this arm "
                "measures decision-only judging."
            )
        prompt_sig = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]

        # Reuse a prior oracle label only when everything that shaped it is
        # unchanged. Policy text alone is not enough: a different model,
        # context, schema, or trust class can change the verdict or trust level.
        if (
            not args.force
            and src == "oracle"
            and (
                existing.get("policy_sig") == policy_sig
                and existing.get("system_sig") == system_sig
                and existing.get("prompt_sig") == prompt_sig
                and existing.get("provider") == args.provider
                and existing.get("model") == args.model
                and existing.get("context_mode") == args.context
                and existing.get("policy_class", "general") == args.policy_class
                and existing.get("label_schema", "decision") == args.label_schema
                and existing.get("vocabulary_sig", "") == vocabulary_sig
                and existing.get("blind_scenario_identifiers", False) == args.blind_scenario_identifiers
                and existing.get("evidence_observations", "producer") == args.evidence_observations
            )
        ):
            skipped += 1
            continue
        if args.limit and labeled >= args.limit:
            break

        if args.dry_run:
            print(f"=== {scenario.get('name')} ===")
            print(prompt)
            print()
            labeled += 1
            continue

        key = VerdictCache.key(args.provider, args.model, policy_nl, system, prompt)
        verdict = cache.get(key)
        if verdict is not None:
            try:
                verdict = validate_oracle_verdict(
                    verdict,
                    vocabulary=vocabulary,
                    evidence_mode=args.label_schema == "evidence",
                )
            except (TypeError, ValueError):
                # Older cache entries may predate the vocabulary-bound schema.
                # Treat them as misses; a valid response will replace the key.
                verdict = None
            else:
                cache_hits += 1
        if verdict is None:
            try:
                if args.provider == "anthropic":
                    verdict = call_oracle_anthropic(args.model, system, prompt)
                else:
                    verdict = call_oracle_litellm(
                        args.model,
                        system,
                        prompt,
                        vocabulary=vocabulary,
                        evidence_mode=args.label_schema == "evidence",
                    )
                verdict = validate_oracle_verdict(
                    verdict,
                    vocabulary=vocabulary,
                    evidence_mode=args.label_schema == "evidence",
                )
            except Exception as e:  # noqa: BLE001 - provider boundary
                print(
                    f"ERROR: oracle provider failed for {scenario.get('name', 'unnamed')}: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                sys.exit(1)
            cache.put(key, verdict, args.provider, args.model)
            cache.save()
            api_calls += 1
        outcome = verdict_outcome(verdict, strict=True)

        # Advisory only when the verdict needs graph topology the
        # judge handles poorly: an explicitly taint-class policy, or
        # graph-rendered context (measurably worse for judges). A
        # deep slice under LINEAR rendering is NOT weak — linear
        # history already collapses conversation depth to exactly
        # what the agent saw, which is the judge's reliable regime.
        # Graph-ONLY rendering measurably degrades judges, so those
        # labels are advisory. "both" keeps the linear narrative and
        # adds the graph beside it, so the degradation argument does
        # not apply; it is treated as trusted. That is a reasoned
        # default, not a measured one --- we have not evaluated judge
        # accuracy under the combined view.
        #
        # context_mode and policy_class below are what convergence
        # actually reads; is_advisory() re-derives this same boolean
        # from them. Writing oracle_weak too keeps the file readable
        # without making the flag load-bearing.
        oracle_weak = is_advisory(
            {
                "source": "oracle",
                "context_mode": args.context,
                "policy_class": args.policy_class,
            }
        )
        scenario["label"] = {
            "verdict": {
                "outcome": outcome,
                "reason": verdict.get("reason", ""),
                "suggested_fix": verdict.get("suggested_fix", ""),
                "suggested_fix_disposition": verdict.get("suggested_fix_disposition", "none"),
                "applicable_clauses": verdict.get("applicable_clauses", []),
                "predicate_labels": verdict.get("predicate_labels", []),
                "missing_preconditions": verdict.get("missing_preconditions", []),
            },
            "schema": LABEL_SCHEMA_VERSION,
            "label_schema": args.label_schema,
            "vocabulary_sig": vocabulary_sig,
            "clause_prefixes": args.clause_prefix,
            "blind_scenario_identifiers": args.blind_scenario_identifiers,
            "evidence_observations": args.evidence_observations,
            "expected_owner": (
                "independent"
                if expected_after_oracle_label(
                    existing,
                    existing_expected,
                    oracle_weak=False,
                    outcome=outcome,
                    provenance_source=provenance_source,
                )
                is existing_expected
                and existing_expected is not None
                else "oracle"
            ),
            "source": "oracle",
            "provider": args.provider,
            "model": args.model,
            "context_mode": args.context,
            "policy_class": args.policy_class,
            "oracle_weak": oracle_weak,
            "policy_sig": policy_sig,
            "system_sig": system_sig,
            "prompt_sig": prompt_sig,
            "base_prompt_sig": base_prompt_sig,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if verdict.get("usage"):
            scenario["label"]["verdict"]["usage"] = verdict["usage"]
        # Only trusted labels become harness assertions, and only when the
        # oracle owns that assertion.  Never overwrite author/spec/human
        # ground truth merely because --force requested a fresh judge label.
        scenario["expected"] = expected_after_oracle_label(
            existing,
            existing_expected,
            oracle_weak=oracle_weak,
            outcome=outcome,
            provenance_source=provenance_source,
        )
        labeled += 1

    out_path = args.out or args.scenarios
    if not args.dry_run:
        out_path.write_text(json.dumps(data, indent=2) + "\n")

    print(
        f"Labeled {labeled} scenarios ({cache_hits} from cache, "
        f"{api_calls} API calls), skipped {skipped} with existing "
        f"labels{' [dry-run]' if args.dry_run else ''}" + ("" if args.dry_run else f" → {out_path}")
    )


if __name__ == "__main__":
    main()
