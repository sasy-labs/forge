"""Turn a suite's taxonomy into the `PolicyMetadata` facts the policy reads.

`agentdojo_policy.dl` contains no suite-specific knowledge. Which tools are
untrusted, which are sinks, which argument names a destination — all of it
arrives as `(rel, a, b)` triples on the policy upload and is materialised as
EDB facts when a session's evaluator initialises. Swapping suites is a matter
of shipping a different metadata block against the same rules.

Keeping the taxonomy out of the rules is not just tidiness. The rules are what
gets reviewed, tested and argued about; the taxonomy is configuration that
changes whenever a tool is added. Mixing them means every new tool edits the
security-relevant file.
"""

from __future__ import annotations

import json

from .taxonomy import OPERATOR, TAXONOMIES


def build(
    suite: str,
    entity_suffixes: tuple[str, ...] | None = None,
    *,
    fetch_gate: bool = False,
    strict_taint: bool = False,
    confidentiality: bool = False,
    gate_unanchorable: bool = False,
    account_provenance: bool = False,
    browse_exception: bool = False,
    selection_check: bool = False,
    strict_consent: bool = False,
    target_exemption: bool = True,
    dictated_content: bool = False,
    own_principals: frozenset[str] = frozenset(),
) -> list[tuple[str, str, str]]:
    """The metadata triples for one suite, in upload order.

    Raises `ValueError` if a tool marked irreversible has no effect kind: the
    intent rule is keyed on the kind, so such a tool would need no consent.
    """
    taxonomy = TAXONOMIES[suite]
    operator = OPERATOR[suite]
    irreversible = operator.consent_required | taxonomy.destructive_sinks
    unclassified = sorted(irreversible - set(operator.effect_kinds))
    if unclassified:
        raise ValueError(
            f"{suite}: irreversible tools with no effect_kind: {', '.join(unclassified)}"
        )
    facts: list[tuple[str, str, str]] = []

    for fn in sorted(taxonomy.untrusted_sources):
        facts.append(("untrusted_source", fn, ""))
    for fn in sorted(taxonomy.trusted_sources):
        facts.append(("trusted_source", fn, ""))
    # URL sinks apply destination provenance checks to fetched URLs as well as
    # messages, blocking link-based exfiltration. Enabling these checks can also
    # deny legitimate fetches when the destination appears only in untrusted content.
    sinks = set(taxonomy.sinks)
    destinations: dict[str, tuple[str, ...]] = {
        fn: tuple(fields) for fn, fields in taxonomy.destination_fields.items()
    }
    # Fetching a URL is a FLOW, not just a read: `get_webpage(url=".../?d=SECRET")`
    # puts the data in the request. So any arm that claims to govern flows has to
    # include it, and the allowlist arm implies it.
    if fetch_gate or strict_taint:
        sinks |= set(operator.url_sinks)
        for fn, fields in operator.url_sinks.items():
            merged = destinations.get(fn, ())
            destinations[fn] = merged + tuple(f for f in fields if f not in merged)
            # Whether the destination is a URL is a fact about the TOOL. The
            # confidentiality rule uses it to tell a fetch that can carry data
            # out from one that cannot.
            facts.append(("url_sink", fn, ""))

    for fn in sorted(sinks):
        facts.append(("sink", fn, ""))
    for fn, fields in sorted(destinations.items()):
        for field in fields:
            facts.append(("destination_field", fn, field))
    for fn, fields in sorted(operator.body_fields.items()):
        for field in fields:
            facts.append(("body_field", fn, field))
    # Which KIND of effect each tool has. The intent rule asks whether the
    # request named an effect of that kind, so a request to delete a file does
    # not also authorise a password change. Every irreversible tool must have
    # one (checked above).
    for fn, kind in sorted(operator.effect_kinds.items()):
        facts.append(("effect_kind", fn, kind))
    for fn, fields in sorted(operator.value_from_user.items()):
        for field in fields:
            facts.append(("value_from_user", fn, field))
    for domain in sorted(operator.own_domains):
        facts.append(("own_domain", domain, ""))
    if operator.own_domains:
        # The same set as one delimited value. The record-splitting check needs
        # every domain at once to decide which records a colleague wrote, and a
        # functor argument cannot be a set.
        facts.append(("own_domain_list", "|".join(sorted(operator.own_domains)), ""))
    for identifier in sorted(operator.self_identifiers):
        # Not `own_domain`: a personal name is not a domain, and the domain
        # rule matches hosts. These are matched whole.
        facts.append(("own_identity", identifier, ""))
    # Sets the policy only iterates travel as one delimited value: Datalog has
    # no set-valued functor argument, so a relation would force a cross product
    # with the index relation for no gain.
    suffixes = tuple(entity_suffixes or operator.entity_suffixes)
    if suffixes:
        facts.append(("entity_suffix", "|".join(suffixes), ""))
    if operator.entity_generic:
        facts.append(("entity_generic", "|".join(operator.entity_generic), ""))
    if suffixes:
        # Only declare the response tool where there is a catalogue to check
        # against; without one the rule has nothing to say.
        facts.append(("response_tool", "respond", ""))

    for fn in sorted(taxonomy.confidential_sources):
        facts.append(("confidential_source", fn, ""))

    if strict_taint:
        # The label-conditioned allowlist arm: no sink once tainted, and none of
        # the argument-level rules. See `StrictTaint` in the policy.
        facts.append(("strict_taint", "on", ""))
    if strict_consent:
        facts.append(("strict_consent", "on", ""))
        # Kinds the target rule does not govern, because there is no value the
        # policy can meaningfully call the target. Declared rather than
        # inferred: which kinds those are is a fact about the tools.
        #
        # Switchable because the two settings are genuinely different policies
        # and both are worth measuring: without the exemption the target rule
        # also demands that a fetched URL be one the user identified, which
        # refuses six clean tasks and incidentally blocks the visit class.
        if target_exemption:
            for kind in sorted(operator.target_exempt_kinds):
                facts.append(("target_exempt_kind", kind, ""))
    if selection_check and operator.selection:
        facts.append(("selection_check", "on", ""))
        for fn, (arg, id_field, fields) in sorted(operator.selection.items()):
            facts.append(("selection_target", fn, arg))
            facts.append(("selection_id_field", fn, id_field))
            for cls, listing_field in sorted(fields.items()):
                facts.append((
                    "selection_field", fn,
                    json.dumps({"class": cls, "field": listing_field}),
                ))
    if dictated_content:
        facts.append(("dictated_content", "on", ""))
        for fn, fields in sorted(operator.dictated_fields.items()):
            for f in fields:
                facts.append(("dictated_field", fn, f))
        if own_principals:
            # One delimited value, for the same reason as the domains: the
            # record split needs the whole set at once and a functor argument
            # cannot be a set. Supplied by the caller from the operator's own
            # directory rather than declared here, so there is one roster and
            # not two.
            facts.append((
                "own_principal_list",
                "|".join(sorted(own_principals)),
                "",
            ))
    if browse_exception:
        facts.append(("browse_exception", "on", ""))
    if account_provenance:
        # Rule K: an account-number destination must come from the user's
        # request or the operator's records.
        facts.append(("account_provenance", "on", ""))
    if gate_unanchorable:
        # Rule 2c: a tainted sink that names no destination has no evidence to
        # weigh, so there is nothing for the anchoring rules to bind to.
        facts.append(("gate_unanchorable", "on", ""))
    if confidentiality or strict_taint:
        # The permitted-flow half. Implied by the allowlist arm, because a
        # policy with only the integrity half is not the published one.
        facts.append(("confidentiality", "on", ""))

    return facts


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "banking"
    for rel, a, b in build(name):
        print(f"{rel}\t{a}\t{b}")
