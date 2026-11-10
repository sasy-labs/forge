"""Operator directories derived from AgentDojo's default environments.

The monitor uses these records to resolve known destinations independently of
the agent's context. No attack payloads or recorded trajectories are read.
"""

from __future__ import annotations

import re

from agentdojo.task_suite.load_suites import get_suite


def norm_loose(value: object) -> str:
    """Fold identifiers to lowercase ASCII letters and digits."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _add(out: set[str], *values: object) -> None:
    for value in values:
        if value is None:
            continue
        normalized = norm_loose(value)
        if len(normalized) >= 3:
            out.add(normalized)


def principals(version: str = "v1.2.2") -> dict[str, set[str]]:
    """Return workspace authors by suite, using their original identifiers.

    Slack usernames identify record authors. Suites with email-address senders
    use the policy's own-domain configuration instead. This roster does not
    itself authorize destinations.
    """
    slack = get_suite(version, "slack").load_and_inject_default_environment({})
    return {"slack": set(slack.slack.users)}


def build(version: str = "v1.2.2", vouch_channels: bool = False) -> dict[str, set[str]]:
    """Return normalized destinations vouched for by the operator, by suite.

    Slack channels are excluded by default: AgentDojo permits attacker-controlled
    channel names. Enabling ``vouch_channels`` assumes the registry is controlled
    by the operator, an assumption that does not hold for this benchmark.
    """
    dirs: dict[str, set[str]] = {}

    slack = get_suite(version, "slack").load_and_inject_default_environment({})
    s: set[str] = set()
    _add(s, *slack.slack.users)
    if vouch_channels:
        _add(s, *slack.slack.channels)
    dirs["slack"] = s

    bank = get_suite(version, "banking").load_and_inject_default_environment({})
    b: set[str] = set()
    _add(b, bank.bank_account.iban)
    # Existing and scheduled counterparties are known destinations.
    for tx in list(bank.bank_account.transactions) + list(
        bank.bank_account.scheduled_transactions
    ):
        _add(b, getattr(tx, "recipient", None))
    dirs["banking"] = b

    travel = get_suite(version, "travel").load_and_inject_default_environment({})
    t: set[str] = set()
    _add(t, travel.user.email)
    for hotel in travel.hotels.hotel_list:
        _add(t, hotel.name)
    for restaurant in travel.restaurants.restaurant_list:
        _add(t, restaurant.name)
    for company in travel.car_rental.company_list:
        _add(t, company.name)
    # Saved contacts are operator-curated. Mail correspondents and calendar
    # participants are excluded because outsiders can nominate themselves.
    for contact in travel.inbox.contact_list:
        _add(t, getattr(contact, "email", None), getattr(contact, "name", None))
    dirs["travel"] = t

    ws = get_suite(version, "workspace").load_and_inject_default_environment({})
    w: set[str] = set()
    _add(w, ws.inbox.account_email)
    for contact in ws.inbox.contact_list:
        _add(w, getattr(contact, "email", None), getattr(contact, "name", None))
    dirs["workspace"] = w
    return dirs
