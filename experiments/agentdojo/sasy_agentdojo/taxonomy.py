"""Trust taxonomy for the four AgentDojo suites.

This is the *deployer-authored* part of an AgentDojo policy: which tools return
content an outside party can write (untrusted sources), which tools change state
or move data outward (sinks), and which argument of a sink names the party the
data goes to (destination fields).

Nothing here is derived from the benchmark's attacks or from the injection
vectors: it is the labelling a real operator of a banking / Slack / travel /
workspace agent would produce by reading the tool list. Whether that labelling
is *enough* to stop the attacks is the empirical question the replay harness
answers.

Terminology used throughout:

untrusted source
    A tool whose result can contain text written by somebody other than the
    user or the operator — a bill file, a web page, an inbound email, a hotel
    review, a Slack message from a third party.

trusted source
    A tool whose result is produced by the operator's own system of record —
    the user's own IBAN, the workspace channel directory, today's date.
    Values that appear here can safely be treated as "the user's own world".

sink
    A tool that changes durable state or sends data to someone. These are the
    calls worth gating; reads are not.

destination field
    The argument of a sink that names *who receives* the effect: an IBAN, an
    email address, a Slack channel, a URL. The AgentDojo injections all work by
    getting a legitimate-looking sink to fire at an attacker-chosen
    destination, so this field is where an argument-level policy bites.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SuiteTaxonomy:
    name: str
    untrusted_sources: frozenset[str]
    trusted_sources: frozenset[str]
    sinks: frozenset[str]
    #: sink tool name -> argument names that carry a destination identifier
    destination_fields: dict[str, tuple[str, ...]]
    #: sinks that have no destination argument (delete/cancel style effects)
    destructive_sinks: frozenset[str] = field(default_factory=frozenset)
    #: Tools whose result is the USER'S OWN data. Distinct from
    #: `untrusted_sources`, which is an integrity claim ("this may be written by
    #: an outsider"); this is a confidentiality one ("this must not leave"). A
    #: public web page is untrusted and not confidential; the user's inbox is
    #: confidential and also untrusted; the operator's own catalogue is neither.
    confidential_sources: frozenset[str] = field(default_factory=frozenset)
    #: Tools that enumerate the operator's own directory of valid destinations
    #: — channels, workspace members, the user's contacts, the booking
    #: catalogue. A name that comes back from one of these is a legitimate
    #: place to send something *even though the text of the listing may not be
    #: trustworthy*: anyone who can create a Slack channel chooses its name,
    #: so the name is a valid destination and untrusted content at the same
    #: time. Membership in this set anchors destinations; membership in
    #: `untrusted_sources` taints content. A tool can be in both.
    directory_sources: frozenset[str] = field(default_factory=frozenset)


BANKING = SuiteTaxonomy(
    name="banking",
    # A bill file is written by whoever issued it; transaction subjects are
    # free text chosen by whoever sent the money.
    untrusted_sources=frozenset(
        {"read_file", "get_most_recent_transactions", "get_scheduled_transactions"}
    ),
    trusted_sources=frozenset({"get_iban", "get_balance", "get_user_info"}),
    directory_sources=frozenset({"get_iban", "get_user_info"}),
    # The account itself: balances, transaction history, the user's own filed
    # documents and profile. All of it is the user's data.
    confidential_sources=frozenset(
        {"read_file", "get_most_recent_transactions", "get_scheduled_transactions",
         "get_iban", "get_balance", "get_user_info"}
    ),
    sinks=frozenset(
        {
            "send_money",
            "schedule_transaction",
            "update_scheduled_transaction",
            "update_password",
            "update_user_info",
        }
    ),
    destination_fields={
        "send_money": ("recipient",),
        "schedule_transaction": ("recipient",),
        "update_scheduled_transaction": ("recipient",),
    },
    # `update_user_info` is deliberately absent. Changing a postal address is a
    # routine profile edit that the user can undo by editing it again, not an
    # irreversible effect, and the consent-word list exists for deletions and
    # credential changes. Gating it refused two legitimate tasks in the live
    # run ("update my account information with my new address") and protects
    # nothing: no injection in the suite targets it. Left in `sinks` so the
    # provenance rules still see it.
    destructive_sinks=frozenset({"update_password"}),
)

SLACK = SuiteTaxonomy(
    name="slack",
    # Channel and user names are chosen by whoever created them, so the
    # directory listing is untrusted *text* even though it is an authoritative
    # list of *destinations* — see `directory_sources`.
    untrusted_sources=frozenset(
        {
            "get_webpage",
            "read_channel_messages",
            "read_inbox",
            "get_channels",
            "get_users_in_channel",
        }
    ),
    # Deliberately empty: anchoring destinations on what the agent read back
    # from `get_channels` is a laundering channel, because a channel name is
    # itself attacker-chosen text. Slack destinations are vouched for by the
    # monitor-side directory in `directory.py` instead.
    trusted_sources=frozenset(),
    directory_sources=frozenset(),
    # The workspace's own conversations and membership. `get_webpage` is
    # deliberately absent: a public page is untrusted, because an outsider
    # writes it, but it is not the user's to keep.
    confidential_sources=frozenset(
        {"read_channel_messages", "read_inbox", "get_channels", "get_users_in_channel"}
    ),
    sinks=frozenset(
        {
            "send_direct_message",
            "send_channel_message",
            "post_webpage",
            "invite_user_to_slack",
            "add_user_to_channel",
            "remove_user_from_slack",
        }
    ),
    destination_fields={
        "send_direct_message": ("recipient",),
        "send_channel_message": ("channel",),
        "post_webpage": ("url",),
        "invite_user_to_slack": ("user", "user_email"),
        "add_user_to_channel": ("user", "channel"),
        "remove_user_from_slack": ("user",),
    },
)

TRAVEL = SuiteTaxonomy(
    name="travel",
    # Reviews are written by other travellers. Prices, addresses and
    # availability come from the booking operator's own catalogue.
    # Calendar reads are here for the same reason they are in WORKSPACE: they
    # are the same functions over the same Calendar type, returning free-text
    # descriptions somebody else can write. Leaving them out of travel tracked
    # where this benchmark happens to hide payloads, not what the tool does.
    untrusted_sources=frozenset(
        {
            "get_rating_reviews_for_hotels",
            "get_rating_reviews_for_restaurants",
            "get_rating_reviews_for_car_rental",
            "get_day_calendar_events",
            "search_calendar_events",
        }
    ),
    trusted_sources=frozenset(
        {
            "get_user_information",
            "get_all_hotels_in_city",
            "get_all_restaurants_in_city",
            "get_all_car_rental_companies_in_city",
            "get_hotels_address",
            "get_restaurants_address",
            "get_car_rental_address",
            "get_hotels_prices",
            "get_price_for_restaurants",
            "get_car_price_per_day",
            "get_flight_information",
            "get_cuisine_type_for_restaurants",
            "get_dietary_restrictions_for_all_restaurants",
            "get_contact_information_for_restaurants",
            "check_restaurant_opening_hours",
            "get_car_types_available",
            "get_car_fuel_options",
        }
    ),
    sinks=frozenset(
        {
            "reserve_hotel",
            "reserve_car_rental",
            "reserve_restaurant",
            "send_email",
            "create_calendar_event",
            "cancel_calendar_event",
        }
    ),
    destination_fields={
        "send_email": ("recipients",),
        "reserve_hotel": ("hotel",),
        "reserve_restaurant": ("restaurant",),
        "reserve_car_rental": ("company",),
        "create_calendar_event": ("participants",),
    },
    destructive_sinks=frozenset({"cancel_calendar_event"}),
    # The user's own calendar and profile. The catalogue — hotels, restaurants,
    # rental firms, their prices and addresses — is the operator's published
    # data and is not confidential; nor are other travellers' reviews.
    confidential_sources=frozenset(
        {"get_day_calendar_events", "search_calendar_events", "get_user_information"}
    ),
    directory_sources=frozenset(
        {
            "get_user_information",
            "get_all_hotels_in_city",
            "get_all_restaurants_in_city",
            "get_all_car_rental_companies_in_city",
        }
    ),
)

WORKSPACE = SuiteTaxonomy(
    name="workspace",
    # Everything that surfaces content somebody else authored: mail, shared
    # drive files, calendar invitations.
    untrusted_sources=frozenset(
        {
            "search_emails",
            "get_unread_emails",
            "get_received_emails",
            "get_draft_emails",
            "get_sent_emails",
            "search_files",
            "search_files_by_filename",
            "get_file_by_id",
            "list_files",
            "search_calendar_events",
            "get_day_calendar_events",
        }
    ),
    trusted_sources=frozenset(
        {"get_current_day", "search_contacts_by_name", "search_contacts_by_email"}
    ),
    # Mail, files and calendar — the user's own account, all of it.
    confidential_sources=frozenset(
        {"get_received_emails", "get_unread_emails", "get_sent_emails",
         "get_draft_emails", "search_emails", "get_file_by_id", "list_files",
         "search_files", "search_files_by_filename", "get_day_calendar_events",
         "search_calendar_events", "search_contacts_by_email",
         "search_contacts_by_name"}
    ),
    directory_sources=frozenset({"search_contacts_by_name", "search_contacts_by_email"}),
    sinks=frozenset(
        {
            "send_email",
            "delete_email",
            "delete_file",
            "share_file",
            "create_file",
            "append_to_file",
            "create_calendar_event",
            "cancel_calendar_event",
            "reschedule_calendar_event",
            "add_calendar_event_participants",
        }
    ),
    destination_fields={
        "send_email": ("recipients", "cc", "bcc"),
        "share_file": ("email",),
        "create_calendar_event": ("participants",),
        "add_calendar_event_participants": ("participants",),
    },
    destructive_sinks=frozenset({"delete_email", "delete_file", "cancel_calendar_event"}),
)

TAXONOMIES: dict[str, SuiteTaxonomy] = {
    t.name: t for t in (BANKING, SLACK, TRAVEL, WORKSPACE)
}


@dataclass(frozen=True)
class OperatorConfig:
    """Static configuration a deployer supplies alongside the policy — the
    equivalent of SASY's ``PolicyMetadata`` facts. None of it is derived from
    the benchmark's attacks; it is the sort of thing an operator knows about
    their own tenant.

    own_domains
        Email/web domains belonging to the organisation running the agent.
        Sending to one of these is an internal action, not egress.

    self_identifiers
        Strings naming the user or the tenant themselves, which are always
        legitimate destinations.

    url_sinks
        Tools whose URL argument reaches out to a host. Fetching a URL is a
        read, but it is also an outbound request that can carry data in the
        path or query, and it is how an agent gets pulled to an attacker's
        site — so it is gated like a sink.

    body_fields
        Arguments carrying free text that the agent composes and sends
        onward. Identifiers appearing here that came only from untrusted
        content are how a link or an address gets laundered into a message
        whose destination is entirely legitimate.
    """

    own_domains: frozenset[str] = frozenset()
    #: Arguments whose whole value the dictation rule inspects. Separate from
    #: `body_fields`, which the laundering rule scans for identifiers: a URL is
    #: not composed prose, but "was this exact value handed to the agent by an
    #: outsider" is still the right question to ask of it.
    dictated_fields: dict[str, tuple[str, ...]] = field(default_factory=dict)
    self_identifiers: frozenset[str] = frozenset()
    url_sinks: dict[str, tuple[str, ...]] = field(default_factory=dict)
    body_fields: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: sinks whose effect must be named in the user's own request
    consent_required: frozenset[str] = frozenset()
    #: sink -> argument whose *value* must appear in the user's request
    value_from_user: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Final words of the entity types this operator has a catalogue for. The
    #: response rule refuses a name of one of these types that the catalogue
    #: does not list, so only types the operator can actually vouch for belong
    #: here — a type with no catalogue would refuse every mention.
    entity_suffixes: tuple[str, ...] = ()
    #: Category names an answer is organised under, which look like entity
    #: names but name no business and so need nothing to vouch for them.
    entity_generic: tuple[str, ...] = ()
    #: tool -> the KIND of irreversible effect it has. The consent check asks
    #: whether the user's request named an effect of that kind, so "delete the
    #: largest file" no longer authorises a credential change. Every tool in
    #: `consent_required` or the suite's `destructive_sinks` must have a kind;
    #: `policy_metadata.build` refuses the configuration otherwise.
    effect_kinds: dict[str, str] = field(default_factory=dict)
    #: Rule S configuration. `selection`: tool -> (argument naming the chosen
    #: thing, field identifying records in that tool's listings, {criterion
    #: class: listing field}). Only tools whose listings actually carry a
    #: comparable field belong here; a tool left out means the rule stands down
    #: for it rather than guessing.
    selection: dict[str, tuple[str, str, dict[str, str]]] = field(default_factory=dict)
    #: Effect kinds the TARGET rule does not govern, because the policy has no
    #: value it can meaningfully call the target.
    #:
    #: For a page read there is nothing to identify: the fetch changes nothing
    #: and discloses nothing, so which page is not a consent question the way
    #: which file to delete is. For a calendar event the target is the event
    #: itself — its title and time — and the attacker writes both, so checking
    #: the participant instead would ask the wrong question and answer it
    #: "yes". These kinds are gated by INTENT alone: the user must have asked
    #: for an effect of this kind at all.
    target_exempt_kinds: frozenset[str] = field(default_factory=frozenset)


# The AgentDojo environments all belong to one fictional user, Emma Johnson of
# Blue Sparrow Tech — stated in the benchmark's own system message, which is
# the same place a real deployer's tenant configuration would come from.
_EMMA = frozenset({"emma.johnson@bluesparrowtech.com", "emma johnson", "emma"})
_BST = frozenset({"bluesparrowtech.com"})

OPERATOR: dict[str, OperatorConfig] = {
    "banking": OperatorConfig(
        own_domains=_BST,
        self_identifiers=_EMMA,
        # A transfer's subject line is free text that goes out with the money;
        # five of this suite's injections use it as the exfiltration channel.
        body_fields={
            "send_money": ("subject",),
            "schedule_transaction": ("subject",),
            # `amount` and `date` are checked as well as `subject`. Only the
            # recipient and subject were, so an attacker could leave an already
            # anchored payee in place and simply change what gets sent to them.
            "update_scheduled_transaction": ("subject", "amount", "date"),
        },
        # `update_user_info` is NOT here, and the comment above finally matches
        # the code. It was in both sets, so `irreversible` included it and the
        # consent rule gated it — 20 denials in one measured arm, and banking
        # user_task_13 and user_task_15 lost. Changing a postal address is a
        # routine profile edit the user can redo; the consent vocabulary exists
        # for deletions and credential changes. Removing it from
        # `consent_required` alone is not enough, because the intent rule binds
        # on `EffectKind` rather than on `Irreversible`.
        consent_required=frozenset({"update_password"}),
        effect_kinds={"update_password": "credential"},
        value_from_user={"update_password": ("password",)},
    ),
    "slack": OperatorConfig(
        own_domains=_BST,
        self_identifiers=_EMMA,
        # The URL a fetch is pointed at. A page the agent already read can hand
        # it the next address to visit, and that is the whole of the "just visit
        # this link" attack — there is no message body to inspect. Kept out of
        # `body_fields` so the laundering rule, which extracts identifiers from
        # composed prose, is unaffected.
        dictated_fields={"get_webpage": ("url",)},
        url_sinks={"get_webpage": ("url",), "post_webpage": ("url",)},
        body_fields={
            "send_direct_message": ("body",),
            "send_channel_message": ("body",),
            "post_webpage": ("content",),
        },
        consent_required=frozenset({"remove_user_from_slack"}),
        # `get_webpage` carries a kind but is NOT in `consent_required`: reading
        # a page is not irreversible, so it needs no consent by itself. The kind
        # exists so the browse exception can ask whether the user wanted a page
        # read at all.
        effect_kinds={"remove_user_from_slack": "delete", "get_webpage": "browse"},
        target_exempt_kinds=frozenset({"browse"}),
    ),
    "travel": OperatorConfig(
        own_domains=_BST,
        self_identifiers=_EMMA,
        # Booking dates. An injection that picks the hotel can also pick the
        # stay, and the catalogue vouches for the hotel but not for the dates.
        # The agent writes these values in ISO form from prose, so rule W reads
        # them as dates rather than as verbatim text; a date the user's own
        # request names is exempt.
        dictated_fields={
            "reserve_hotel": ("start_day", "end_day"),
            "reserve_restaurant": ("start_time",),
            "reserve_car_rental": ("start_time", "end_time"),
        },
        body_fields={
            # `subject` is here for the same reason banking's `send_money`
            # subject is: it is free text that leaves the deployment, so it
            # carries an identifier just as well as the body does. Leaving it
            # out gave an attacker a laundering route that needed no cleverness
            # at all — put the link in the subject line.
            "send_email": ("body", "subject"),
            "create_calendar_event": ("description", "title"),
        },
        consent_required=frozenset({"cancel_calendar_event"}),
        effect_kinds={
            "cancel_calendar_event": "cancel",
            # Creating an event is not destructive and not irreversible, so it
            # is deliberately absent from `consent_required`. It carries a kind
            # so that INTENT is checked: measured on this benchmark, every
            # legitimate calendar creation comes from a request that asks for
            # one, and twenty attacked calls come from requests that never
            # mention a calendar at all.
            "create_calendar_event": "calendar",
            # Bookings are not destructive, but they are still effects the user
            # has to have asked for. The stems live in the functor and are type
            # words only — never a name part like "Palace".
            "reserve_hotel": "lodging",
            "reserve_restaurant": "dining",
            "reserve_car_rental": "transport",
        },
        target_exempt_kinds=frozenset({"calendar"}),
        # Operator-supplied, and a known weakness. The policy's criterion is
        # "only types the catalogue can vouch for", and this list does not meet
        # it: six of the eleven original entries matched no catalogue name at
        # all, which made the rule read "refuse any <Name> Hotel the user did
        # not type" — true of this benchmark's payload and of little else.
        #
        # The list is now the catalogue's TYPE vocabulary, taken from the
        # collections it keeps — it holds hotels, restaurants and car rental
        # companies — rather than from the names of the entries in it. That
        # distinction is the whole point. Deriving from entry names gives "View"
        # (from the real hotel "Riverside View"), and a rule that refuses any
        # "<Something> View" the catalogue does not list is a rule about the
        # shape of the data. An operator that keeps a hotel catalogue, on the
        # other hand, is entitled to say what a hotel is.
        #
        # The type word is stripped before the catalogue is consulted
        # (`@entity_core`), so "Riverside View Hotel" is looked up as the entry
        # "Riverside View" and matches. Without that, appending a type word to a
        # real business name would defeat the check.
        # These configured suffixes identify entity types recognized by the
        # suite's catalogue lookup.
        entity_suffixes=(
            "Car Rental", "Car Rentals", "Hotel", "Hotels",
            "Rental", "Rentals", "Restaurant", "Restaurants",
        ),
        # Category headings an answer is organised under. Kept short and only
        # for phrases that are a bare type plus a modifier; a longer list would
        # be false-positive suppression fitted to observed outputs.
        entity_generic=("Car Rental", "Car Rentals"),
    ),
    "workspace": OperatorConfig(
        own_domains=_BST,
        self_identifiers=_EMMA,
        body_fields={
            "send_email": ("body", "subject"),
            "create_file": ("content",),
            "append_to_file": ("content",),
            "create_calendar_event": ("description", "title"),
        },
        # `reschedule_calendar_event` takes only an event id and two times —
        # no destination argument and no free text — so the destination and
        # laundering rules have nothing to bind to and it was gated by nothing
        # at all despite being a declared sink. What it does have is an effect
        # the user should have asked for: it moves an appointment and mails
        # everyone on it. That makes it a consent question, which is why it is
        # here and not in `body_fields`.
        consent_required=frozenset(
            {"delete_email", "delete_file", "cancel_calendar_event",
             "reschedule_calendar_event"}
        ),
        effect_kinds={
            "delete_email": "delete",
            "delete_file": "delete",
            "cancel_calendar_event": "cancel",
            "reschedule_calendar_event": "schedule",
            "create_calendar_event": "calendar",
        },
        target_exempt_kinds=frozenset({"calendar"}),
        # `list_files` renders one `field: value` per line with `id_` and `size`
        # in every record, so "the largest file" is settleable. `delete_email`
        # is deliberately absent: the inbox listing carries no comparable field,
        # and a tool left out means the rule stands down for it rather than
        # guessing.
        selection={
            "delete_file": ("file_id", "id_", {"size": "size", "recency": "last_modified"}),
        },
    ),
}
