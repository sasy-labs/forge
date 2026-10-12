/* Custom C++ functors for airline_policy.dl (tau2-airline).
 *
 * Content classifiers (keyword/phrase matchers over message text) and
 * a few JSON helpers that reach into the airline payload shapes the
 * captured traces actually carry.  Phrase lists are grounded in the
 * run5 capture: every phrase below either appears in a captured user
 * turn or is a close morphological variant of one.
 */

#include <souffle/SouffleInterface.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return out;
}

bool word_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '\'';
}

/* Substring hit that must not sit inside a longer word — keeps "yes"
 * from matching "yesterday" and "ill" from matching "will". */
bool word_hit(const std::string& hay, size_t pos, size_t len) {
    if (pos > 0 && word_char(hay[pos - 1])) return false;
    size_t end = pos + len;
    if (end < hay.size() && word_char(hay[end])) return false;
    return true;
}

/* 1 iff any phrase occurs.  `whole_word` phrases are additionally
 * required to stand alone. */
bool any_phrase(const std::string& lower, const char* const* phrases,
                bool whole_word) {
    for (const char* const* p = phrases; *p; ++p) {
        size_t len = std::strlen(*p);
        size_t at = 0;
        while ((at = lower.find(*p, at)) != std::string::npos) {
            if (!whole_word || word_hit(lower, at, len)) return true;
            at += 1;
        }
    }
    return false;
}

/* Words that flip the meaning of a confirmation token that follows
 * them ("don't proceed", "not yet", "before we proceed"). */
const char* const kNegators[] = {
    "not ", "n't ", "never ", "cannot ", "can't ", "without ",
    "hold off", "before ", "unless ", "instead of ", "rather than ",
    nullptr,
};

bool negated_at(const std::string& lower, size_t pos) {
    size_t start = pos > 26 ? pos - 26 : 0;
    std::string pre = lower.substr(start, pos - start);
    for (const char* const* n = kNegators; *n; ++n)
        if (pre.find(*n) != std::string::npos) return true;
    return false;
}

/* Explicit consent to proceed with an action the agent just described.
 * Single-token phrases are matched on word boundaries; every hit is
 * rejected when a negator precedes it. */
const char* const kConfirmWord[] = {
    "yes", "yeah", "yep", "proceed", "confirmed", "agreed", nullptr,
};
const char* const kConfirmPhrase[] = {
    "go ahead", "looks good", "sounds good", "that works",
    "i confirm", "do it", "sounds perfect", nullptr,
};

bool confirms(const std::string& lower) {
    for (const char* const* p = kConfirmWord; *p; ++p) {
        size_t len = std::strlen(*p), at = 0;
        while ((at = lower.find(*p, at)) != std::string::npos) {
            if (word_hit(lower, at, len) && !negated_at(lower, at))
                return true;
            at += 1;
        }
    }
    for (const char* const* p = kConfirmPhrase; *p; ++p) {
        size_t len = std::strlen(*p), at = 0;
        while ((at = lower.find(*p, at)) != std::string::npos) {
            size_t end = at + len;
            /* "please do" must not be read out of "please don't". */
            bool trailing_word = end < lower.size() && word_char(lower[end]);
            if (!trailing_word && !negated_at(lower, at)) return true;
            at += 1;
        }
    }
    return false;
}

/* ── Cancellation reasons ───────────────────────────────────────── */

/* Insurance-covered reasons: health, weather, and other significant
 * personal circumstances (schedule conflicts, change of plans). */
const char* const kCoveredPhrase[] = {
    "change of plan", "changed my plan", "plans changed",
    "plans have changed", "change in plans", "schedule conflict",
    "conflicting schedule", "work conflict", "personal reason",
    "personal circumstance", "family emergency", "passed away",
    "medical", "hospital", "surgery", "injury", "injured",
    "food poisoning", "covid", "the flu", "weather", "storm",
    "hurricane", "blizzard", "snowstorm", "flooding", nullptr,
};
const char* const kCoveredWord[] = {
    "sick", "ill", "illness", "health", "emergency", "snowed",
    nullptr,
};

bool covered_reason(const std::string& lower) {
    return any_phrase(lower, kCoveredPhrase, false)
        || any_phrase(lower, kCoveredWord, true);
}

/* The airline cancelled the flight — an eligibility ground of its own,
 * separate from the insurance-covered reasons. */
const char* const kAirlineCancelled[] = {
    "airline cancelled", "airline canceled",
    "flight was cancelled", "flight was canceled",
    "flight got cancelled", "flight got canceled",
    "flight has been cancelled", "flight has been canceled",
    "you cancelled my flight", "you canceled my flight",
    "cancelled by the airline", "canceled by the airline", nullptr,
};

/* Any stated reason at all, including the open "other reasons"
 * category the policy names (no longer needed / booked twice). */
const char* const kOtherReason[] = {
    "no longer need", "no longer want", "don't need the flight",
    "do not need the flight", "won't be able to travel",
    "will not be able to travel", "cannot travel", "can't travel",
    "booked it by mistake", "booked by mistake", "booked twice",
    "double booked", "double-booked", "wrong date", "wrong flight",
    nullptr,
};

/* ── JSON helpers over the airline payloads ─────────────────────── */

/* Bounds of the JSON object that encloses `pos`, or false. */
bool enclosing_object(const std::string& s, size_t pos,
                      size_t& start, size_t& end) {
    int depth = 0;
    size_t i = pos;
    while (true) {
        if (s[i] == '}') ++depth;
        else if (s[i] == '{') {
            if (depth == 0) { start = i; break; }
            --depth;
        }
        if (i == 0) return false;
        --i;
    }
    depth = 0;
    for (i = start; i < s.size(); ++i) {
        if (s[i] == '{') ++depth;
        else if (s[i] == '}') {
            if (--depth == 0) { end = i; return true; }
        }
    }
    return false;
}

/* Value of "key": "..." inside [start, end], or "". */
std::string flat_field(const std::string& s, size_t start, size_t end,
                       const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t at = s.find(needle, start);
    if (at == std::string::npos || at > end) return "";
    size_t q = s.find('"', at + needle.size());
    if (q == std::string::npos || q > end) return "";
    size_t q2 = s.find('"', q + 1);
    if (q2 == std::string::npos || q2 > end) return "";
    return s.substr(q + 1, q2 - q - 1);
}

/* Free checked bags per passenger, by membership and cabin. */
int free_bags_per_pax(const std::string& membership,
                      const std::string& cabin) {
    int base;
    if (cabin == "basic_economy") base = 0;
    else if (cabin == "economy") base = 1;
    else if (cabin == "business") base = 2;
    else return -1;
    if (membership == "regular") return base;
    if (membership == "silver") return base + 1;
    if (membership == "gold") return base + 2;
    return -1;
}

}  // namespace

extern "C" {

/* @user_confirms(text) — 1 iff the message is explicit consent to
 * proceed with the action the agent just described. */
souffle::RamDomain user_confirms(souffle::SymbolTable* st,
                                 souffle::RecordTable*,
                                 souffle::RamDomain text) {
    return confirms(to_lower(st->decode(text))) ? 1 : 0;
}

/* @covered_cancellation_reason(text) — 1 iff the message states a
 * reason travel insurance covers. */
souffle::RamDomain covered_cancellation_reason(souffle::SymbolTable* st,
                                               souffle::RecordTable*,
                                               souffle::RamDomain text) {
    return covered_reason(to_lower(st->decode(text))) ? 1 : 0;
}

/* @airline_cancelled_flight(text) — 1 iff the message says the airline
 * cancelled the flight. */
souffle::RamDomain airline_cancelled_flight(souffle::SymbolTable* st,
                                            souffle::RecordTable*,
                                            souffle::RamDomain text) {
    std::string lower = to_lower(st->decode(text));
    return any_phrase(lower, kAirlineCancelled, false) ? 1 : 0;
}

/* @cancellation_reason_stated(text) — 1 iff the message states any
 * cancellation reason: covered, airline-cancelled, or one of the
 * "other reasons" the policy leaves open. */
souffle::RamDomain cancellation_reason_stated(souffle::SymbolTable* st,
                                              souffle::RecordTable*,
                                              souffle::RamDomain text) {
    std::string lower = to_lower(st->decode(text));
    if (covered_reason(lower)) return 1;
    if (any_phrase(lower, kAirlineCancelled, false)) return 1;
    if (any_phrase(lower, kOtherReason, false)) return 1;
    return 0;
}

/* @user_requested_compensation(text) — 1 iff the user asked for
 * compensation (the agent may never offer it unprompted). */
souffle::RamDomain user_requested_compensation(souffle::SymbolTable* st,
                                               souffle::RecordTable*,
                                               souffle::RamDomain text) {
    static const char* const kPhrase[] = {
        "compensation", "compensate", "voucher", "travel credit",
        "credit for the inconvenience", "make it right",
        "something for the trouble", "certificate for the trouble",
        nullptr,
    };
    std::string lower = to_lower(st->decode(text));
    return any_phrase(lower, kPhrase, false) ? 1 : 0;
}

/* @user_requested_insurance(text) — 1 iff the user asked to buy travel
 * insurance. */
souffle::RamDomain user_requested_insurance(souffle::SymbolTable* st,
                                            souffle::RecordTable*,
                                            souffle::RamDomain text) {
    static const char* const kPhrase[] = {
        "insurance", "insure the trip", "travel protection", nullptr,
    };
    std::string lower = to_lower(st->decode(text));
    if (!any_phrase(lower, kPhrase, false)) return 0;
    static const char* const kDecline[] = {
        "no insurance", "without insurance", "don't want insurance",
        "do not want insurance", "no travel insurance",
        "don't need insurance", "do not need insurance",
        "skip the insurance", nullptr,
    };
    return any_phrase(lower, kDecline, false) ? 0 : 1;
}

/* @user_requested_bags(text) — 1 iff the user asked for checked bags. */
souffle::RamDomain user_requested_bags(souffle::SymbolTable* st,
                                       souffle::RecordTable*,
                                       souffle::RamDomain text) {
    static const char* const kPhrase[] = {
        "checked bag", "check a bag", "check in a bag", "baggage",
        "suitcase", "luggage", nullptr,
    };
    static const char* const kWord[] = {"bag", "bags", nullptr};
    std::string lower = to_lower(st->decode(text));
    if (!(any_phrase(lower, kPhrase, false)
          || any_phrase(lower, kWord, true)))
        return 0;
    static const char* const kNone[] = {
        "no checked bag", "no bags", "no baggage", "no luggage",
        "don't have any bag", "do not have any bag",
        "without any bag", "no suitcase", nullptr,
    };
    return any_phrase(lower, kNone, false) ? 0 : 1;
}

/* @count_payment_prefix(args, prefix) — how many entries of the
 * `payment_methods` array carry a payment_id with this prefix
 * (payment ids are "credit_card_…", "gift_card_…", "certificate_…"). */
souffle::RamDomain count_payment_prefix(souffle::SymbolTable* st,
                                        souffle::RecordTable*,
                                        souffle::RamDomain args,
                                        souffle::RamDomain prefix) {
    const std::string& s = st->decode(args);
    const std::string& pre = st->decode(prefix);
    souffle::RamDomain n = 0;
    const std::string needle = "\"payment_id\"";
    size_t at = 0;
    while ((at = s.find(needle, at)) != std::string::npos) {
        size_t q = s.find('"', at + needle.size());
        if (q == std::string::npos) break;
        size_t q2 = s.find('"', q + 1);
        if (q2 == std::string::npos) break;
        if (s.compare(q + 1, pre.size(), pre) == 0) ++n;
        at = q2 + 1;
    }
    return n;
}

/* @flight_leg_field(payload, flight_number, field) — the named field of
 * the flight object carrying this flight number, anywhere in the
 * payload (a reservation's `flights` or a search result), or "".
 * Reservation legs and search results are flat objects, so the field
 * is read from the object enclosing the flight number. */
souffle::RamDomain flight_leg_field(souffle::SymbolTable* st,
                                    souffle::RecordTable*,
                                    souffle::RamDomain payload,
                                    souffle::RamDomain flight_number,
                                    souffle::RamDomain field) {
    const std::string& s = st->decode(payload);
    const std::string& num = st->decode(flight_number);
    const std::string& key = st->decode(field);
    if (num.empty()) return st->encode("");

    const std::string needle = "\"flight_number\"";
    size_t at = 0;
    while ((at = s.find(needle, at)) != std::string::npos) {
        size_t q = s.find('"', at + needle.size());
        if (q == std::string::npos) break;
        size_t q2 = s.find('"', q + 1);
        if (q2 == std::string::npos) break;
        if (s.compare(q + 1, q2 - q - 1, num) == 0) {
            size_t start = 0, end = 0;
            if (enclosing_object(s, at, start, end)) {
                std::string v = flat_field(s, start, end, key);
                if (!v.empty()) return st->encode(v);
            }
        }
        at = q2 + 1;
    }
    return st->encode("");
}

/* @any_flight_before(payload, cutoff_date) — 1 iff any leg of the
 * reservation departs strictly before `cutoff_date` (ISO dates compare
 * lexicographically), i.e. has already been flown. */
souffle::RamDomain any_flight_before(souffle::SymbolTable* st,
                                     souffle::RecordTable*,
                                     souffle::RamDomain payload,
                                     souffle::RamDomain cutoff) {
    const std::string& s = st->decode(payload);
    const std::string& cut = st->decode(cutoff);
    const std::string needle = "\"date\"";
    size_t at = 0;
    while ((at = s.find(needle, at)) != std::string::npos) {
        size_t q = s.find('"', at + needle.size());
        if (q == std::string::npos) break;
        size_t q2 = s.find('"', q + 1);
        if (q2 == std::string::npos) break;
        std::string date = s.substr(q + 1, q2 - q - 1);
        if (date.size() == 10 && date < cut) return 1;
        at = q2 + 1;
    }
    return 0;
}

/* @expected_paid_bags(membership, cabin, passengers, total) — how many
 * of `total` checked bags are chargeable, given the free allowance for
 * this membership and cabin.  -1 when the allowance is unknown, so the
 * caller can skip the check rather than guess. */
souffle::RamDomain expected_paid_bags(souffle::SymbolTable* st,
                                      souffle::RecordTable*,
                                      souffle::RamDomain membership,
                                      souffle::RamDomain cabin,
                                      souffle::RamDomain passengers,
                                      souffle::RamDomain total) {
    int per_pax = free_bags_per_pax(to_lower(st->decode(membership)),
                                   to_lower(st->decode(cabin)));
    if (per_pax < 0) return -1;
    int npax = static_cast<int>(passengers);
    int bags = static_cast<int>(total);
    if (npax <= 0 || bags < 0) return -1;
    int paid = bags - per_pax * npax;
    return paid > 0 ? paid : 0;
}

/* @json_array_count(payload, path) — length of the array at `path`, as a
 * signed number so it can be compared with @json_get_int results and
 * passed to @expected_paid_bags.  0 on miss or non-array. */
souffle::RamDomain json_array_count(souffle::SymbolTable* st,
                                    souffle::RecordTable*,
                                    souffle::RamDomain payload,
                                    souffle::RamDomain path) {
    const std::string& s = st->decode(payload);
    const std::string& key = st->decode(path);
    size_t at = s.find("\"" + key + "\"");
    if (at == std::string::npos) return 0;
    size_t i = s.find('[', at);
    if (i == std::string::npos) return 0;
    /* The key must be followed by the array, not by a later one. */
    for (size_t j = at + key.size() + 2; j < i; ++j)
        if (!std::isspace(static_cast<unsigned char>(s[j])) && s[j] != ':')
            return 0;
    int depth = 0;
    bool in_str = false, esc = false, seen_elem = false;
    souffle::RamDomain n = 0;
    for (; i < s.size(); ++i) {
        char c = s[i];
        if (esc) { esc = false; continue; }
        if (c == '\\') { esc = true; continue; }
        if (c == '"') { in_str = !in_str; if (depth == 1) seen_elem = true; continue; }
        if (in_str) continue;
        if (c == '[' || c == '{') {
            ++depth;
            if (depth == 2) seen_elem = true;
        } else if (c == ']' || c == '}') {
            --depth;
            if (depth == 0) { if (seen_elem) ++n; break; }
        } else if (c == ',' && depth == 1) {
            ++n;
            seen_elem = false;
        } else if (depth == 1
                   && !std::isspace(static_cast<unsigned char>(c))) {
            seen_elem = true;
        }
    }
    return n;
}

}  // extern "C"
