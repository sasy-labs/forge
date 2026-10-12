/*
 * Custom Soufflé functors for the tau2-airline policy
 * (outputs/variant-set/run3/airline_policy.dl).
 *
 * Three groups:
 *   1. Content checks over conversation text — explicit user
 *      confirmation, stated cancellation reason, insurance-covered
 *      reason, an explicit compensation request. Keyword tables are
 *      derived from what users actually said in the captured traces
 *      (outputs/variant-set/run3/scenarios.json), not from guesswork.
 *   2. The checked-bag allowance table from the policy spec.
 *   3. Small helpers Datalog cannot express directly: ISO-date
 *      ordering and a signed array length.
 *
 * Matching is case-insensitive, word-boundary aware (so "yes" does
 * not fire inside "eyes"), and normalizes the Unicode right single
 * quote U+2019 to ASCII "'" because the trace text uses it
 * throughout ("that’s correct", "I’d like").
 */

#include <souffle/SouffleInterface.h>

#include <cctype>
#include <cstring>
#include <string>

namespace {

/* Lowercase and fold U+2019 / U+2018 to ASCII apostrophe. */
std::string normalize(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(in[i]);
        /* U+2018 = E2 80 98, U+2019 = E2 80 99 */
        if (c == 0xE2 && i + 2 < in.size()
            && static_cast<unsigned char>(in[i + 1]) == 0x80
            && (static_cast<unsigned char>(in[i + 2]) == 0x98
                || static_cast<unsigned char>(in[i + 2]) == 0x99)) {
            out.push_back('\'');
            i += 2;
            continue;
        }
        out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

/* True iff `phrase` occurs in `text` delimited by non-alphanumerics.
   Bytes >= 0x80 are not alphanumeric in the C locale, so a
   multi-byte character counts as a boundary — which is what we
   want for text like "bien sûr!". */
bool has_phrase(const std::string& text, const char* phrase) {
    const size_t plen = std::strlen(phrase);
    if (plen == 0) return false;
    size_t pos = 0;
    while ((pos = text.find(phrase, pos)) != std::string::npos) {
        const bool left_ok =
            (pos == 0)
            || !std::isalnum(static_cast<unsigned char>(text[pos - 1]));
        const size_t end = pos + plen;
        const bool right_ok =
            (end >= text.size())
            || !std::isalnum(static_cast<unsigned char>(text[end]));
        if (left_ok && right_ok) return true;
        pos += 1;
    }
    return false;
}

bool has_any(const std::string& text, const char* const* table) {
    for (const char* const* p = table; *p; ++p)
        if (has_phrase(text, *p)) return true;
    return false;
}

/* Explicit affirmative confirmation — the spec's "explicit user
   confirmation (yes)". Deliberately excludes bare instructions
   ("please book", "go ahead and cancel"): an instruction to act is
   not a confirmation of details the agent listed back. */
const char* const kConfirm[] = {
    "yes", "yeah", "yep", "yup", "correct", "confirmed", "i confirm",
    "sounds good", "looks good", "that works", "agreed", "i agree",
    "absolutely", "affirmative", "okay", "ok", "please proceed",
    "do it", "perfect", nullptr,
};

/* A stated reason for cancelling. The spec allows "change of plan,
   airline cancelled flight, or other reasons", so this is broad on
   purpose: the requirement is that the user gave *a* reason, not
   that the reason falls in a narrow set. */
const char* const kCancelReason[] = {
    "change of plan", "change of plans", "changed my plan",
    "changed my plans", "plans changed", "change my plan",
    "no longer need", "no longer able", "don't need", "do not need",
    "by mistake", "made a mistake", "mistake", "wrong flight",
    "booked the wrong", "airline cancelled", "airline canceled",
    "flight was cancelled", "flight was canceled",
    "cancelled by the airline", "canceled by the airline",
    "sick", "illness", "ill", "health", "medical", "hospital",
    "injury", "emergency", "weather", "storm", "snow", "hurricane",
    "schedule conflict", "conflict", "can't travel", "cannot travel",
    "unable to travel", "won't be able", "not going", "rebook",
    "reschedule", "personal reasons", "family", nullptr,
};

/* Reasons the travel insurance covers: health-related,
   weather-related, and other significant personal circumstances
   (schedule conflicts, change of plans). */
const char* const kCoveredReason[] = {
    "health", "sick", "illness", "ill", "medical", "hospital",
    "injury", "emergency", "weather", "storm", "snow", "hurricane",
    "schedule conflict", "conflict", "change of plan",
    "change of plans", "plans changed", "personal", nullptr,
};

/* The user asking for compensation. The spec forbids offering it
   proactively, so the certificate needs an explicit ask on record. */
const char* const kCompensation[] = {
    "compensation", "compensate", "certificate", "voucher", "refund",
    "reimburse", "credit", "make it right", "goodwill", nullptr,
};

/* Free checked bags per passenger, by membership level and cabin. */
int free_bags(const std::string& membership, const std::string& cabin) {
    int base;
    if (membership == "regular")     base = 0;
    else if (membership == "silver") base = 1;
    else if (membership == "gold")   base = 2;
    else return -1;  /* unknown membership — caller denies */

    if (cabin == "basic_economy") return base + 0;
    if (cabin == "economy")       return base + 1;
    if (cabin == "business")      return base + 2;
    return -1;  /* unknown cabin — caller denies */
}

/* "2024-05-17" or "2024-05-17T06:00:00" -> 20240517, or -1. */
long iso_day(const std::string& s) {
    if (s.size() < 10) return -1;
    long v = 0;
    for (size_t i = 0; i < 10; ++i) {
        char c = s[i];
        if (i == 4 || i == 7) {
            if (c != '-') return -1;
            continue;
        }
        if (c < '0' || c > '9') return -1;
        v = v * 10 + (c - '0');
    }
    return v;
}

}  // namespace

extern "C" {

/* Shared JSON array length, re-exposed with a signed Datalog type so
   passenger/flight counts can be used in arithmetic without casts. */
souffle::RamDomain json_array_len(
    souffle::SymbolTable*, souffle::RecordTable*,
    souffle::RamDomain, souffle::RamDomain);

souffle::RamDomain airline_json_count(
    souffle::SymbolTable* st, souffle::RecordTable* rt,
    souffle::RamDomain json_sym, souffle::RamDomain path_sym)
{
    return json_array_len(st, rt, json_sym, path_sym);
}

souffle::RamDomain airline_user_confirms(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    return has_any(normalize(st->decode(text_sym)), kConfirm) ? 1 : 0;
}

souffle::RamDomain airline_cancel_reason(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    return has_any(normalize(st->decode(text_sym)), kCancelReason) ? 1 : 0;
}

souffle::RamDomain airline_covered_reason(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    return has_any(normalize(st->decode(text_sym)), kCoveredReason) ? 1 : 0;
}

souffle::RamDomain airline_requests_compensation(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    return has_any(normalize(st->decode(text_sym)), kCompensation) ? 1 : 0;
}

/* Free checked bags for the whole reservation, or -1 if membership
   or cabin is unrecognized. */
souffle::RamDomain airline_free_bag_allowance(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain membership_sym, souffle::RamDomain cabin_sym,
    souffle::RamDomain npax)
{
    const int per_pax =
        free_bags(st->decode(membership_sym), st->decode(cabin_sym));
    const int n = static_cast<int>(npax);
    if (per_pax < 0 || n < 0) return -1;
    return static_cast<souffle::RamDomain>(per_pax * n);
}

/* The nonfree_baggages the reservation should carry: everything
   above the free allowance, floored at zero. -1 when the allowance
   is unknown, so the caller can deny rather than compare to junk. */
souffle::RamDomain airline_nonfree_expected(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain membership_sym, souffle::RamDomain cabin_sym,
    souffle::RamDomain npax, souffle::RamDomain total)
{
    const int per_pax =
        free_bags(st->decode(membership_sym), st->decode(cabin_sym));
    const int n = static_cast<int>(npax);
    const int tot = static_cast<int>(total);
    if (per_pax < 0 || n < 0 || tot < 0) return -1;
    const int excess = tot - (per_pax * n);
    return static_cast<souffle::RamDomain>(excess > 0 ? excess : 0);
}

/* 1 iff ISO date/datetime d1 falls on a strictly earlier day than
   d2. 0 when either side does not parse — an unreadable date must
   not silently read as "in the past". */
souffle::RamDomain airline_date_before(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain d1_sym, souffle::RamDomain d2_sym)
{
    const long a = iso_day(st->decode(d1_sym));
    const long b = iso_day(st->decode(d2_sym));
    if (a < 0 || b < 0) return 0;
    return (a < b) ? 1 : 0;
}

}  // extern "C"
