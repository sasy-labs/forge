/*
    Companion C++ functors for airline_policy.dl (tau2 airline domain).

    Two families:

      1. Content matchers over customer turns (user_affirms,
         states_cancel_reason, covered_cancel_reason,
         requests_compensation).  Deterministic word-boundary phrase
         matching — no LLM.  Every phrase below was taken from text
         that actually appears in outputs/variant-set/run2/scenarios.json
         or is a direct restatement of the spec's own wording.

      2. Structured readers over tool payloads (flight_list_sig,
         any_flight_before, res_passenger_count, count_payment_kind,
         payments_in_profile).  These reuse the JSON accessors that
         functors_common.cpp already exports rather than re-parsing
         JSON by hand; the harness compiles both files into one
         library, so the symbols resolve at link time.
*/

#include <souffle/SouffleInterface.h>

#include <algorithm>
#include <cctype>
#include <string>

/* ── Reused accessors from functors_common.cpp ─────────────────── */

extern "C" souffle::RamDomain json_array_len(
    souffle::SymbolTable*, souffle::RecordTable*,
    souffle::RamDomain, souffle::RamDomain);

extern "C" souffle::RamDomain json_array_get_field_str(
    souffle::SymbolTable*, souffle::RecordTable*,
    souffle::RamDomain, souffle::RamDomain, souffle::RamDomain,
    souffle::RamDomain);

extern "C" souffle::RamDomain json_object_has_key(
    souffle::SymbolTable*, souffle::RecordTable*,
    souffle::RamDomain, souffle::RamDomain, souffle::RamDomain);

namespace {

std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return out;
}

bool is_word_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

// Substring match that will not fire inside a longer word, so "yes"
// does not match "eyes" or "yesterday". Phrases containing spaces or
// punctuation work the same way — only the outer edges are checked.
bool contains_phrase(const std::string& hay, const std::string& needle) {
    if (needle.empty()) return false;
    size_t pos = hay.find(needle);
    while (pos != std::string::npos) {
        bool left_ok = pos == 0 || !is_word_char(hay[pos - 1]) ||
                       !is_word_char(needle.front());
        size_t end = pos + needle.size();
        bool right_ok = end >= hay.size() || !is_word_char(hay[end]) ||
                        !is_word_char(needle.back());
        if (left_ok && right_ok) return true;
        pos = hay.find(needle, pos + 1);
    }
    return false;
}

bool any_phrase(const std::string& lowered, const char* const* phrases) {
    for (const char* const* p = phrases; *p; ++p) {
        if (contains_phrase(lowered, *p)) return true;
    }
    return false;
}

// Helper wrappers so the reused accessors read like plain calls.
unsigned array_len(souffle::SymbolTable* st, const std::string& json,
                   const char* path) {
    souffle::RamDomain n = json_array_len(
        st, nullptr, st->encode(json), st->encode(path));
    return static_cast<unsigned>(n);
}

std::string array_field(souffle::SymbolTable* st, const std::string& json,
                        const char* path, unsigned idx, const char* field) {
    souffle::RamDomain sym = json_array_get_field_str(
        st, nullptr, st->encode(json), st->encode(path),
        static_cast<souffle::RamDomain>(idx), st->encode(field));
    return st->decode(sym);
}

/* ── Phrase tables ─────────────────────────────────────────────── */

// Explicit go-ahead. Grounded in the confirmations the captured runs
// actually contain, e.g. "Yes, please go ahead with all those
// changes.", "Yes, that's fine—please go ahead...".
const char* const kAffirm[] = {
    "yes", "yeah", "yep", "yes please", "go ahead", "please go ahead",
    "please proceed", "proceed with", "i confirm", "confirmed",
    "that's correct", "thats correct", "sounds good", "that works",
    "do it", "let's do it", "lets do it", "please book it",
    "please cancel it", "book it", "sign me up", nullptr,
};

// Withdrawals of consent. Any of these vetoes an affirmative token in
// the same turn, so "no, not yet" cannot be read as a go-ahead.
const char* const kVeto[] = {
    "not yet", "hold off", "don't proceed", "do not proceed",
    "don't go ahead", "do not go ahead", "no thanks", "no thank you",
    "nope", "never mind", "nevermind", "cancel that request", nullptr,
};

// Any stated reason for cancelling — the spec requires the agent to
// obtain one (change of plan / airline cancelled / other).
const char* const kCancelReason[] = {
    "change of plan", "change of plans", "changed my plan",
    "changed my plans", "plans changed", "no longer need",
    "no longer needed", "don't need", "do not need", "not going",
    "can't make it", "cannot make it", "cancelled by the airline",
    "canceled by the airline", "airline cancelled", "airline canceled",
    "flight was cancelled", "flight was canceled", "mistake",
    "by accident", "accidentally", "sick", "ill", "illness", "health",
    "medical", "hospital", "weather", "storm", "hurricane", "snow",
    "emergency", "family", "funeral", "work conflict",
    "schedule conflict", "scheduling conflict", "double booked",
    "double-booked", "other reasons", nullptr,
};

// Insurance-covered reasons only: health, weather, and other
// significant personal circumstances (schedule conflicts, change of
// plans) — the spec's own list.
const char* const kCoveredReason[] = {
    "sick", "ill", "illness", "health", "medical", "hospital",
    "injury", "injured", "weather", "storm", "hurricane", "snow",
    "blizzard", "emergency", "family emergency", "funeral",
    "schedule conflict", "scheduling conflict", "work conflict",
    "change of plan", "change of plans", "changed my plans",
    "personal circumstances", nullptr,
};

// The customer asking for compensation. The spec forbids offering it
// proactively, so the request has to be in the customer's own words.
const char* const kCompRequest[] = {
    "compensation", "compensate", "compensated", "certificate",
    "voucher", "reimburse", "reimbursement", "make it right",
    "what can you offer", "anything you can offer",
    "something for the trouble", "for the inconvenience",
    "for my trouble", nullptr,
};

}  // namespace

extern "C" {

/*
 * @user_affirms(text) -> 1 when the turn is an explicit go-ahead.
 */
souffle::RamDomain user_affirms(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string lowered = to_lower(st->decode(text_sym));
    if (any_phrase(lowered, kVeto)) return 0;
    return any_phrase(lowered, kAffirm) ? 1 : 0;
}

/*
 * @states_cancel_reason(text) -> 1 when the turn states any reason
 * for cancelling.
 */
souffle::RamDomain states_cancel_reason(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string lowered = to_lower(st->decode(text_sym));
    return any_phrase(lowered, kCancelReason) ? 1 : 0;
}

/*
 * @covered_cancel_reason(text) -> 1 when the stated reason is one
 * travel insurance covers.
 */
souffle::RamDomain covered_cancel_reason(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string lowered = to_lower(st->decode(text_sym));
    return any_phrase(lowered, kCoveredReason) ? 1 : 0;
}

/*
 * @requests_compensation(text) -> 1 when the customer asks for
 * compensation.
 */
souffle::RamDomain requests_compensation(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string lowered = to_lower(st->decode(text_sym));
    return any_phrase(lowered, kCompRequest) ? 1 : 0;
}

/*
 * @flight_list_sig(json) -> "FLIGHT|DATE;FLIGHT|DATE;..." over the
 * payload's "flights" array, or "" when there is none.
 *
 * Both a tool-call argument list and a stored reservation carry the
 * same two identifying fields per segment (the reservation also
 * carries origin / destination / price, which are ignored), so the
 * two signatures are comparable: equal signature == same flights,
 * i.e. a cabin-only change.
 */
souffle::RamDomain flight_list_sig(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain json_sym)
{
    const std::string json = st->decode(json_sym);
    unsigned n = array_len(st, json, "flights");
    std::string sig;
    for (unsigned i = 0; i < n; ++i) {
        sig += array_field(st, json, "flights", i, "flight_number");
        sig += "|";
        sig += array_field(st, json, "flights", i, "date");
        sig += ";";
    }
    return st->encode(sig);
}

/*
 * @any_flight_before(json, iso_date) -> 1 when any segment in the
 * payload's "flights" array departs strictly before iso_date.
 * Dates are "YYYY-MM-DD", so a lexicographic compare is a date
 * compare. Segments with no date are ignored.
 */
souffle::RamDomain any_flight_before(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain json_sym, souffle::RamDomain date_sym)
{
    const std::string json = st->decode(json_sym);
    const std::string cutoff = st->decode(date_sym);
    if (cutoff.empty()) return 0;
    unsigned n = array_len(st, json, "flights");
    for (unsigned i = 0; i < n; ++i) {
        std::string date = array_field(st, json, "flights", i, "date");
        if (date.empty()) continue;
        if (date < cutoff) return 1;
    }
    return 0;
}

/*
 * @flight_list_has(json, flight_number) -> 1 when the payload's
 * "flights" array contains that flight number. Used to tie a
 * get_flight_status answer to the reservation it was asked about.
 */
souffle::RamDomain flight_list_has(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain json_sym, souffle::RamDomain number_sym)
{
    const std::string json = st->decode(json_sym);
    const std::string wanted = st->decode(number_sym);
    if (wanted.empty()) return 0;
    unsigned n = array_len(st, json, "flights");
    for (unsigned i = 0; i < n; ++i) {
        if (array_field(st, json, "flights", i, "flight_number") == wanted) {
            return 1;
        }
    }
    return 0;
}

/*
 * @res_passenger_count(json) -> number of entries in "passengers",
 * as a signed number so it can be multiplied by a per-passenger rate.
 */
souffle::RamDomain res_passenger_count(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain json_sym)
{
    const std::string json = st->decode(json_sym);
    return static_cast<souffle::RamDomain>(
        array_len(st, json, "passengers"));
}

/*
 * @count_payment_kind(args, prefix) -> how many entries of the
 * booking's "payment_methods" array have a payment_id starting with
 * prefix ("certificate", "credit_card", "gift_card").
 */
souffle::RamDomain count_payment_kind(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain args_sym, souffle::RamDomain prefix_sym)
{
    const std::string args = st->decode(args_sym);
    const std::string prefix = st->decode(prefix_sym);
    unsigned n = array_len(st, args, "payment_methods");
    unsigned hits = 0;
    for (unsigned i = 0; i < n; ++i) {
        std::string id =
            array_field(st, args, "payment_methods", i, "payment_id");
        if (id.compare(0, prefix.size(), prefix) == 0) ++hits;
    }
    return static_cast<souffle::RamDomain>(hits);
}

/*
 * @payments_in_profile(args, profile) -> 1 when every payment_id the
 * booking uses is a key of the profile's "payment_methods" object.
 * An empty payment list vacuously passes; the API rejects that case
 * on its own.
 */
souffle::RamDomain payments_in_profile(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain args_sym, souffle::RamDomain profile_sym)
{
    const std::string args = st->decode(args_sym);
    const std::string profile = st->decode(profile_sym);
    unsigned n = array_len(st, args, "payment_methods");
    for (unsigned i = 0; i < n; ++i) {
        std::string id =
            array_field(st, args, "payment_methods", i, "payment_id");
        if (id.empty()) return 0;
        souffle::RamDomain has = json_object_has_key(
            st, nullptr, st->encode(profile),
            st->encode("payment_methods"), st->encode(id));
        if (has == 0) return 0;
    }
    return 1;
}

}  // extern "C"
