// Companion functors for airline_policy.dl (tau2 airline domain).
//
// Every phrase table below is grounded in the captured decision points at
// outputs/variant-set/run3/scenarios.json — the wording comes from what the
// simulated end user actually said in those traces, plus the vocabulary the
// spec itself enumerates (cancellation reasons, insurance-covered reasons).
//
// JSON access reuses the shared parser exported by functors_common.cpp
// rather than re-implementing one: both translation units are compiled into
// the same library, so the forward declarations below resolve at link time.

#include <souffle/SouffleInterface.h>

#include <algorithm>
#include <cctype>
#include <string>

// ── Shared JSON accessors from functors_common.cpp ──────────────────────
extern "C" {
souffle::RamDomain json_array_len(
    souffle::SymbolTable*, souffle::RecordTable*,
    souffle::RamDomain, souffle::RamDomain);
souffle::RamDomain json_array_get_field_str(
    souffle::SymbolTable*, souffle::RecordTable*,
    souffle::RamDomain, souffle::RamDomain,
    souffle::RamDomain, souffle::RamDomain);
souffle::RamDomain json_object_has_key(
    souffle::SymbolTable*, souffle::RecordTable*,
    souffle::RamDomain, souffle::RamDomain, souffle::RamDomain);
}

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
    unsigned char u = static_cast<unsigned char>(c);
    return std::isalnum(u) != 0 || c == '_';
}

// Substring match that will not fire inside a longer word. "confirm" must
// not match "confirmation number", and "ok" must not match "booking".
bool contains_word(const std::string& hay, const std::string& needle) {
    if (needle.empty()) return false;
    size_t pos = 0;
    while ((pos = hay.find(needle, pos)) != std::string::npos) {
        bool left_ok = (pos == 0) || !is_word_char(hay[pos - 1]);
        size_t end = pos + needle.size();
        bool right_ok = (end >= hay.size()) || !is_word_char(hay[end]);
        if (left_ok && right_ok) return true;
        ++pos;
    }
    return false;
}

bool any_word(const std::string& lower, const char* const* phrases) {
    for (const char* const* p = phrases; *p; ++p)
        if (contains_word(lower, *p)) return true;
    return false;
}

// ── Thin wrappers over the shared JSON parser ───────────────────────────

int arr_len(souffle::SymbolTable* st, souffle::RecordTable* rt,
            const std::string& json, const std::string& path) {
    return static_cast<int>(json_array_len(
        st, rt, st->encode(json), st->encode(path)));
}

std::string arr_field(souffle::SymbolTable* st, souffle::RecordTable* rt,
                      const std::string& json, const std::string& path,
                      int idx, const std::string& field) {
    return st->decode(json_array_get_field_str(
        st, rt, st->encode(json), st->encode(path),
        static_cast<souffle::RamDomain>(idx), st->encode(field)));
}

bool obj_has_key(souffle::SymbolTable* st, souffle::RecordTable* rt,
                 const std::string& json, const std::string& path,
                 const std::string& key) {
    return json_object_has_key(st, rt, st->encode(json),
                               st->encode(path), st->encode(key)) != 0;
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() &&
           s.compare(0, prefix.size(), prefix) == 0;
}

// ── Phrase tables ───────────────────────────────────────────────────────

// Explicit affirmative confirmation. Deliberately EXCLUDES "go ahead",
// "sure" and "please" — in the captured traces those appear in turns that
// issue a fresh instruction rather than confirming a listed plan (see the
// book_reservation decision point the oracle denied, whose latest user turn
// reads "please go ahead and cancel the entire reservation ... let's go with
// the JFK-SEA option").
const char* const kAffirm[] = {
    "yes", "yeah", "yep", "yup",
    "sounds good", "looks good", "that works", "that's fine",
    "thats fine", "i confirm", "confirmed", "agreed",
    "please proceed", "proceed with", "go ahead and proceed",
    "do it", nullptr,
};

// A stated reason for cancelling. The first three groups are the three
// reason kinds the spec enumerates (change of plan / airline cancelled /
// other reasons); the rest are the ways users phrase "other reasons".
const char* const kCancelReason[] = {
    "change of plan", "change of plans", "changed my plan",
    "changed my plans", "plans changed", "plans have changed",
    "change in plans", "no longer need", "no longer needed",
    "don't need", "dont need", "do not need", "won't be able",
    "wont be able", "can't make", "cant make", "cannot make",
    "airline cancelled", "airline canceled", "flight was cancelled",
    "flight was canceled", "flight got cancelled", "flight got canceled",
    "cancelled by the airline", "canceled by the airline",
    "sick", "illness", "ill", "health", "medical", "hospital",
    "emergency", "family", "bereavement", "weather", "storm",
    "hurricane", "snow", "schedule conflict", "scheduling conflict",
    "work conflict", "mistake", "by accident", "accidentally",
    "wrong date", "wrong flight", "booked the wrong",
    "reason for cancel", "reason is", "because", nullptr,
};

// Reasons the travel insurance covers: health-related, weather-related, and
// other significant personal circumstances (schedule conflicts, change of
// plans) — verbatim from the spec's "Travel insurance" section.
const char* const kCoveredReason[] = {
    "sick", "illness", "ill", "health", "medical", "hospital",
    "doctor", "injury", "injured", "emergency", "family emergency",
    "bereavement", "death", "weather", "storm", "hurricane", "snow",
    "flood", "schedule conflict", "scheduling conflict",
    "work conflict", "change of plan", "change of plans",
    "changed my plan", "changed my plans", "plans changed",
    "plans have changed", "personal circumstances", nullptr,
};

// The user asserting that the airline cancelled a flight. This is the
// spec's own channel for the "flight is cancelled by airline" cancellation
// condition — it is listed as one of the reasons the user provides.
const char* const kAirlineCancelled[] = {
    "airline cancelled", "airline canceled",
    "flight was cancelled", "flight was canceled",
    "flight got cancelled", "flight got canceled",
    "cancelled by the airline", "canceled by the airline",
    "cancelled my flight", "canceled my flight",
    "cancelled the flight", "canceled the flight",
    "you cancelled", "you canceled", nullptr,
};

// The user asking for compensation. "Do not proactively offer a
// compensation unless the user explicitly asks for one."
const char* const kCompensationAsk[] = {
    "compensation", "compensate", "certificate", "voucher",
    "credit", "reimburse", "reimbursement", "refund",
    "make it right", "make this right", nullptr,
};

const char* const kInsurance[] = {
    "insurance", "insured", "insure", nullptr,
};

}  // namespace

extern "C" {

// 1 iff `text` carries an explicit affirmative confirmation.
souffle::RamDomain user_affirms(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym) {
    return any_word(to_lower(st->decode(text_sym)), kAffirm) ? 1 : 0;
}

// 1 iff `text` names EVERY flight number in `args`.flights (and there is at
// least one). This is the "the user specified the action themselves" branch
// of the confirmation requirement.
souffle::RamDomain names_all_flights(
    souffle::SymbolTable* st, souffle::RecordTable* rt,
    souffle::RamDomain args_sym, souffle::RamDomain text_sym) {
    const std::string args = st->decode(args_sym);
    const std::string text = to_lower(st->decode(text_sym));
    int n = arr_len(st, rt, args, "flights");
    if (n <= 0) return 0;
    for (int i = 0; i < n; ++i) {
        std::string fn = arr_field(st, rt, args, "flights", i,
                                   "flight_number");
        if (fn.empty()) return 0;
        if (text.find(to_lower(fn)) == std::string::npos) return 0;
    }
    return 1;
}

// 1 iff the flights requested in `args` differ from the reservation's
// current flights — different count, flight number, or date. A call that
// leaves them identical is a cabin-only change.
souffle::RamDomain flights_differ(
    souffle::SymbolTable* st, souffle::RecordTable* rt,
    souffle::RamDomain args_sym, souffle::RamDomain res_sym) {
    const std::string args = st->decode(args_sym);
    const std::string res = st->decode(res_sym);
    int n = arr_len(st, rt, args, "flights");
    int m = arr_len(st, rt, res, "flights");
    if (n != m) return 1;
    for (int i = 0; i < n; ++i) {
        if (arr_field(st, rt, args, "flights", i, "flight_number") !=
            arr_field(st, rt, res, "flights", i, "flight_number"))
            return 1;
        if (arr_field(st, rt, args, "flights", i, "date") !=
            arr_field(st, rt, res, "flights", i, "date"))
            return 1;
    }
    return 0;
}

// 1 iff the reservation payload contains a flight with this flight number.
souffle::RamDomain reservation_has_flight(
    souffle::SymbolTable* st, souffle::RecordTable* rt,
    souffle::RamDomain res_sym, souffle::RamDomain fn_sym) {
    const std::string res = st->decode(res_sym);
    const std::string fn = st->decode(fn_sym);
    if (fn.empty()) return 0;
    int n = arr_len(st, rt, res, "flights");
    for (int i = 0; i < n; ++i)
        if (arr_field(st, rt, res, "flights", i, "flight_number") == fn)
            return 1;
    return 0;
}

// 1 iff every payment id in `args`.payment_methods is a key of the user
// profile's payment_methods object (which is an object keyed by payment id,
// not an array — see K5). 0 when the list is empty or any id is unknown.
souffle::RamDomain payments_in_profile(
    souffle::SymbolTable* st, souffle::RecordTable* rt,
    souffle::RamDomain args_sym, souffle::RamDomain profile_sym) {
    const std::string args = st->decode(args_sym);
    const std::string profile = st->decode(profile_sym);
    int n = arr_len(st, rt, args, "payment_methods");
    if (n <= 0) return 0;
    for (int i = 0; i < n; ++i) {
        std::string pid = arr_field(st, rt, args, "payment_methods", i,
                                    "payment_id");
        if (pid.empty()) return 0;
        if (!obj_has_key(st, rt, profile, "payment_methods", pid))
            return 0;
    }
    return 1;
}

// How many entries of `args`.payment_methods have a payment id starting
// with `prefix` ("certificate_", "credit_card_", "gift_card_").
souffle::RamDomain payment_kind_count(
    souffle::SymbolTable* st, souffle::RecordTable* rt,
    souffle::RamDomain args_sym, souffle::RamDomain prefix_sym) {
    const std::string args = st->decode(args_sym);
    const std::string prefix = st->decode(prefix_sym);
    int n = arr_len(st, rt, args, "payment_methods");
    souffle::RamDomain count = 0;
    for (int i = 0; i < n; ++i) {
        std::string pid = arr_field(st, rt, args, "payment_methods", i,
                                    "payment_id");
        if (starts_with(pid, prefix)) ++count;
    }
    return count;
}

// Free checked bags PER PASSENGER for a membership level and cabin, from
// the spec's "Checked bag allowance" table. -1 when either value is not one
// of the domain's enumerated values, so the caller can skip the check
// instead of guessing.
souffle::RamDomain free_bag_allowance(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain membership_sym, souffle::RamDomain cabin_sym) {
    const std::string m = to_lower(st->decode(membership_sym));
    const std::string c = to_lower(st->decode(cabin_sym));
    int base;
    if (m == "regular") base = 0;
    else if (m == "silver") base = 1;
    else if (m == "gold") base = 2;
    else return -1;
    int bump;
    if (c == "basic_economy") bump = 0;
    else if (c == "economy") bump = 1;
    else if (c == "business") bump = 2;
    else return -1;
    return static_cast<souffle::RamDomain>(base + bump);
}

souffle::RamDomain states_cancellation_reason(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym) {
    return any_word(to_lower(st->decode(text_sym)), kCancelReason) ? 1 : 0;
}

souffle::RamDomain states_covered_reason(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym) {
    return any_word(to_lower(st->decode(text_sym)), kCoveredReason) ? 1 : 0;
}

souffle::RamDomain states_airline_cancelled(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym) {
    return any_word(to_lower(st->decode(text_sym)), kAirlineCancelled)
               ? 1 : 0;
}

souffle::RamDomain requests_compensation(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym) {
    return any_word(to_lower(st->decode(text_sym)), kCompensationAsk)
               ? 1 : 0;
}

souffle::RamDomain mentions_insurance(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym) {
    return any_word(to_lower(st->decode(text_sym)), kInsurance) ? 1 : 0;
}

// Number of tool calls the current message proposes. The policy uses this
// for the spec's "only make one tool call at a time" rule; `tools_json` is
// a JSON array of {"name": ..., "arguments": ...} objects.
souffle::RamDomain tool_call_count(
    souffle::SymbolTable* st, souffle::RecordTable* rt,
    souffle::RamDomain tools_sym) {
    const std::string tools = st->decode(tools_sym);
    int n = arr_len(st, rt, tools, "");
    return n < 0 ? 0 : static_cast<souffle::RamDomain>(n);
}

}  // extern "C"
