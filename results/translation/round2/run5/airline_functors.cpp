// Companion C++ functors for airline_policy.dl (tau2-airline, run5).
//
// Every content check in the policy is a deterministic keyword /
// structural matcher implemented here. Phrase tables are grounded in
// what the captured customers in outputs/variant-set/run5/scenarios.json
// actually wrote, not in what a customer might imagine writing.
//
// The JSON-shape helpers (flights_differ, any_flight_departed,
// count_payment_prefix) reuse the exported C-ABI accessors from
// functors_common.cpp rather than re-parsing JSON here.

#include <souffle/SouffleInterface.h>

#include <algorithm>
#include <cctype>
#include <string>

// ── Reused from functors_common.cpp (same shared library) ──────────
extern "C" souffle::RamDomain json_array_len(
    souffle::SymbolTable*, souffle::RecordTable*,
    souffle::RamDomain, souffle::RamDomain);

extern "C" souffle::RamDomain json_array_get_field_str(
    souffle::SymbolTable*, souffle::RecordTable*,
    souffle::RamDomain, souffle::RamDomain,
    souffle::RamDomain, souffle::RamDomain);

namespace {

std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return out;
}

bool any_of(const std::string& lower, const char* const* phrases) {
    for (const char* const* p = phrases; *p; ++p) {
        if (lower.find(*p) != std::string::npos) return true;
    }
    return false;
}

// Explicit go-ahead. Deliberately excludes bare requests ("please
// update the name", "sure, my reservation id is ..."): in the traces
// those turns precede any listing of action details, and the policy
// treats them as a request to act, not as confirmation that the listed
// details are approved.
const char* const kAffirm[] = {
    "yes", "yeah", "yep",
    "go ahead", "go for it", "proceed", "move forward",
    "confirm",
    "sounds good", "sounds perfect", "looks good",
    "that's correct", "thats correct",
    "that's fine", "thats fine",
    "that works", "works for me",
    "do it", "let's go", "lets go", "let's do", "lets do",
    "i agree", "agreed", "approve", "authorize",
    "okay", "ok,", "alright",
    nullptr,
};

// A go-ahead that is conditioned on a figure the agent has not
// supplied yet is not an explicit go-ahead. Grounded in the run5
// bookings where the customer wrote "let me know how much will be
// charged to my master card before you finalize the bookings".
const char* const kPending[] = {
    "before you finalize", "before finalizing", "before we finalize",
    "before you book", "before booking",
    "before we proceed", "before proceeding",
    "before you make", "before you complete", "before you charge",
    nullptr,
};

// Reasons a customer gives for cancelling. Covers the three buckets
// the policy names (change of plan, airline cancelled the flight,
// other reasons) plus the insurance-covered categories.
const char* const kCancelReason[] = {
    "change of plan", "changed my mind", "change my mind",
    "no longer", "don't need", "do not need",
    "cancelled by the airline", "canceled by the airline",
    "airline cancelled", "airline canceled",
    "flight was cancelled", "flight was canceled",
    "schedule conflict", "conflicting", "overlap",
    "double booked", "double-booked",
    "can't make", "cannot make", "won't be able", "wont be able",
    "doesn't work", "does not work",
    "sick", "illness", "health", "emergency",
    "weather", "personal reasons", "family",
    nullptr,
};

// The customer asking for compensation. The policy forbids offering
// it unprompted, so the agent may only send a certificate after one
// of these appears in a customer turn.
const char* const kCompensation[] = {
    "compensation", "compensate", "voucher", "reimburse",
    "make it right", "make up for",
    "certificate for the trouble", "refund for the trouble",
    "credit for the inconvenience",
    "something for the inconvenience",
    "for the inconvenience", "for my trouble", "for the trouble",
    nullptr,
};

}  // namespace

extern "C" {

/**
 * @user_confirms(text) -> 0 or 1
 *
 * 1 iff `text` is an explicit go-ahead that is not conditioned on
 * information the agent still owes the customer.
 */
souffle::RamDomain user_confirms(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string lower = to_lower(st->decode(text_sym));
    if (!any_of(lower, kAffirm)) return 0;
    if (any_of(lower, kPending)) return 0;
    return 1;
}

/**
 * @confirmation_deferred(text) -> 0 or 1
 *
 * 1 iff `text` puts off the decision until the agent supplies a
 * figure it still owes. Separated from user_confirms so it can veto
 * the payment-instrument route to confirmation as well.
 */
souffle::RamDomain confirmation_deferred(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    return any_of(to_lower(st->decode(text_sym)), kPending) ? 1 : 0;
}

/**
 * @payment_last_four(profile_payload, payment_id) -> digits or ""
 *
 * The last_four digits the customer's profile records for
 * `payment_id`, or "" when the profile has no such entry or the
 * instrument has no digits (gift cards and travel certificates carry
 * an amount instead). Scans the object for `payment_id` under
 * payment_methods with brace matching, so a last_four belonging to a
 * different card can never be returned.
 */
souffle::RamDomain payment_last_four(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain profile_sym, souffle::RamDomain pid_sym)
{
    const std::string profile = st->decode(profile_sym);
    const std::string pid = st->decode(pid_sym);
    if (pid.empty()) return st->encode("");

    const std::string methods_key = "\"payment_methods\"";
    const size_t methods_at = profile.find(methods_key);
    if (methods_at == std::string::npos) return st->encode("");

    const std::string pid_key = "\"" + pid + "\"";
    size_t at = profile.find(pid_key, methods_at);
    if (at == std::string::npos) return st->encode("");

    // Walk to this key's object value and find its matching brace.
    at = profile.find('{', at + pid_key.size());
    if (at == std::string::npos) return st->encode("");
    int depth = 0;
    size_t end = at;
    for (; end < profile.size(); ++end) {
        if (profile[end] == '{') ++depth;
        else if (profile[end] == '}' && --depth == 0) break;
    }
    if (end >= profile.size()) return st->encode("");

    const std::string object = profile.substr(at, end - at);
    const std::string digits_key = "\"last_four\"";
    size_t d = object.find(digits_key);
    if (d == std::string::npos) return st->encode("");
    d = object.find('"', d + digits_key.size());  // opening quote
    if (d == std::string::npos) return st->encode("");
    const size_t close = object.find('"', d + 1);
    if (close == std::string::npos) return st->encode("");
    return st->encode(object.substr(d + 1, close - d - 1));
}

/**
 * @cancel_reason(text) -> 0 or 1
 *
 * 1 iff `text` states a reason for cancelling.
 */
souffle::RamDomain cancel_reason(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    return any_of(to_lower(st->decode(text_sym)), kCancelReason) ? 1 : 0;
}

/**
 * @compensation_requested(text) -> 0 or 1
 *
 * 1 iff `text` asks for compensation.
 */
souffle::RamDomain compensation_requested(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    return any_of(to_lower(st->decode(text_sym)), kCompensation) ? 1 : 0;
}

/**
 * @contains_ci(haystack, needle) -> 0 or 1
 *
 * Case-insensitive substring test. An empty needle returns 0 rather
 * than the vacuous 1 std::string::find would give, so a missing
 * argument can never ground a rule by accident.
 */
souffle::RamDomain contains_ci(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain hay_sym, souffle::RamDomain needle_sym)
{
    const std::string needle = to_lower(st->decode(needle_sym));
    if (needle.empty()) return 0;
    const std::string hay = to_lower(st->decode(hay_sym));
    return hay.find(needle) != std::string::npos ? 1 : 0;
}

/**
 * @flights_differ(call_args, reservation_payload) -> 0 or 1
 *
 * 1 iff the `flights` array proposed in `call_args` is not the same
 * itinerary as the one in `reservation_payload`, comparing element by
 * element on flight_number and date. The reservation payload carries
 * extra per-flight fields (origin, destination, price) that the call
 * arguments never carry, so only the two shared fields are compared.
 *
 * Returns 0 when the call proposes no flights at all — a cabin-only
 * change is not a flight change.
 */
souffle::RamDomain flights_differ(
    souffle::SymbolTable* st, souffle::RecordTable* rt,
    souffle::RamDomain args_sym, souffle::RamDomain res_sym)
{
    const souffle::RamDomain path = st->encode("flights");
    const souffle::RamDomain f_num = st->encode("flight_number");
    const souffle::RamDomain f_date = st->encode("date");

    const auto n_args =
        static_cast<int>(json_array_len(st, rt, args_sym, path));
    const auto n_res =
        static_cast<int>(json_array_len(st, rt, res_sym, path));

    if (n_args == 0) return 0;
    if (n_args != n_res) return 1;

    for (int i = 0; i < n_args; ++i) {
        const souffle::RamDomain idx = static_cast<souffle::RamDomain>(i);
        const std::string a_num = st->decode(
            json_array_get_field_str(st, rt, args_sym, path, idx, f_num));
        const std::string r_num = st->decode(
            json_array_get_field_str(st, rt, res_sym, path, idx, f_num));
        if (a_num != r_num) return 1;

        const std::string a_date = st->decode(
            json_array_get_field_str(st, rt, args_sym, path, idx, f_date));
        const std::string r_date = st->decode(
            json_array_get_field_str(st, rt, res_sym, path, idx, f_date));
        // A reservation payload always dates its flights; the call
        // arguments do too. If either side omits the date, fall back
        // to the flight number comparison already made above.
        if (!a_date.empty() && !r_date.empty() && a_date != r_date) {
            return 1;
        }
    }
    return 0;
}

/**
 * @any_flight_departed(reservation_payload, today) -> 0 or 1
 *
 * 1 iff any flight in the reservation is dated strictly before
 * `today` (both "YYYY-MM-DD"), i.e. that portion of the trip has been
 * flown. Dates that do not parse as YYYY-MM-DD are ignored rather
 * than treated as past, so a malformed payload cannot silently block
 * every cancellation.
 */
souffle::RamDomain any_flight_departed(
    souffle::SymbolTable* st, souffle::RecordTable* rt,
    souffle::RamDomain res_sym, souffle::RamDomain today_sym)
{
    const std::string today = st->decode(today_sym);
    if (today.size() != 10) return 0;

    const souffle::RamDomain path = st->encode("flights");
    const souffle::RamDomain f_date = st->encode("date");
    const auto n =
        static_cast<int>(json_array_len(st, rt, res_sym, path));

    for (int i = 0; i < n; ++i) {
        const std::string d = st->decode(json_array_get_field_str(
            st, rt, res_sym, path,
            static_cast<souffle::RamDomain>(i), f_date));
        if (d.size() != 10) continue;
        if (d < today) return 1;  // ISO dates order lexicographically
    }
    return 0;
}

/**
 * @count_payment_prefix(call_args, prefix) -> count
 *
 * Number of entries in the `payment_methods` array of `call_args`
 * whose payment_id starts with `prefix` ("certificate", "credit_card",
 * "gift_card").
 */
souffle::RamDomain count_payment_prefix(
    souffle::SymbolTable* st, souffle::RecordTable* rt,
    souffle::RamDomain args_sym, souffle::RamDomain prefix_sym)
{
    const std::string prefix = st->decode(prefix_sym);
    if (prefix.empty()) return 0;

    const souffle::RamDomain path = st->encode("payment_methods");
    const souffle::RamDomain field = st->encode("payment_id");
    const auto n =
        static_cast<int>(json_array_len(st, rt, args_sym, path));

    souffle::RamDomain hits = 0;
    for (int i = 0; i < n; ++i) {
        const std::string pid = st->decode(json_array_get_field_str(
            st, rt, args_sym, path,
            static_cast<souffle::RamDomain>(i), field));
        if (pid.compare(0, prefix.size(), prefix) == 0) ++hits;
    }
    return hits;
}

}  // extern "C"
