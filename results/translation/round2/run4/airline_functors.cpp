/* Custom C++ functors for outputs/variant-set/run4/airline_policy.dl
   (tau2 airline domain).

   Two groups:

   1. Content classifiers over message text. These stand in for
      @llm_check_fn, which this deployment cannot run. They are
      deterministic case-insensitive matchers over phrase lists drawn
      from the run4 traces rather than from imagination, with
      word-boundary checks so short tokens like "yes" do not fire on
      "eyes" and a negation guard so "that's not correct" is not read
      as agreement.

   2. JSON helpers over tool arguments and tool results, layered on the
      shared accessors in functors_common.cpp (compiled into the same
      library) so there is only one JSON scanner to trust.

   Every classifier is conservative: when the signal is absent or
   ambiguous it returns 0, and the policy rule that consumes it decides
   whether 0 means allow or deny.
*/

#include <souffle/SouffleInterface.h>

#include <cctype>
#include <cstring>
#include <string>

/* ── Shared JSON accessors from functors_common.cpp ───────────────── */
extern "C" souffle::RamDomain json_array_len(
    souffle::SymbolTable*, souffle::RecordTable*,
    souffle::RamDomain, souffle::RamDomain);

extern "C" souffle::RamDomain json_array_get_field_str(
    souffle::SymbolTable*, souffle::RecordTable*,
    souffle::RamDomain, souffle::RamDomain,
    souffle::RamDomain, souffle::RamDomain);

extern "C" souffle::RamDomain json_object_has_key(
    souffle::SymbolTable*, souffle::RecordTable*,
    souffle::RamDomain, souffle::RamDomain, souffle::RamDomain);

namespace {

/* ── Text utilities ──────────────────────────────────────────────── */

// Lowercase, folding U+2018 / U+2019 (typographic single quotes) to
// ASCII '. The tau2 user simulator writes contractions with curly
// apostrophes ("I'd like to cancel"), and the phrase lists below use
// ASCII ones — without this fold, phrases spanning an apostrophe never
// match.
std::string to_lower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        if (i + 2 < s.size()
            && static_cast<unsigned char>(s[i]) == 0xE2
            && static_cast<unsigned char>(s[i + 1]) == 0x80
            && (static_cast<unsigned char>(s[i + 2]) == 0x98
                || static_cast<unsigned char>(s[i + 2]) == 0x99)) {
            out.push_back('\'');
            i += 3;
            continue;
        }
        out.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(s[i]))));
        ++i;
    }
    return out;
}

bool is_word_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0;
}

// True if `needle` occurs in `hay` bracketed by non-alphanumerics.
bool word_match(const std::string& hay, const char* needle) {
    const std::size_t nlen = std::strlen(needle);
    if (nlen == 0) return false;
    std::size_t pos = 0;
    while ((pos = hay.find(needle, pos)) != std::string::npos) {
        const bool left_ok = pos == 0 || !is_word_char(hay[pos - 1]);
        const bool right_ok =
            pos + nlen == hay.size() || !is_word_char(hay[pos + nlen]);
        if (left_ok && right_ok) return true;
        pos += 1;
    }
    return false;
}

// A negation in the ~14 characters before a hit flips its meaning:
// "that's not correct" and "I don't want to cancel" both contain a
// token that would otherwise read as assent or as intent.
bool negated_before(const std::string& hay, std::size_t pos) {
    const std::size_t window = 14;
    const std::size_t start = pos > window ? pos - window : 0;
    const std::string before = hay.substr(start, pos - start);
    static const char* const negators[] = {
        "not ", "n't ", "never ", "no longer ", "cannot ", "can't ",
    };
    for (const char* neg : negators) {
        if (before.find(neg) != std::string::npos) return true;
    }
    return false;
}

// word_match that skips negated hits.
bool word_match_positive(const std::string& hay, const char* needle) {
    const std::size_t nlen = std::strlen(needle);
    if (nlen == 0) return false;
    std::size_t pos = 0;
    while ((pos = hay.find(needle, pos)) != std::string::npos) {
        const bool left_ok = pos == 0 || !is_word_char(hay[pos - 1]);
        const bool right_ok =
            pos + nlen == hay.size() || !is_word_char(hay[pos + nlen]);
        if (left_ok && right_ok && !negated_before(hay, pos)) return true;
        pos += 1;
    }
    return false;
}

bool any_word(const std::string& hay, const char* const* needles) {
    for (const char* const* n = needles; *n; ++n) {
        if (word_match_positive(hay, *n)) return true;
    }
    return false;
}

// Does the text open with this token? Used for assent words that are
// only assent in the opening position. "sure" is the reason this
// exists: run4 scenario 56's final user turn says "can you make sure I
// get a full refund", which is a plea, not a confirmation — while
// "Sure, go ahead and cancel it" is a confirmation.
bool starts_with_word(const std::string& hay, const char* needle) {
    std::size_t i = 0;
    while (i < hay.size() && !is_word_char(hay[i]) && hay[i] != '\''
           && static_cast<unsigned char>(hay[i]) < 0x80) {
        ++i;
    }
    const std::size_t nlen = std::strlen(needle);
    if (hay.compare(i, nlen, needle) != 0) return false;
    const std::size_t after = i + nlen;
    return after == hay.size() || !is_word_char(hay[after]);
}

bool any_leading(const std::string& hay, const char* const* needles) {
    for (const char* const* n = needles; *n; ++n) {
        if (starts_with_word(hay, *n)) return true;
    }
    return false;
}

std::string decode(souffle::SymbolTable* st, souffle::RamDomain sym) {
    return st->decode(sym);
}

}  // namespace

extern "C" {

/* ── @user_affirms(text) ──────────────────────────────────────────
   Explicit assent, as the spec's "obtain explicit user confirmation
   (yes)" asks for.

   Deliberately NOT assent: "go ahead" on its own. run4 scenario 54's
   final user turn — "Please go ahead and use my certificates first,
   and then pay the remaining balance with my 7447 card" — instructs
   the agent how to pay without ever agreeing to a presented summary,
   and the oracle denies the booking that followed it. The allowed
   write in scenario 41 is confirmed by "Yes, that sounds perfect!
   Please go ahead …", which the "yes" already covers. */
souffle::RamDomain user_affirms(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string text = to_lower(decode(st, text_sym));

    // Unambiguous assent anywhere in the turn.
    static const char* const strong[] = {
        "yes", "yeah", "yep", "yup", "affirmative", "confirmed",
        "correct", "agreed", nullptr,
    };
    if (any_word(text, strong)) return 1;

    // Assent phrases.
    static const char* const phrases[] = {
        "sounds good", "sounds perfect", "sounds great", "sounds right",
        "looks good", "looks right", "that works", "that's right",
        "that's correct", "i confirm", "please do that", "please do so",
        "go for it", "let's do it", "lets do it", "do it", "please book it",
        "please cancel it", "please proceed with the cancellation",
        nullptr,
    };
    for (const char* const* p = phrases; *p; ++p) {
        const std::size_t pos = text.find(*p);
        if (pos != std::string::npos && !negated_before(text, pos)) return 1;
    }

    // Weaker tokens count only when the turn opens with them.
    static const char* const leading[] = {
        "sure", "ok", "okay", "absolutely", "certainly", "definitely",
        "of course", "perfect", "great", nullptr,
    };
    if (any_leading(text, leading)) return 1;

    return 0;
}

/* ── @mentions_booking(text) ─────────────────────────────────────── */
souffle::RamDomain mentions_booking(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string text = to_lower(decode(st, text_sym));
    static const char* const words[] = {
        "book", "books", "booked", "booking", "reserve", "reserved",
        "reservation", "purchase", "buy", "ticket", "tickets",
        "pay", "payment", "itinerary", nullptr,
    };
    return any_word(text, words) ? 1 : 0;
}

/* ── @mentions_cancellation(text) ─────────────────────────────────
   "refund" is deliberately absent. In run4 scenario 20 the user says
   "Yes, please proceed with the refund request" and the agent then
   cancels a reservation nobody discussed cancelling; the oracle denies
   it because the confirmation was not for the cancellation. Reading
   "refund" as consent to cancel would allow exactly that. */
souffle::RamDomain mentions_cancellation(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string text = to_lower(decode(st, text_sym));
    static const char* const words[] = {
        "cancel", "cancels", "cancelled", "canceled", "cancelling",
        "canceling", "cancellation", "cancelation", nullptr,
    };
    return any_word(text, words) ? 1 : 0;
}

/* ── @mentions_modification(text) ─────────────────────────────────── */
souffle::RamDomain mentions_modification(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string text = to_lower(decode(st, text_sym));
    static const char* const words[] = {
        "change", "changes", "changing", "modify", "modified",
        "modification", "update", "updated", "switch", "switching",
        "rebook", "reschedule", "move", "swap", "upgrade", "downgrade",
        "cabin", "baggage", "bag", "bags", "luggage", "passenger",
        "passengers", nullptr,
    };
    return any_word(text, words) ? 1 : 0;
}

/* ── @states_cancellation_reason(text) ────────────────────────────
   Spec: "The agent must also obtain the reason for cancellation
   (change of plan, airline cancelled flight, or other reasons)."
   Broad by design — this rule should push the agent to ask, not deny
   users who gave a reason in wording the list did not anticipate. */
souffle::RamDomain states_cancellation_reason(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string text = to_lower(decode(st, text_sym));

    static const char* const phrases[] = {
        "change of plan", "change of plans", "plans changed",
        "plans have changed", "changed my mind", "change my mind",
        "no longer need", "don't need", "do not need", "not going",
        "can't travel", "cannot travel", "can't make it",
        "unable to travel", "won't be able", "will not be able",
        "something came up", "came up", "family emergency",
        "personal reasons", "personal circumstances", "no longer able",
        "airline cancelled", "airline canceled", "flight was cancelled",
        "flight was canceled", "flight got cancelled", "was delayed",
        "is delayed", "schedule conflict", "scheduling conflict",
        "work conflict", "double booked", "passed away", "double-booked",
        nullptr,
    };
    for (const char* const* p = phrases; *p; ++p) {
        if (text.find(*p) != std::string::npos) return 1;
    }

    // Bare "cancelled" / "canceled" / "delayed" are deliberately absent.
    // A user asking to cancel says those words about their own request —
    // run4 scenario 55's "I want to make sure the right one gets
    // canceled" is not a reason for cancelling — so admitting them here
    // made this rule fire for practically every cancellation
    // conversation, i.e. never. The reason-bearing forms ("the airline
    // cancelled", "the flight was delayed") are in the phrase list above.
    static const char* const words[] = {
        "emergency", "sick", "ill", "illness", "injury", "injured",
        "hospital", "hospitalized", "doctor", "medical", "health",
        "weather", "storm", "hurricane", "snowstorm", "blizzard",
        "flooding", "bereavement", "funeral", "surgery", "quarantine",
        "visa", "passport",
        nullptr,
    };
    return any_word(text, words) ? 1 : 0;
}

/* ── @covered_cancellation_reason(text) ───────────────────────────
   Spec: "Covered reasons include: health-related issues,
   weather-related issues, and other significant personal
   circumstances (schedule conflicts, change of plans)." Narrower than
   states_cancellation_reason — it only admits the covered families,
   because it unlocks a cancellation rather than merely satisfying the
   "give a reason" requirement. */
souffle::RamDomain covered_cancellation_reason(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string text = to_lower(decode(st, text_sym));

    static const char* const phrases[] = {
        // health-related
        "health issue", "health problem", "health reasons",
        "medical emergency", "medical issue", "medical reasons",
        "not well", "unwell", "in the hospital",
        // weather-related
        "bad weather", "weather conditions", "severe weather",
        "weather-related", "weather related",
        // other significant personal circumstances
        "family emergency", "personal emergency", "personal circumstances",
        "schedule conflict", "scheduling conflict", "work conflict",
        "change of plan", "change of plans", "plans changed",
        "plans have changed", "changed my mind", "change my mind",
        "passed away", "bereavement",
        nullptr,
    };
    for (const char* const* p = phrases; *p; ++p) {
        if (text.find(*p) != std::string::npos) return 1;
    }

    static const char* const words[] = {
        "sick", "ill", "illness", "injury", "injured", "hospital",
        "hospitalized", "surgery", "doctor", "medical", "health",
        "weather", "storm", "hurricane", "snowstorm", "blizzard",
        "flooding", "funeral", "emergency", "quarantine",
        nullptr,
    };
    return any_word(text, words) ? 1 : 0;
}

/* ── @user_requested_compensation(text) ───────────────────────────
   Spec: "Do not proactively offer a compensation unless the user
   explicitly asks for one." */
souffle::RamDomain user_requested_compensation(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string text = to_lower(decode(st, text_sym));

    static const char* const phrases[] = {
        "anything else you can offer", "what can you offer",
        "can you offer", "make it right", "make up for",
        "some kind of credit", "goodwill gesture", "as a gesture",
        "travel voucher", "bonus miles", "loyalty points",
        nullptr,
    };
    for (const char* const* p = phrases; *p; ++p) {
        if (text.find(*p) != std::string::npos) return 1;
    }

    static const char* const words[] = {
        "compensation", "compensate", "compensated", "voucher",
        "vouchers", "certificate", "certificates", "reimburse",
        "reimbursement", "recompense", nullptr,
    };
    return any_word(text, words) ? 1 : 0;
}

/* ── @count_json_array(json, path) ───────────────────────────────
   Array length as a `number`, so it can take part in the baggage and
   compensation arithmetic (@json_array_len returns `unsigned`). */
souffle::RamDomain count_json_array(
    souffle::SymbolTable* st, souffle::RecordTable* rt,
    souffle::RamDomain json_sym, souffle::RamDomain path_sym)
{
    return json_array_len(st, rt, json_sym, path_sym);
}

/* ── @count_payment_prefix(args, prefix) ──────────────────────────
   How many entries of args.payment_methods carry a payment_id with
   this prefix. tau2 payment ids are "certificate_…", "credit_card_…"
   and "gift_card_…", which is what makes prefix matching exact. */
souffle::RamDomain count_payment_prefix(
    souffle::SymbolTable* st, souffle::RecordTable* rt,
    souffle::RamDomain args_sym, souffle::RamDomain prefix_sym)
{
    const std::string prefix = decode(st, prefix_sym);
    const souffle::RamDomain path = st->encode("payment_methods");
    const souffle::RamDomain field = st->encode("payment_id");

    const souffle::RamDomain n = json_array_len(st, rt, args_sym, path);
    souffle::RamDomain count = 0;
    for (souffle::RamDomain i = 0; i < n; ++i) {
        const souffle::RamDomain pid_sym =
            json_array_get_field_str(st, rt, args_sym, path, i, field);
        const std::string pid = decode(st, pid_sym);
        if (pid.compare(0, prefix.size(), prefix) == 0) ++count;
    }
    return count;
}

/* ── @payment_not_in_profile(args, profile) ───────────────────────
   1 iff some payment_id the booking wants to charge is absent from the
   profile's payment_methods object. Spec: "All payment methods must
   already be in user profile for safety reasons."

   A booking that names no payment method at all returns 0 — the
   emptiness is not evidence of an off-profile method, and the
   remaining booking rules cover that case. */
souffle::RamDomain payment_not_in_profile(
    souffle::SymbolTable* st, souffle::RecordTable* rt,
    souffle::RamDomain args_sym, souffle::RamDomain profile_sym)
{
    const souffle::RamDomain path = st->encode("payment_methods");
    const souffle::RamDomain field = st->encode("payment_id");

    const souffle::RamDomain n = json_array_len(st, rt, args_sym, path);
    for (souffle::RamDomain i = 0; i < n; ++i) {
        const souffle::RamDomain pid_sym =
            json_array_get_field_str(st, rt, args_sym, path, i, field);
        const std::string pid = decode(st, pid_sym);
        if (pid.empty()) continue;
        if (json_object_has_key(st, rt, profile_sym, path, pid_sym) == 0) {
            return 1;
        }
    }
    return 0;
}

/* ── @any_flight_date_before(json, cutoff) ────────────────────────
   1 iff any leg of the payload's `flights` array is dated strictly
   before the cutoff — that leg has already been flown. Dates are ISO
   "YYYY-MM-DD", so lexicographic comparison is chronological.

   Legs dated ON the cutoff day are not treated as flown: the payload
   carries a date but no departure time, so the only sound reading of
   a same-day leg is "unknown", and get_flight_status is what resolves
   it (the landed / flying branches of ResHasFlownFlight). */
souffle::RamDomain any_flight_date_before(
    souffle::SymbolTable* st, souffle::RecordTable* rt,
    souffle::RamDomain json_sym, souffle::RamDomain cutoff_sym)
{
    const std::string cutoff = decode(st, cutoff_sym);
    if (cutoff.empty()) return 0;

    const souffle::RamDomain path = st->encode("flights");
    const souffle::RamDomain field = st->encode("date");

    const souffle::RamDomain n = json_array_len(st, rt, json_sym, path);
    for (souffle::RamDomain i = 0; i < n; ++i) {
        const souffle::RamDomain date_sym =
            json_array_get_field_str(st, rt, json_sym, path, i, field);
        const std::string date = decode(st, date_sym);
        if (date.empty()) continue;
        if (date < cutoff) return 1;
    }
    return 0;
}

}  // extern "C"
