/* Custom C++ functors for the tau2-airline policy (run1).

   Two families:

   1. Content classifiers over message text (confirmation, cancellation
      reason, compensation request). Case-insensitive phrase matchers,
      used instead of @llm_check_fn so the policy is deterministic and
      testable offline. Every phrase list is grounded in language that
      actually appears in the captured customer turns of
      outputs/variant-set/run1/scenarios.json.

   2. JSON readers that the shared functors_common.cpp cannot express:
      resolving a flight number's endpoints out of an arbitrary payload
      (search results are top-level ARRAYS, reservations nest the legs
      under "flights"), counting payment methods by id prefix, and
      comparing ISO dates.

   Conservative by construction: when a classifier is unsure it returns
   0 and when a reader cannot resolve a value it returns "" / -1, so the
   rule that consumes it does not fire. Every rule built on these
   functors is a DENY rule, so "unsure" always means "do not block".
*/

#include <souffle/SouffleInterface.h>
#include <cctype>
#include <cstring>
#include <string>

namespace {

// ── Text utilities ──────────────────────────────────────────────

// Lowercase, folding U+2018 / U+2019 (typographic single quotes) to
// ASCII '. The captured user turns are written with curly apostrophes
// ("that’s correct"), and the phrase tables below use ASCII ones —
// without this fold, "that's correct" never matches.
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

bool contains(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}

bool any_of(const std::string& hay, const char* const* needles) {
    for (const char* const* p = needles; *p; ++p) {
        if (contains(hay, *p)) return true;
    }
    return false;
}

// True if `needle` occurs in `hay` bracketed by non-alphanumeric
// characters (or string edges) — so "yes" does not match inside "eyes".
bool contains_word(const std::string& hay, const char* needle) {
    const std::size_t n = std::strlen(needle);
    std::size_t pos = 0;
    while ((pos = hay.find(needle, pos)) != std::string::npos) {
        const bool left_ok =
            pos == 0
            || !std::isalnum(static_cast<unsigned char>(hay[pos - 1]));
        const std::size_t after = pos + n;
        const bool right_ok =
            after >= hay.size()
            || !std::isalnum(static_cast<unsigned char>(hay[after]));
        if (left_ok && right_ok) return true;
        pos += n;
    }
    return false;
}

// True if any phrase in `cues` occurs, and within `window` characters
// AFTER that occurrence one of `follow` occurs. Used to require that an
// affirmative or a proceed-verb is actually directed at going ahead
// with a named action rather than answering an unrelated question.
bool cue_followed_by(const std::string& hay, const char* const* cues,
                     const char* const* follow, std::size_t window,
                     bool cue_is_word) {
    for (const char* const* c = cues; *c; ++c) {
        const std::size_t clen = std::strlen(*c);
        std::size_t pos = 0;
        while ((pos = hay.find(*c, pos)) != std::string::npos) {
            bool boundary_ok = true;
            if (cue_is_word) {
                const bool left_ok =
                    pos == 0
                    || !std::isalnum(
                        static_cast<unsigned char>(hay[pos - 1]));
                const std::size_t after = pos + clen;
                const bool right_ok =
                    after >= hay.size()
                    || !std::isalnum(
                        static_cast<unsigned char>(hay[after]));
                boundary_ok = left_ok && right_ok;
            }
            if (boundary_ok) {
                const std::size_t start = pos + clen;
                const std::string tail =
                    start >= hay.size()
                        ? std::string()
                        : hay.substr(start, window);
                if (any_of(tail, follow)) return true;
            }
            pos += clen;
        }
    }
    return false;
}

// ── JSON utilities ──────────────────────────────────────────────

// Read a string field out of the JSON object that encloses position
// `p`. Walks back to the object's '{', forward to its matching '}'
// (quote-aware, so braces inside strings do not confuse the depth
// counter), then looks for "field": "value" inside that span. Bounding
// the search to one object is what stops a sibling leg's airport from
// being returned.
std::string field_in_enclosing_object(const std::string& json,
                                      std::size_t p,
                                      const std::string& field) {
    const std::size_t obj_start = json.rfind('{', p);
    if (obj_start == std::string::npos) return "";
    int depth = 1;
    std::size_t i = obj_start + 1;
    std::size_t obj_end = std::string::npos;
    while (i < json.size() && depth > 0) {
        const char ch = json[i];
        if (ch == '{') {
            ++depth;
        } else if (ch == '}') {
            --depth;
            if (depth == 0) obj_end = i;
        } else if (ch == '"') {
            ++i;
            while (i < json.size() && json[i] != '"') {
                if (json[i] == '\\' && i + 1 < json.size()) ++i;
                ++i;
            }
        }
        ++i;
    }
    if (obj_end == std::string::npos) return "";

    const std::string key = "\"" + field + "\"";
    const std::size_t f = json.find(key, obj_start);
    if (f == std::string::npos || f > obj_end) return "";
    const std::size_t colon = json.find(':', f + key.size());
    if (colon == std::string::npos || colon > obj_end) return "";
    std::size_t v = colon + 1;
    while (v < json.size()
           && std::isspace(static_cast<unsigned char>(json[v]))) ++v;
    if (v >= json.size() || json[v] != '"') return "";
    const std::size_t q1 = v + 1;
    const std::size_t q2 = json.find('"', q1);
    if (q2 == std::string::npos || q2 > obj_end) return "";
    return json.substr(q1, q2 - q1);
}

// Resolve `field` for the leg whose "flight_number" equals
// `flight_num`, wherever that leg object sits in `json` — a top-level
// search-result array, a nested "flights" list, anything. Returns ""
// when the flight is absent or the field is missing.
std::string flight_field(const std::string& json,
                         const std::string& flight_num,
                         const std::string& field) {
    static const char kKey[] = "\"flight_number\"";
    const std::size_t key_len = sizeof(kKey) - 1;
    std::size_t pos = 0;
    while (true) {
        const std::size_t p = json.find(kKey, pos);
        if (p == std::string::npos) return "";
        const std::size_t colon = json.find(':', p + key_len);
        if (colon == std::string::npos) return "";
        std::size_t v = colon + 1;
        while (v < json.size()
               && std::isspace(static_cast<unsigned char>(json[v]))) ++v;
        if (v >= json.size() || json[v] != '"') {
            pos = v + 1;
            continue;
        }
        const std::size_t q1 = v + 1;
        const std::size_t q2 = json.find('"', q1);
        if (q2 == std::string::npos) return "";
        if (q2 - q1 == flight_num.size()
            && json.compare(q1, flight_num.size(), flight_num) == 0) {
            const std::string got =
                field_in_enclosing_object(json, p, field);
            if (!got.empty()) return got;
        }
        pos = q2 + 1;
    }
}

// ── Phrase tables ───────────────────────────────────────────────

// Continuations that turn a bare "yes" into assent to PROCEED with the
// action the agent just laid out, as opposed to a "yes" answering some
// other question inside a fresh request. Grounded in the captured
// turns: "Yes, please proceed with the booking", "Yes, that's the
// correct reservation", "Yes, please use the saved gift card" all
// confirm; "Yes, I want to keep Economy ... Also, I'd like to add one
// checked bag" does not.
const char* const kAssentFollow[] = {
    "please proceed", "please go ahead", "go ahead", "proceed",
    "please use", "please book", "please cancel", "please apply",
    "please charge", "please update", "please confirm", "please do",
    "that's correct", "that is correct", "that's the correct",
    "that is the correct", "that's right", "that is right",
    "let's do", "confirmed", nullptr};

// Imperatives that instruct the agent to carry the action out.
const char* const kProceedCue[] = {
    "go ahead and", "go ahead with", "proceed with", "process the",
    nullptr};

// ... but only when the instruction NAMES the action. "Please go ahead
// with that" refers back to a plan the agent has not restated, which is
// exactly the case the policy treats as unconfirmed.
const char* const kActionNoun[] = {
    "change", "upgrade", "downgrade", "book", "cancel", "update",
    "modif", "payment", "reservation", "baggage", "bag", "certificate",
    "refund", nullptr};

const char* const kCancellationReason[] = {
    "change of plan", "changed my plan", "change my plan",
    "airline cancelled", "airline canceled",
    "flight was cancelled", "flight was canceled",
    "flight got cancelled", "flight got canceled",
    "no longer need", "no longer travel", "no longer going",
    "can't travel", "cannot travel", "can't make", "cannot make",
    "sick", "ill", "illness", "health", "hospital", "emergency",
    "weather", "storm", "hurricane", "snow",
    "schedule conflict", "conflict", "work", "family",
    "double booked", "double-booked", "by mistake", "by accident",
    "don't need", "do not need", "reason", "because", nullptr};

// Reasons the spec says travel insurance covers: health, weather, and
// other significant personal circumstances (schedule conflicts, change
// of plans).
const char* const kCoveredReason[] = {
    "sick", "ill", "illness", "health", "hospital", "medical",
    "injur", "emergency", "surgery",
    "weather", "storm", "hurricane", "snow", "blizzard", "flood",
    "schedule conflict", "scheduling conflict", "conflict",
    "change of plan", "changed my plan", "change my plan",
    "family", "funeral", "bereavement", "personal circumstance",
    nullptr};

const char* const kAirlineCancelled[] = {
    "airline cancelled", "airline canceled",
    "cancelled by the airline", "canceled by the airline",
    "flight was cancelled", "flight was canceled",
    "flight got cancelled", "flight got canceled",
    "you cancelled my flight", "you canceled my flight", nullptr};

const char* const kCompensationRequest[] = {
    "compensation", "compensate", "reimburse", "voucher",
    "travel certificate", "certificate for", "credit for the trouble",
    "make it right", "what can you do for me", "any compensation",
    nullptr};

}  // namespace

extern "C" {

// ── Confirmation ────────────────────────────────────────────────

// 1 iff `text` explicitly confirms going ahead with the action the
// agent proposed. Two accepted shapes, both of which appear verbatim in
// the captured turns:
//   (a) an affirmative "yes" whose continuation is assent to proceed;
//   (b) an imperative to proceed that names the action.
souffle::RamDomain user_confirms_action(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym) {
    const std::string text = to_lower(st->decode(text_sym));
    static const char* const kYes[] = {"yes", "yep", "yeah", nullptr};
    if (cue_followed_by(text, kYes, kAssentFollow, 48, true)) return 1;
    if (cue_followed_by(text, kProceedCue, kActionNoun, 48, false))
        return 1;
    return 0;
}

// ── Cancellation reasons ────────────────────────────────────────

// 1 iff the customer gave any reason for wanting to cancel.
souffle::RamDomain cancellation_reason_stated(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym) {
    const std::string text = to_lower(st->decode(text_sym));
    return any_of(text, kCancellationReason) ? 1 : 0;
}

// 1 iff the stated reason is one travel insurance covers.
souffle::RamDomain covered_cancellation_reason(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym) {
    const std::string text = to_lower(st->decode(text_sym));
    return any_of(text, kCoveredReason) ? 1 : 0;
}

// 1 iff the text says the AIRLINE cancelled the flight.
souffle::RamDomain airline_cancelled_flight(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym) {
    const std::string text = to_lower(st->decode(text_sym));
    return any_of(text, kAirlineCancelled) ? 1 : 0;
}

// ── Compensation ────────────────────────────────────────────────

// 1 iff the customer asked for compensation themselves.
souffle::RamDomain user_requested_compensation(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym) {
    const std::string text = to_lower(st->decode(text_sym));
    return any_of(text, kCompensationRequest) ? 1 : 0;
}

// ── Flight endpoints ────────────────────────────────────────────

souffle::RamDomain flight_origin(souffle::SymbolTable* st,
                                 souffle::RecordTable*,
                                 souffle::RamDomain json_sym,
                                 souffle::RamDomain flight_sym) {
    return st->encode(flight_field(st->decode(json_sym),
                                   st->decode(flight_sym), "origin"));
}

souffle::RamDomain flight_destination(souffle::SymbolTable* st,
                                      souffle::RecordTable*,
                                      souffle::RamDomain json_sym,
                                      souffle::RamDomain flight_sym) {
    return st->encode(flight_field(st->decode(json_sym),
                                   st->decode(flight_sym),
                                   "destination"));
}

// ── Dates ───────────────────────────────────────────────────────

// 1 iff ISO date `a` is strictly before ISO date `b`. Compares the
// leading YYYY-MM-DD only, so "2024-05-13" and "2024-05-13T05:37:57"
// compare identically. Returns 0 on anything that is not a full date,
// so an unparseable value never triggers the deny rules that use it.
// Souffle's own symbol ordering is symbol-table order, NOT
// lexicographic — hence this functor rather than a `<` on symbols.
souffle::RamDomain date_before(souffle::SymbolTable* st,
                               souffle::RecordTable*,
                               souffle::RamDomain a_sym,
                               souffle::RamDomain b_sym) {
    const std::string a = st->decode(a_sym);
    const std::string b = st->decode(b_sym);
    if (a.size() < 10 || b.size() < 10) return 0;
    for (int i = 0; i < 10; ++i) {
        const char ca = a[i];
        const char cb = b[i];
        const bool sep = (i == 4 || i == 7);
        if (sep) {
            if (ca != '-' || cb != '-') return 0;
            continue;
        }
        if (!std::isdigit(static_cast<unsigned char>(ca))
            || !std::isdigit(static_cast<unsigned char>(cb)))
            return 0;
    }
    return a.compare(0, 10, b, 0, 10) < 0 ? 1 : 0;
}

// ── Payment methods ─────────────────────────────────────────────

// Number of entries in a book_reservation "payment_methods" array whose
// payment_id starts with `prefix` ("gift_card", "credit_card",
// "certificate"). Returns 0 when the array is absent.
souffle::RamDomain count_payment_prefix(souffle::SymbolTable* st,
                                        souffle::RecordTable*,
                                        souffle::RamDomain json_sym,
                                        souffle::RamDomain prefix_sym) {
    const std::string json = st->decode(json_sym);
    const std::string prefix = st->decode(prefix_sym);
    static const char kKey[] = "\"payment_id\"";
    const std::size_t key_len = sizeof(kKey) - 1;
    souffle::RamDomain n = 0;
    std::size_t pos = 0;
    while (true) {
        const std::size_t p = json.find(kKey, pos);
        if (p == std::string::npos) break;
        const std::size_t colon = json.find(':', p + key_len);
        if (colon == std::string::npos) break;
        std::size_t v = colon + 1;
        while (v < json.size()
               && std::isspace(static_cast<unsigned char>(json[v]))) ++v;
        if (v >= json.size() || json[v] != '"') {
            pos = v + 1;
            continue;
        }
        const std::size_t q1 = v + 1;
        const std::size_t q2 = json.find('"', q1);
        if (q2 == std::string::npos) break;
        if (q2 - q1 >= prefix.size()
            && json.compare(q1, prefix.size(), prefix) == 0)
            ++n;
        pos = q2 + 1;
    }
    return n;
}

}  // extern "C"
