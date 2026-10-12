/* Custom C++ functors for the tau2-airline policy (run1 variant).

   Two families:

     1. Content classifiers over user message text — case-insensitive
        phrase matchers used instead of @llm_check_fn. Phrase lists are
        grounded in what users actually said in the captured traces
        (outputs/variant-set/run1/scenarios.json), not in imagined
        phrasings. They are conservative: when in doubt they return 0
        (condition not met) so benign traffic is not blocked, EXCEPT
        for user_confirms_action, where 0 means "no confirmation
        found" and therefore blocks a write — that one is the
        permissive-by-design direction (any clear go-ahead counts).

     2. JSON structure helpers over tool-call arguments and tool
        results. The airline payloads are small, so these use a
        minimal recursive scanner rather than a full JSON library:
        objects and arrays are walked, strings honour backslash
        escapes, and every accessor returns an empty/zero value when
        the shape is not what it expected.
*/

#include <souffle/SouffleInterface.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace {

/* ── Text utilities ─────────────────────────────────────────────── */

// Lowercase, folding U+2018 / U+2019 (typographic single quotes) to
// ASCII '. The user simulator writes contractions with curly
// apostrophes ("I'd like", "don't"), so phrase lists written with
// ASCII apostrophes would otherwise never match.
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

template <std::size_t N>
bool any_of(const std::string& hay, const char* const (&needles)[N]) {
    for (std::size_t i = 0; i < N; ++i) {
        if (contains(hay, needles[i])) return true;
    }
    return false;
}

/* ── Minimal JSON scanner ───────────────────────────────────────── */
//
// The airline tool payloads are plain JSON objects/arrays of scalars.
// These helpers walk them without allocating a parse tree.

const char* skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    return p;
}

// Advance past the string starting at `p` (which points at the
// opening quote). Returns the position just past the closing quote.
const char* skip_string(const char* p) {
    if (*p != '"') return p;
    ++p;
    while (*p) {
        if (*p == '\\' && p[1]) {
            p += 2;
            continue;
        }
        if (*p == '"') return p + 1;
        ++p;
    }
    return p;
}

// Decode the JSON string at `p` (pointing at the opening quote).
// Only the escapes the airline payloads use are decoded; any other
// escape is passed through as its literal character.
std::string read_string(const char* p) {
    std::string out;
    if (*p != '"') return out;
    ++p;
    while (*p) {
        if (*p == '\\' && p[1]) {
            char c = p[1];
            switch (c) {
                case 'n': out.push_back('\n'); break;
                case 't': out.push_back('\t'); break;
                case 'r': out.push_back('\r'); break;
                case 'u': {
                    // Keep the escape verbatim — the fields these
                    // functors read (ids, dates, airport codes) are
                    // ASCII, so decoding is unnecessary.
                    out.append(p, p + 2);
                    break;
                }
                default: out.push_back(c); break;
            }
            p += 2;
            continue;
        }
        if (*p == '"') break;
        out.push_back(*p);
        ++p;
    }
    return out;
}

// Advance past the JSON value starting at `p`.
const char* skip_value(const char* p) {
    p = skip_ws(p);
    if (*p == '"') return skip_string(p);
    if (*p == '{' || *p == '[') {
        char open = *p;
        char close = (open == '{') ? '}' : ']';
        int depth = 0;
        while (*p) {
            if (*p == '"') {
                p = skip_string(p);
                continue;
            }
            if (*p == open) ++depth;
            else if (*p == close) {
                --depth;
                if (depth == 0) return p + 1;
            }
            ++p;
        }
        return p;
    }
    while (*p && *p != ',' && *p != '}' && *p != ']') ++p;
    return p;
}

// Pointer to the value of `name` in the object starting at `obj`
// (which must point at '{'), or nullptr. Only the object's own keys
// are considered — nested objects are skipped over, not searched.
const char* object_field(const char* obj, const char* name) {
    if (!obj) return nullptr;
    const char* p = skip_ws(obj);
    if (*p != '{') return nullptr;
    ++p;
    while (*p) {
        p = skip_ws(p);
        if (*p == '}') return nullptr;
        if (*p != '"') return nullptr;
        std::string key = read_string(p);
        p = skip_string(p);
        p = skip_ws(p);
        if (*p != ':') return nullptr;
        ++p;
        p = skip_ws(p);
        if (key == name) return p;
        p = skip_value(p);
        p = skip_ws(p);
        if (*p == ',') ++p;
    }
    return nullptr;
}

// String value of field `name` in the object at `obj`, or "".
std::string object_str(const char* obj, const char* name) {
    const char* v = object_field(obj, name);
    if (!v || *v != '"') return std::string();
    return read_string(v);
}

// Call `visit` with a pointer to every JSON object reachable from the
// value at `p` (including `p` itself), depth-first. Used to find
// flight objects wherever they sit: reservation payloads nest them
// under "flights", search_direct_flight returns a flat array, and
// search_onestop_flight returns an array of arrays.
template <typename F>
const char* walk_objects(const char* p, const F& visit, int depth = 0) {
    p = skip_ws(p);
    if (depth > 12) return skip_value(p);
    if (*p == '{') {
        visit(p);
        const char* q = p + 1;
        while (*q) {
            q = skip_ws(q);
            if (*q == '}') return q + 1;
            if (*q != '"') return skip_value(p);
            q = skip_string(q);
            q = skip_ws(q);
            if (*q != ':') return skip_value(p);
            ++q;
            q = walk_objects(q, visit, depth + 1);
            q = skip_ws(q);
            if (*q == ',') ++q;
        }
        return q;
    }
    if (*p == '[') {
        const char* q = p + 1;
        while (*q) {
            q = skip_ws(q);
            if (*q == ']') return q + 1;
            q = walk_objects(q, visit, depth + 1);
            q = skip_ws(q);
            if (*q == ',') ++q;
        }
        return q;
    }
    return skip_value(p);
}

// Call `visit` with each element of the top-level array field `name`
// of the object `json`. No-op when the field is missing or not an
// array.
template <typename F>
void each_array_elem(const std::string& json, const char* name,
                     const F& visit) {
    const char* arr = object_field(json.c_str(), name);
    if (!arr) return;
    arr = skip_ws(arr);
    if (*arr != '[') return;
    const char* p = arr + 1;
    while (*p) {
        p = skip_ws(p);
        if (*p == ']') return;
        visit(p);
        p = skip_value(p);
        p = skip_ws(p);
        if (*p == ',') ++p;
    }
}

// Keys of the object field `name` of `json` (e.g. the payment ids in
// a user profile's "payment_methods" map).
std::vector<std::string> object_keys(const std::string& json,
                                     const char* name) {
    std::vector<std::string> keys;
    const char* obj = object_field(json.c_str(), name);
    if (!obj) return keys;
    obj = skip_ws(obj);
    if (*obj != '{') return keys;
    const char* p = obj + 1;
    while (*p) {
        p = skip_ws(p);
        if (*p == '}') return keys;
        if (*p != '"') return keys;
        keys.push_back(read_string(p));
        p = skip_string(p);
        p = skip_ws(p);
        if (*p != ':') return keys;
        ++p;
        p = skip_value(p);
        p = skip_ws(p);
        if (*p == ',') ++p;
    }
    return keys;
}

/* ── Domain helpers ─────────────────────────────────────────────── */

// (flight_number, date) pairs of the top-level "flights" array.
std::vector<std::pair<std::string, std::string>> flight_keys(
    const std::string& json) {
    std::vector<std::pair<std::string, std::string>> out;
    each_array_elem(json, "flights", [&](const char* elem) {
        std::string num = object_str(elem, "flight_number");
        if (num.empty()) return;
        out.emplace_back(num, object_str(elem, "date"));
    });
    std::sort(out.begin(), out.end());
    return out;
}

// (last_name, dob) pairs of the top-level "passengers" array,
// last name lowercased so casing differences are not read as a
// different traveller.
std::vector<std::pair<std::string, std::string>> passenger_keys(
    const std::string& json) {
    std::vector<std::pair<std::string, std::string>> out;
    each_array_elem(json, "passengers", [&](const char* elem) {
        std::string last = to_lower(object_str(elem, "last_name"));
        std::string dob = object_str(elem, "dob");
        if (last.empty() && dob.empty()) return;
        out.emplace_back(last, dob);
    });
    return out;
}

// Payment ids referenced by a tool call's arguments — both shapes:
// book_reservation's "payment_methods" array of {payment_id, amount}
// and update_reservation_*'s single "payment_id" string.
std::vector<std::string> arg_payment_ids(const std::string& args) {
    std::vector<std::string> ids;
    std::string single = object_str(args.c_str(), "payment_id");
    if (!single.empty()) ids.push_back(single);
    each_array_elem(args, "payment_methods", [&](const char* elem) {
        std::string id = object_str(elem, "payment_id");
        if (id.empty()) id = object_str(elem, "id");
        if (!id.empty()) ids.push_back(id);
    });
    return ids;
}

}  // namespace

extern "C" {

/* ── 1. Content classifiers ─────────────────────────────────────── */

// 1 iff the user turn contains an explicit, action-directed
// go-ahead for the action the agent has just laid out.
//
// Grounded in the captured traces: confirmations there read "Yes,
// that's the correct reservation—please go ahead...", "Yes, please
// proceed with the booking...", "Please proceed with this change".
// Non-confirmations that must NOT match include "Yes, I want to keep
// Economy (not Basic Economy)" (a preference stated inside a list of
// requests) and "If the saved gift card can cover that, I'd like to
// go with two bags" (a conditional question). So a bare "yes" is not
// enough: it must be a "yes" that hands over the action, or an
// imperative go-ahead.
souffle::RamDomain user_confirms_action(souffle::SymbolTable* st,
                                        souffle::RecordTable*,
                                        souffle::RamDomain text) {
    std::string t = to_lower(st->decode(text));

    // "yes" bound to a hand-over, not to a preference.
    static const char* const kAffirmed[] = {
        "yes, please", "yes please", "yes, go", "yes go ahead",
        "yes, that's correct", "yes that's correct",
        "yes, that is correct", "yes, that's the correct",
        "yes, that's right", "yes, correct", "yes, proceed",
        "yes, i'd like to go ahead", "yes, i want to proceed",
        "yes, let's", "yes, confirm", "yes, book", "yes, cancel",
        "yes, apply", "yes, do", "yes—please", "yes - please",
        "yes, i confirm", "yes, i'd like to proceed",
        "yes, that's correct", "yes, i'd like to go",
    };
    if (any_of(t, kAffirmed)) return 1;

    // Imperative go-aheads. "please let me know" / "please make sure"
    // are deliberately absent: they are requests for information,
    // not authorisation to write.
    static const char* const kGoAhead[] = {
        "go ahead", "please proceed", "proceed with the",
        "proceed with this", "proceed with that", "please continue",
        "i confirm", "that's confirmed", "confirmed.",
        "please book", "please make the booking", "please cancel",
        "please complete", "please finalize", "please finalise",
        "please apply", "please make the change",
        "please make these changes", "please make those changes",
        "please process", "please update the reservation",
        "please go with", "let's do it", "lets do it",
        "let's proceed", "sounds good, please", "that works, please",
        "please submit", "please execute",
    };
    if (any_of(t, kGoAhead)) return 1;

    return 0;
}

// 1 iff a user turn states a reason for cancelling. The airline
// policy requires the agent to obtain one (change of plan, airline
// cancelled, or other reasons) before calling the cancel API.
souffle::RamDomain user_states_cancel_reason(souffle::SymbolTable* st,
                                             souffle::RecordTable*,
                                             souffle::RamDomain text) {
    std::string t = to_lower(st->decode(text));
    static const char* const kReasons[] = {
        "change of plan", "changed my plan", "change in plan",
        "plans changed", "plans have changed", "no longer need",
        "no longer able", "don't need the", "do not need the",
        "can't make it", "cannot make it", "won't be able to travel",
        "airline cancelled", "airline canceled",
        "flight was cancelled", "flight was canceled",
        "flight got cancelled", "flight got canceled",
        "cancelled my flight", "canceled my flight",
        "sick", "ill", "illness", "health", "medical", "hospital",
        "doctor", "injury", "injured", "emergency", "family matter",
        "funeral", "passed away", "weather", "storm", "hurricane",
        "snow", "flooding", "schedule conflict", "scheduling conflict",
        "work conflict", "conflict with", "double booked",
        "booked by mistake", "made a mistake", "wrong flight",
        "wrong date", "accidentally booked", "personal reasons",
        "personal circumstances", "visa", "passport",
    };
    return any_of(t, kReasons) ? 1 : 0;
}

// 1 iff the stated reason falls under travel-insurance coverage:
// health-related, weather-related, or other significant personal
// circumstances (the policy names schedule conflicts and change of
// plans as examples of the latter).
souffle::RamDomain covered_cancel_reason(souffle::SymbolTable* st,
                                         souffle::RecordTable*,
                                         souffle::RamDomain text) {
    std::string t = to_lower(st->decode(text));
    static const char* const kCovered[] = {
        // health
        "sick", "ill", "illness", "health", "medical", "hospital",
        "doctor", "injury", "injured", "surgery", "emergency",
        // weather
        "weather", "storm", "hurricane", "snow", "flooding", "flood",
        // significant personal circumstances
        "schedule conflict", "scheduling conflict", "work conflict",
        "conflict with", "change of plan", "changed my plan",
        "change in plan", "plans changed", "plans have changed",
        "family matter", "family emergency", "funeral",
        "passed away", "personal circumstances", "personal reasons",
        "bereavement",
    };
    return any_of(t, kCovered) ? 1 : 0;
}

// 1 iff the user explicitly asks for compensation. The policy
// forbids proactive offers, so a certificate needs this evidence.
souffle::RamDomain user_requests_compensation(souffle::SymbolTable* st,
                                              souffle::RecordTable*,
                                              souffle::RamDomain text) {
    std::string t = to_lower(st->decode(text));
    static const char* const kAsk[] = {
        "compensation", "compensate", "compensated",
        "travel certificate", "certificate for", "voucher",
        "credit for the", "make it right", "make up for",
        "something for the trouble", "reimburse", "reimbursement",
        "what can you do for me", "any goodwill", "goodwill gesture",
    };
    return any_of(t, kAsk) ? 1 : 0;
}

// 1 iff the user turn talks about checked bags at all — the
// evidence that a bag count on a call was actually asked for.
souffle::RamDomain user_mentions_bags(souffle::SymbolTable* st,
                                      souffle::RecordTable*,
                                      souffle::RamDomain text) {
    std::string t = to_lower(st->decode(text));
    static const char* const kBags[] = {
        "bag", "bags", "baggage", "luggage", "suitcase", "suitcases",
        "check a case", "checked item", "carry-on", "carry on",
    };
    return any_of(t, kBags) ? 1 : 0;
}

/* ── 2. JSON structure helpers ──────────────────────────────────── */

// 1 iff the flight list in `args` is the same set of
// (flight_number, date) pairs as the flight list in `res`.
// A cabin-only change therefore reads as "flights unchanged", which
// is what lets a basic-economy reservation change cabin while its
// flights stay put.
souffle::RamDomain flights_same(souffle::SymbolTable* st,
                                souffle::RecordTable*,
                                souffle::RamDomain args,
                                souffle::RamDomain res) {
    auto a = flight_keys(st->decode(args));
    auto b = flight_keys(st->decode(res));
    if (a.empty() || b.empty()) return 0;
    return a == b ? 1 : 0;
}

// 1 iff every traveller on the reservation has been replaced: no
// passenger in `args` shares a last name or a date of birth with any
// passenger in `res`. Correcting a name or a typo keeps one of the
// two, so this fires only on a wholesale substitution of who flies.
souffle::RamDomain passengers_substituted(souffle::SymbolTable* st,
                                          souffle::RecordTable*,
                                          souffle::RamDomain args,
                                          souffle::RamDomain res) {
    auto a = passenger_keys(st->decode(args));
    auto b = passenger_keys(st->decode(res));
    if (a.empty() || b.empty()) return 0;
    for (const auto& x : a) {
        for (const auto& y : b) {
            if (!x.first.empty() && x.first == y.first) return 0;
            if (!x.second.empty() && x.second == y.second) return 0;
        }
    }
    return 1;
}

// 1 iff any flight in the reservation `res` has already been flown:
// its status says so, or its date is strictly before `cutoff`
// (an ISO date, compared lexicographically — valid for YYYY-MM-DD).
souffle::RamDomain any_flight_flown(souffle::SymbolTable* st,
                                    souffle::RecordTable*,
                                    souffle::RamDomain res,
                                    souffle::RamDomain cutoff) {
    std::string cut = st->decode(cutoff);
    bool flown = false;
    each_array_elem(st->decode(res), "flights", [&](const char* elem) {
        std::string status = to_lower(object_str(elem, "status"));
        if (status == "flying" || status == "landed") flown = true;
        std::string date = object_str(elem, "date");
        if (!date.empty() && !cut.empty() && date < cut) flown = true;
    });
    return flown ? 1 : 0;
}

// 1 iff some flight in the reservation `res` carries `status`
// (e.g. "cancelled" or "delayed").
souffle::RamDomain res_flight_status_present(souffle::SymbolTable* st,
                                             souffle::RecordTable*,
                                             souffle::RamDomain res,
                                             souffle::RamDomain status) {
    std::string want = to_lower(st->decode(status));
    bool hit = false;
    each_array_elem(st->decode(res), "flights", [&](const char* elem) {
        if (to_lower(object_str(elem, "status")) == want) hit = true;
    });
    return hit ? 1 : 0;
}

// 1 iff some payment id used by the call `args` is absent from the
// user profile `profile` (whose "payment_methods" is a map keyed by
// payment id). Returns 0 when the call names no payment method or
// the profile carries no payment map — absence of evidence must not
// deny.
souffle::RamDomain payment_missing_from_profile(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain args, souffle::RamDomain profile) {
    auto ids = arg_payment_ids(st->decode(args));
    if (ids.empty()) return 0;
    auto known = object_keys(st->decode(profile), "payment_methods");
    if (known.empty()) return 0;
    for (const auto& id : ids) {
        if (std::find(known.begin(), known.end(), id) == known.end())
            return 1;
    }
    return 0;
}

// How many payment methods in `args` have ids starting with
// `prefix` ("credit_card", "gift_card", "certificate").
souffle::RamDomain count_payment_prefix(souffle::SymbolTable* st,
                                        souffle::RecordTable*,
                                        souffle::RamDomain args,
                                        souffle::RamDomain prefix) {
    std::string pre = st->decode(prefix);
    auto ids = arg_payment_ids(st->decode(args));
    souffle::RamDomain n = 0;
    for (const auto& id : ids) {
        if (id.compare(0, pre.size(), pre) == 0) ++n;
    }
    return n;
}

// 1 iff `haystack` (any tool payload: a reservation, a
// search_direct_flight array, a search_onestop_flight array of
// arrays) describes the flight numbered `fnum`.
souffle::RamDomain flight_known(souffle::SymbolTable* st,
                                souffle::RecordTable*,
                                souffle::RamDomain haystack,
                                souffle::RamDomain fnum) {
    std::string want = st->decode(fnum);
    if (want.empty()) return 0;
    std::string hay = st->decode(haystack);
    bool hit = false;
    walk_objects(hay.c_str(), [&](const char* obj) {
        if (object_str(obj, "flight_number") == want) hit = true;
    });
    return hit ? 1 : 0;
}

// 1 iff `haystack` says flight `fnum` departs from or arrives at
// `airport`. Used to test whether a proposed itinerary still serves
// the reservation's origin and destination.
souffle::RamDomain flight_serves(souffle::SymbolTable* st,
                                 souffle::RecordTable*,
                                 souffle::RamDomain haystack,
                                 souffle::RamDomain fnum,
                                 souffle::RamDomain airport) {
    std::string want = st->decode(fnum);
    std::string ap = st->decode(airport);
    if (want.empty() || ap.empty()) return 0;
    std::string hay = st->decode(haystack);
    bool hit = false;
    walk_objects(hay.c_str(), [&](const char* obj) {
        if (object_str(obj, "flight_number") != want) return;
        if (object_str(obj, "origin") == ap
            || object_str(obj, "destination") == ap) {
            hit = true;
        }
    });
    return hit ? 1 : 0;
}

// Number of elements in the top-level array field `field` of `json`,
// as a signed number so policy rules can compare it without casts.
souffle::RamDomain json_len(souffle::SymbolTable* st,
                            souffle::RecordTable*,
                            souffle::RamDomain json,
                            souffle::RamDomain field) {
    souffle::RamDomain n = 0;
    each_array_elem(st->decode(json), st->decode(field).c_str(),
                    [&](const char*) { ++n; });
    return n;
}
}  // extern "C"
