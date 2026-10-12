/* Custom content functors for airline_policy.dl (tau2-airline).
 *
 * Three deterministic text checks the policy needs and the shared
 * functor library does not provide:
 *
 *   user_confirms      — is this user turn an explicit "yes"?
 *   asks_compensation  — did the user ask for compensation?
 *   date_before        — is ISO date `a` earlier than ISO date `b`?
 *
 * The phrase tables are grounded in the captured run-4 traces: every
 * phrase below either appears in a real confirming turn in
 * outputs/variant-set/run4/scenarios.json, or is an immediate
 * variant of one. Two turns in those traces are the reason the
 * tables are shaped the way they are:
 *
 *   "Please go ahead and use my certificates first, and then pay the
 *    remaining balance with my 7447 card."      (NOT a confirmation —
 *    the user is directing the payment split, and the agent had never
 *    listed the booking for approval)
 *   "...can you make sure I get a full refund?" (NOT a confirmation —
 *    "sure" here is part of "make sure")
 *
 * So "go ahead" is absent from the tables entirely, and the softer
 * affirmatives ("ok", "sure", "of course", ...) only count when they
 * open the message, where they are unambiguously an answer to the
 * agent's question.
 */

#include <souffle/SouffleInterface.h>

#include <algorithm>
#include <cctype>
#include <string>

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
    return std::isalpha(u) != 0;
}

// Substring search that will not fire inside a longer word, so
// "yes" does not match "eyes" and "ok" does not match "book".
bool contains_word(const std::string& hay, const std::string& needle) {
    if (needle.empty()) return false;
    size_t pos = hay.find(needle);
    while (pos != std::string::npos) {
        bool left_ok = (pos == 0) || !is_word_char(hay[pos - 1]);
        size_t end = pos + needle.size();
        bool right_ok = (end >= hay.size()) || !is_word_char(hay[end]);
        if (left_ok && right_ok) return true;
        pos = hay.find(needle, pos + 1);
    }
    return false;
}

// Drop leading whitespace and the quoting / bullet characters that
// show up at the start of transcribed turns.
std::string lstrip(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        char c = s[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
            c == '"' || c == '\'' || c == '*' || c == '-') {
            ++i;
        } else {
            break;
        }
    }
    return s.substr(i);
}

bool starts_with_word(const std::string& s, const std::string& word) {
    if (s.size() < word.size()) return false;
    if (s.compare(0, word.size(), word) != 0) return false;
    return s.size() == word.size() || !is_word_char(s[word.size()]);
}

// Unambiguous affirmations — count anywhere in the turn.
const char* kAffirmAnywhere[] = {
    "yes", "yeah", "yep", "yup",
    "confirmed", "i confirm", "please confirm",
    "i agree", "agreed",
    "sounds good", "sounds perfect", "sounds great",
    "that works", "that's correct", "that is correct",
    "absolutely", "please proceed", "let's do it",
    nullptr,
};

// Softer affirmations — only count as an answer, i.e. when they
// open the turn.
const char* kAffirmLeading[] = {
    "ok", "okay", "sure", "correct", "confirm", "confirmed",
    "of course", "certainly", "perfect", "great", "agreed", "yes",
    nullptr,
};

const char* kCompensation[] = {
    "compensation", "compensate", "compensated",
    "voucher", "certificate", "credit",
    "reimburse", "reimbursement",
    "miles", "points",
    "make this right", "make it right",
    "what can you offer", "anything else you can offer",
    "goodwill",
    nullptr,
};

// The date portion of an ISO 8601 value, e.g. "2024-05-20" from
// "2024-05-20T09:00:00". Fixed-width, so a lexicographic compare of
// two of these is a chronological compare.
std::string iso_date(const std::string& s) {
    if (s.size() < 10) return std::string();
    std::string d = s.substr(0, 10);
    if (d[4] != '-' || d[7] != '-') return std::string();
    for (size_t i : {0u, 1u, 2u, 3u, 5u, 6u, 8u, 9u}) {
        if (std::isdigit(static_cast<unsigned char>(d[i])) == 0) {
            return std::string();
        }
    }
    return d;
}

}  // namespace

extern "C" {

souffle::RamDomain user_confirms(souffle::SymbolTable* st,
                                 souffle::RecordTable*,
                                 souffle::RamDomain text) {
    const std::string lower = to_lower(st->decode(text));
    for (const char** p = kAffirmAnywhere; *p != nullptr; ++p) {
        if (contains_word(lower, *p)) return 1;
    }
    const std::string head = lstrip(lower);
    for (const char** p = kAffirmLeading; *p != nullptr; ++p) {
        if (starts_with_word(head, *p)) return 1;
    }
    return 0;
}

souffle::RamDomain asks_compensation(souffle::SymbolTable* st,
                                     souffle::RecordTable*,
                                     souffle::RamDomain text) {
    const std::string lower = to_lower(st->decode(text));
    for (const char** p = kCompensation; *p != nullptr; ++p) {
        if (contains_word(lower, *p)) return 1;
    }
    return 0;
}

souffle::RamDomain date_before(souffle::SymbolTable* st,
                               souffle::RecordTable*,
                               souffle::RamDomain a,
                               souffle::RamDomain b) {
    const std::string da = iso_date(st->decode(a));
    const std::string db = iso_date(st->decode(b));
    // An unparseable date is never reported as past: the policy uses
    // this to decide that an itinerary has already been flown, and
    // guessing "yes" there would deny cancellations it should allow.
    if (da.empty() || db.empty()) return 0;
    return da < db ? 1 : 0;
}

}  // extern "C"
