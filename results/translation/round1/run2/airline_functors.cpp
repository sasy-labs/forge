/* Custom C++ functors for the tau2-airline policy.

   Three groups:

   1. Content classifiers over customer turns (explicit confirmation,
      cancellation reason, insurance-covered reason, airline-cancelled
      claim). Case-insensitive phrase matching, with typographic
      apostrophes folded to ASCII — the user simulator writes "don't"
      with U+2019, and the phrase tables are written with ASCII '.
      Short tokens ("yes", "ok") match on word boundaries so "eyes"
      and "book" do not count as agreement.

   2. Domain arithmetic: the free checked-bag allowance per passenger,
      which is a table of (membership level x cabin class) in the
      written policy.

   3. One structural helper the shared functors cannot express: a
      lexicographic ISO-date comparison. The shared @hours_between
      needs a full timestamp and returns an absolute difference, so it
      cannot order two bare flight dates.

   Classifiers are deliberately conservative: when nothing matches they
   return 0, and the rules are written so that 0 means "not proven",
   never "proven safe".
*/

#include <souffle/SouffleInterface.h>

#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

namespace {

// ── Text utilities ──────────────────────────────────────────────

// Lowercase, folding U+2018 / U+2019 (typographic single quotes) to
// ASCII '. Captured customer turns use curly apostrophes.
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
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

// Substring match anywhere.
bool has_phrase(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}

// Substring match bracketed by non-word characters — for short tokens
// where a bare find() would fire inside a longer word.
bool has_word(const std::string& hay, const char* needle) {
    const std::string n(needle);
    if (n.empty()) return false;
    std::size_t pos = hay.find(n);
    while (pos != std::string::npos) {
        const bool left = pos == 0 || !is_word_char(hay[pos - 1]);
        const std::size_t end = pos + n.size();
        const bool right = end >= hay.size() || !is_word_char(hay[end]);
        if (left && right) return true;
        pos = hay.find(n, pos + 1);
    }
    return false;
}

// Word match that ignores occurrences following one of `bad_prefixes`.
// "sure" is agreement; "make sure" / "not sure" is not.
bool has_word_not_after(const std::string& hay, const char* needle,
                        const char* const* bad_prefixes) {
    const std::string n(needle);
    if (n.empty()) return false;
    std::size_t pos = hay.find(n);
    while (pos != std::string::npos) {
        const bool left = pos == 0 || !is_word_char(hay[pos - 1]);
        const std::size_t end = pos + n.size();
        const bool right = end >= hay.size() || !is_word_char(hay[end]);
        if (left && right) {
            bool excluded = false;
            for (const char* const* p = bad_prefixes; *p; ++p) {
                const std::size_t len = std::string(*p).size();
                if (pos >= len && hay.compare(pos - len, len, *p) == 0) {
                    excluded = true;
                    break;
                }
            }
            if (!excluded) return true;
        }
        pos = hay.find(n, pos + 1);
    }
    return false;
}

bool any_phrase(const std::string& hay,
                const char* const* needles) {
    for (const char* const* p = needles; *p; ++p) {
        if (has_phrase(hay, *p)) return true;
    }
    return false;
}

bool any_word(const std::string& hay, const char* const* needles) {
    for (const char* const* p = needles; *p; ++p) {
        if (has_word(hay, *p)) return true;
    }
    return false;
}

// ── Phrase tables ───────────────────────────────────────────────

// An explicit affirmative. Short forms are matched as whole words.
const char* const kAffirmWords[] = {
    "yes", "yeah", "yep", "yup", "ok", "okay", "correct",
    "confirmed", "affirmative", nullptr};
// "sure" only counts as agreement outside these idioms.
const char* const kSureExclusions[] = {
    "make ", "makes ", "made ", "not ", "for ", "be ", "am ",
    "n't ", "nt ", "being ", "want to ", "wanted to ", nullptr};
const char* const kAffirmPhrases[] = {
    "sounds good", "that works", "that's fine", "thats fine",
    "that is fine", "i confirm", "you may", "please do", nullptr};

// An instruction to carry the action out now. Deliberately excludes
// bare "confirm" ("my confirmation number is ..." is an identifier,
// not consent) and excludes "let's go with X" (a choice among
// options, not authorization to execute).
const char* const kProceedPhrases[] = {
    "go ahead", "proceed", "do it", "book it", "cancel it",
    "process it", "make the change", "make those changes",
    "make the changes", "i'm ready", "im ready", "i am ready",
    "please book", "please cancel", "please update", "please add",
    "please remove", "please apply", "please charge", nullptr};

// A stated reason for cancelling.
const char* const kCancelReasonPhrases[] = {
    "change of plan", "changed my plan", "plans changed",
    "plans have changed", "no longer need", "no longer want",
    "no longer traveling", "no longer travelling", "not traveling",
    "not travelling", "by mistake", "made a mistake", "booked wrong",
    "wrong date", "wrong flight", "double booked", "cancelled by",
    "canceled by", "was cancelled", "was canceled", "airline cancelled",
    "airline canceled", "schedule conflict", "scheduling conflict",
    "work conflict", "conflict with", "family emergency", "emergency",
    "health", "medical", "sick", "illness", "injury",
    "hospital", "doctor", "weather", "storm", "snow", "hurricane",
    "flood", "personal reason", "personal circumstance",
    "can't make it", "cannot make it", "won't be able",
    "will not be able", "trip is off", "trip got cancelled", nullptr};
const char* const kCancelReasonWords[] = {
    "mistake", "ill", nullptr};

// Reasons the travel insurance covers: health, weather, and other
// significant personal circumstances (schedule conflicts, change of
// plans).
const char* const kCoveredReasonPhrases[] = {
    "health", "medical", "sick", "illness", "injury", "hospital",
    "doctor", "surgery", "weather", "storm", "snow", "hurricane",
    "flood", "fog", "schedule conflict", "scheduling conflict",
    "work conflict", "conflict with", "change of plan",
    "changed my plan", "plans changed", "plans have changed",
    "family emergency", "emergency", "personal reason",
    "personal circumstance", "bereavement", "funeral", nullptr};
const char* const kCoveredReasonWords[] = {"ill", nullptr};

// The customer claiming the airline cancelled a flight. Requires a
// past form, so "I want to cancel my flight" does not match.
const char* const kAirlineCancelledPhrases[] = {
    "airline cancelled", "airline canceled", "flight was cancelled",
    "flight was canceled", "flight got cancelled",
    "flight got canceled", "you cancelled my", "you canceled my",
    "cancelled by the airline", "canceled by the airline",
    "cancelled my flight", "canceled my flight",
    "cancelled our flight", "canceled our flight", nullptr};

// The customer asking for compensation. Compensation is never offered
// proactively, so this is what makes the topic admissible.
const char* const kCompensationPhrases[] = {
    "compensation", "compensate", "voucher", "credit for the",
    "make it right", "something for the trouble",
    "something for the inconvenience", "certificate for",
    "any compensation", "reimburse", "make up for", nullptr};

}  // namespace

extern "C" {

/**
 * @user_confirms(text) -> 0/1
 *
 * 1 iff the customer turn carries BOTH an explicit affirmative and an
 * instruction to go ahead. The written policy asks for "explicit user
 * confirmation (yes) to proceed", so a bare restatement of a wish
 * ("I'd like to book option 2") and a bare affirmative aimed at some
 * other step ("Yes, please search for the return flight") both read
 * as 0.
 */
souffle::RamDomain user_confirms(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string t = to_lower(st->decode(text_sym));
    const bool affirm =
        any_word(t, kAffirmWords) || any_phrase(t, kAffirmPhrases)
        || has_word_not_after(t, "sure", kSureExclusions);
    if (!affirm) return 0;
    return any_phrase(t, kProceedPhrases) ? 1 : 0;
}

/**
 * @states_cancel_reason(text) -> 0/1
 * 1 iff the customer turn states a reason for cancelling.
 */
souffle::RamDomain states_cancel_reason(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string t = to_lower(st->decode(text_sym));
    if (any_phrase(t, kCancelReasonPhrases)) return 1;
    return any_word(t, kCancelReasonWords) ? 1 : 0;
}

/**
 * @states_covered_reason(text) -> 0/1
 * 1 iff the stated reason is one travel insurance covers: health,
 * weather, or another significant personal circumstance.
 */
souffle::RamDomain states_covered_reason(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string t = to_lower(st->decode(text_sym));
    if (any_phrase(t, kCoveredReasonPhrases)) return 1;
    return any_word(t, kCoveredReasonWords) ? 1 : 0;
}

/**
 * @says_airline_cancelled(text) -> 0/1
 * 1 iff the customer says the airline cancelled a flight.
 */
souffle::RamDomain says_airline_cancelled(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string t = to_lower(st->decode(text_sym));
    return any_phrase(t, kAirlineCancelledPhrases) ? 1 : 0;
}

/**
 * @asks_for_compensation(text) -> 0/1
 * 1 iff the customer turn asks for compensation.
 */
souffle::RamDomain asks_for_compensation(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string t = to_lower(st->decode(text_sym));
    return any_phrase(t, kCompensationPhrases) ? 1 : 0;
}

/**
 * @free_bags(membership, cabin) -> free checked bags per passenger
 *
 * The allowance table from the written policy. Returns -1 when either
 * input is unknown, so rules can tell "no allowance" (0) apart from
 * "cannot compute" (-1) and refuse to judge the second case.
 */
souffle::RamDomain free_bags(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain membership_sym,
    souffle::RamDomain cabin_sym)
{
    const std::string m = to_lower(st->decode(membership_sym));
    const std::string c = to_lower(st->decode(cabin_sym));

    int base;
    if (m == "regular")     base = 0;
    else if (m == "silver") base = 1;
    else if (m == "gold")   base = 2;
    else return -1;

    int cabin_bonus;
    if (c == "basic_economy")  cabin_bonus = 0;
    else if (c == "economy")   cabin_bonus = 1;
    else if (c == "business")  cabin_bonus = 2;
    else return -1;

    return static_cast<souffle::RamDomain>(base + cabin_bonus);
}

/**
 * @date_before(a, b) -> 0/1
 *
 * 1 iff ISO date/timestamp `a` denotes a strictly earlier day than
 * `b`. Compares the leading YYYY-MM-DD lexicographically, which is
 * chronological for ISO dates. Returns 0 if either side is not a
 * 10-character-or-longer ISO date, so an unparseable value never
 * makes a flight look flown.
 */
souffle::RamDomain date_before(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain a_sym, souffle::RamDomain b_sym)
{
    const std::string& a = st->decode(a_sym);
    const std::string& b = st->decode(b_sym);
    if (a.size() < 10 || b.size() < 10) return 0;
    return a.compare(0, 10, b, 0, 10) < 0 ? 1 : 0;
}

}  // extern "C"
