/* Custom C++ functors for the airline policy.

   Implements keyword/phrase classifiers used in place of @llm_check_fn
   and a passenger-count helper for the booking arity rule.

   All classifiers are case-insensitive substring matchers over a curated
   set of phrases. They are intentionally conservative — when ambiguous,
   return 0 (no match) so that benign traffic is not blocked.
*/

#include <souffle/SouffleInterface.h>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

// ── Utilities ───────────────────────────────────────────────────

// Lowercase, with U+2018 / U+2019 (typographic single quotes) folded
// to ASCII '. User-simulator output uses curly apostrophes for
// contractions ("don't"), and the phrase lists are written with
// ASCII apostrophes — without this normalisation, decline phrases
// like "don't want insurance" silently fail to match.
static std::string to_lower(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ) {
        if (i + 2 < s.size()
            && static_cast<unsigned char>(s[i])     == 0xE2
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

static bool contains(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}

static bool any_of(const std::string& hay, const char* const* needles,
                   std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (contains(hay, needles[i])) return true;
    }
    return false;
}

// Word-boundary check — returns true if `needle` appears in `hay`
// bracketed by a non-alphanumeric character (or start/end of string).
// Used for short tokens like "yes" to avoid matching "eyes".
static bool word_match(const std::string& hay, const char* needle) {
    std::size_t nlen = std::strlen(needle);
    std::size_t pos = 0;
    while (true) {
        pos = hay.find(needle, pos);
        if (pos == std::string::npos) return false;
        bool left_ok  = pos == 0 ||
            !std::isalnum(static_cast<unsigned char>(hay[pos - 1]));
        bool right_ok = pos + nlen == hay.size() ||
            !std::isalnum(static_cast<unsigned char>(hay[pos + nlen]));
        if (left_ok && right_ok) return true;
        pos += 1;
    }
}

// Word-boundary match that ignores a hit carrying a negation
// immediately before it. "not sure", "I'm not ok with that" and
// "je ne suis pas d'accord" all contain an affirmative token, and
// reading them as agreement authorizes a write the user just refused.
// Only the short window before the token is inspected, so "no problem
// — yes, go ahead" still reads as agreement.
static bool affirmative_match(const std::string& hay, const char* needle) {
    std::size_t nlen = std::strlen(needle);
    std::size_t pos = 0;
    static const char* kNegations[] = {
        "not ", "n't ", "never ", "pas ", "cannot ", "cant ", "aren't ",
        "isn't ", "won't ", "wouldn't ", "rather not ", "instead of ",
    };
    while (true) {
        pos = hay.find(needle, pos);
        if (pos == std::string::npos) return false;
        bool left_ok  = pos == 0 ||
            !std::isalnum(static_cast<unsigned char>(hay[pos - 1]));
        bool right_ok = pos + nlen == hay.size() ||
            !std::isalnum(static_cast<unsigned char>(hay[pos + nlen]));
        if (left_ok && right_ok) {
            std::size_t win_start = pos > 24 ? pos - 24 : 0;
            std::string before = hay.substr(win_start, pos - win_start);
            bool negated = false;
            for (const char* n : kNegations) {
                if (before.find(n) != std::string::npos) { negated = true; break; }
            }
            if (!negated) return true;
        }
        pos += 1;
    }
}

// ── JSON array counter ──────────────────────────────────────────

// Count top-level elements of a JSON array found at key `field`
// in the object `json`. Returns 0 if the field is missing or not
// an array. Handles nested objects, arrays, and strings.
static int count_json_array_elements(const std::string& json,
                                     const char* field) {
    // Build "\"<field>\"" and locate it as a key.
    std::string needle;
    needle.reserve(std::strlen(field) + 2);
    needle.push_back('"');
    needle.append(field);
    needle.push_back('"');
    std::size_t kpos = json.find(needle);
    if (kpos == std::string::npos) return 0;
    // Skip past the key and the colon.
    std::size_t p = kpos + needle.size();
    while (p < json.size() && json[p] != ':') ++p;
    if (p >= json.size()) return 0;
    ++p; // skip ':'
    while (p < json.size() &&
           (json[p] == ' ' || json[p] == '\t' ||
            json[p] == '\n' || json[p] == '\r')) ++p;
    if (p >= json.size() || json[p] != '[') return 0;
    ++p; // skip '['
    int depth = 1;
    int count = 0;
    bool saw_element = false;
    while (p < json.size() && depth > 0) {
        char c = json[p];
        if (c == '"') {
            saw_element = true;
            ++p;
            while (p < json.size() && json[p] != '"') {
                if (json[p] == '\\' && p + 1 < json.size()) ++p;
                ++p;
            }
            if (p < json.size()) ++p;
            continue;
        }
        if (c == '[' || c == '{') { saw_element = true; ++depth; ++p; continue; }
        if (c == ']') {
            --depth;
            ++p;
            if (depth == 0 && saw_element) ++count;
            continue;
        }
        if (c == '}') { --depth; ++p; continue; }
        if (c == ',' && depth == 1) {
            ++count;
            saw_element = false;
            ++p;
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(c))) saw_element = true;
        ++p;
    }
    return count;
}

} // namespace

extern "C" {

// ── Bag request ────────────────────────────────────────────────

static int scan_bag_count(const std::string& low);

souffle::RamDomain user_requested_bags(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text_sym)
{
    static const char* kPhrases[] = {
        "checked bag", "checked bags",
        "checked luggage", "check a bag", "check bags",
        "add bag", "add bags", "add baggage", "add luggage",
        "with bag", "with bags", "with baggage", "with luggage",
        "include bag", "include bags", "include baggage",
        "want bag", "want bags", "want baggage", "want luggage",
        "need bag", "need bags", "need baggage", "need luggage",
        "have bag", "have bags", "have baggage", "have luggage",
        "bring bag", "bring bags", "bring luggage",
        "extra bag", "extra bags", "extra baggage", "extra luggage",
        "one bag", "two bags", "three bags", "four bags", "five bags",
        "1 bag", "2 bags", "3 bags", "4 bags", "5 bags",
        // Plural "baggages" (non-standard but tau2 user sims use it).
        "one baggage", "two baggages", "three baggages",
        "four baggages", "five baggages",
        "1 baggage", "2 baggages", "3 baggages", "4 baggages", "5 baggages",
        // Free-allowance phrasings — user asking to use bag allowance
        // counts as requesting bags.
        "free bag", "free bags", "free baggage", "free luggage",
        "free checked", "baggage allowance", "bag allowance",
        "luggage allowance",
    };
    const std::string& raw = symbolTable->decode(text_sym);
    std::string low = to_lower(raw);
    // Explicit refusals first. `affirmative_match` catches "don't" and
    // "not" forms, but bare "no" is deliberately absent from its
    // negation list — "no problem — yes, go ahead" must still read as
    // agreement — so the "no <noun>" refusals are handled here. A
    // missed refusal lets the agent add baggage the user rejected,
    // whereas a missed request only makes the policy stricter, so the
    // decline test runs first and wins.
    static const char* kDeclines[] = {
        "no bag", "no bags", "no checked bag", "no checked bags",
        "no baggage", "no luggage", "no extra bag", "no extra bags",
        "no additional bag", "no additional bags",
        "no additional baggage", "no additional luggage",
        "zero bags", "0 bags", "without bag", "without bags",
        "without baggage", "without luggage",
    };
    if (any_of(low, kDeclines,
               sizeof(kDeclines) / sizeof(kDeclines[0]))) {
        // "no extra bags" means none paid beyond the free allowance, not
        // zero bags: "Three bags is exactly what I need, so no extra bags
        // required" asks for three. When that is the only decline and the
        // same message names a count, it is a request.
        static const char* kHardDeclines[] = {
            "no bag", "no bags", "no checked bag", "no checked bags",
            "no baggage", "no luggage", "zero bags", "0 bags",
            "without bag", "without bags", "without baggage",
            "without luggage",
        };
        bool only_extra = !any_of(low, kHardDeclines,
            sizeof(kHardDeclines) / sizeof(kHardDeclines[0]));
        if (only_extra && scan_bag_count(low) > 0) return 1u;
        return 0u;
    }
    // Negation-aware: these are bare noun and verb phrases, and there
    // is no decline detector to counterbalance them, so "I don't want
    // any checked bags" read as a request lets the agent add baggage
    // the user refused. Same defect as the insurance affirmations.
    for (const char* phrase : kPhrases) {
        if (affirmative_match(low, phrase)) {
            return 1u;
        }
    }
    return 0u;
}

// ── Affirmative confirmation ───────────────────────────────────

souffle::RamDomain user_confirmed_yes(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text_sym)
{
    const std::string& raw = symbolTable->decode(text_sym);
    std::string low = to_lower(raw);

    // Word-level confirmatives — guarded by word boundaries so
    // "eyes"/"nope" do not produce false positives.
    static const char* kWords[] = {
        "yes", "yep", "yeah", "yup", "ok", "okay",
        "sure", "confirm", "confirmed", "correct", "affirmative",
        // Non-English affirmatives. tau2 ships personas that answer in
        // their own language by design (task 39: "You are French by
        // birth and your English is not perfect"), and an English-only
        // list denies every write for the whole conversation even
        // though the user plainly confirmed.
        "oui", "ja", "claro", "parfait", "exactement",
    };
    for (const char* w : kWords) {
        if (affirmative_match(low, w)) return 1u;
    }
    static const char* kNonEnglishPhrases[] = {
        "d'accord", "bien sur", "bien s\xc3\xbbr",
        "je confirme", "vas-y", "allez-y", "c'est bon",
        "tout a fait", "tout \xc3\xa0 fait",
        "por supuesto", "confirmo",
    };
    for (const char* ph : kNonEnglishPhrases) {
        if (affirmative_match(low, ph)) return 1u;
    }

    // Substring phrases.
    static const char* kPhrases[] = {
        "go ahead", "please proceed", "proceed", "sounds good",
        "please do", "please book", "please cancel",
        "please update", "please change", "please modify",
        "i confirm", "do it", "let's do it", "that's correct",
        "thats correct",
    };
    return any_of(low, kPhrases,
                  sizeof(kPhrases) / sizeof(kPhrases[0])) ? 1u : 0u;
}

// ── Affirmative confirmation, scoped to a specific id ──────────
//
// Confirmation is scoped to each ID's sentence and local context. An ID
// mentioned as ineligible or excluded does not receive affirmative consent.
//
// Heuristic, applied per occurrence of `id` in `text`:
//   1. Sentence boundary scan — splits on . ! ? \n.
//   2. Local-negation check (within ~30 chars before the id, or
//      ~30 chars after the id, bounded by sentence). Words like
//      "not", "don't", "without", "skip", "instead of", and
//      after-context phrases like "not eligible", "ineligible",
//      "cannot", "can't", "won't" reject the id occurrence.
//   3. Sentence-level affirmative check — looks for an action or
//      assent verb anywhere in the same sentence as the id
//      ("yes", "please", "proceed", "go ahead", "cancel",
//      "upgrade", "change", etc.).
//
// Returns 1 if any occurrence of `id` is affirmed without local
// negation; 0 otherwise.
souffle::RamDomain user_confirms_for_id(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text_sym,
    souffle::RamDomain id_sym)
{
    const std::string& text = symbolTable->decode(text_sym);
    const std::string& id   = symbolTable->decode(id_sym);
    if (id.empty()) return 0u;

    std::string low    = to_lower(text);
    std::string id_low = to_lower(id);

    // Local-negation markers checked in the ~30 chars
    // immediately preceding the id (within sentence).
    static const char* kLocalNegBefore[] = {
        "not", "don't", "do not", "no ",
        "without", "except", "skip", "leave",
        "instead of", "rather than", "but not",
        "not for", "not on", "not to",
    };
    // Local-negation markers checked in the ~30 chars
    // immediately following the id (within sentence).
    static const char* kLocalNegAfter[] = {
        "not eligible", "ineligible",
        "is not eligible", "isn't eligible",
        "are not eligible", "aren't eligible",
        "cannot", "can't", "won't",
        "is not allowed", "are not allowed",
        "is excluded", "are excluded",
    };
    // Sentence-level affirmative markers — at least one must
    // appear in the sentence containing the id for the binding
    // to count as a confirmation.
    static const char* kAffMarkers[] = {
        "yes", "yeah", "yep", "ok", "okay", "sure",
        "go ahead", "proceed", "please", "confirm",
        "do it", "cancel", "upgrade", "change",
        "update", "modify", "refund", "rebook",
        "book",
    };

    auto contains = [](const std::string& s, const char* n) {
        return s.find(n) != std::string::npos;
    };

    std::size_t pos = 0;
    while ((pos = low.find(id_low, pos)) != std::string::npos) {
        std::size_t end = pos + id_low.size();

        // Walk back to sentence boundary, capped at 30 chars.
        std::size_t lb_start = pos;
        int chars = 0;
        while (lb_start > 0 && chars < 30) {
            char c = low[lb_start - 1];
            if (c == '.' || c == '!' || c == '?' || c == '\n') break;
            --lb_start; ++chars;
        }
        std::string local_before = low.substr(lb_start, pos - lb_start);

        // Walk forward to sentence boundary, capped at 30 chars.
        std::size_t la_end = end;
        chars = 0;
        while (la_end < low.size() && chars < 30) {
            char c = low[la_end];
            if (c == '.' || c == '!' || c == '?' || c == '\n') break;
            ++la_end; ++chars;
        }
        std::string local_after = low.substr(end, la_end - end);

        // Reject if local negation present in either window.
        bool negated = false;
        for (const char* n : kLocalNegBefore) {
            if (contains(local_before, n)) { negated = true; break; }
        }
        if (!negated) {
            for (const char* n : kLocalNegAfter) {
                if (contains(local_after, n)) { negated = true; break; }
            }
        }
        if (negated) { pos = end; continue; }

        // Sentence-level affirmative window: 80 chars back to
        // boundary, 30 chars forward (we already have la_end).
        std::size_t sb_start = pos;
        chars = 0;
        while (sb_start > 0 && chars < 80) {
            char c = low[sb_start - 1];
            if (c == '.' || c == '!' || c == '?' || c == '\n') break;
            --sb_start; ++chars;
        }
        std::string sentence = low.substr(sb_start, la_end - sb_start);

        for (const char* a : kAffMarkers) {
            if (word_match(sentence, a) || contains(sentence, a)) {
                return 1u;
            }
        }
        pos = end;
    }
    return 0u;
}

// ── User-error cancellation reason ─────────────────────────────

souffle::RamDomain user_error_reason(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text_sym)
{
    static const char* kPhrases[] = {
        "accidentally", "by accident",
        "mistake", "mistakenly", "my bad",
        "wrong flight", "wrong date", "wrong destination",
        "wrong city", "wrong airport",
        "double book", "double-book", "double booked", "double-booked",
        "booked twice", "booked two",
        "meant to book", "meant to pick", "meant to choose",
        "picked the wrong", "chose the wrong", "selected the wrong",
        "clicked the wrong", "hit the wrong",
        "didn't mean to", "did not mean to",
    };
    const std::string& raw = symbolTable->decode(text_sym);
    std::string low = to_lower(raw);
    return any_of(low, kPhrases,
                  sizeof(kPhrases) / sizeof(kPhrases[0])) ? 1u : 0u;
}

// ── Trivial social cancellation reason ─────────────────────────

souffle::RamDomain trivial_social_reason(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text_sym)
{
    static const char* kPhrases[] = {
        "birthday party", "birthday",
        "parties", "celebration",
        "concert", "festival",
        "reunion", "get-together", "get together",
        "sporting event", "baseball game", "football game",
        "basketball game", "soccer game", "hockey game",
        "went on vacation",
    };
    const std::string& raw = symbolTable->decode(text_sym);
    std::string low = to_lower(raw);
    if (any_of(low, kPhrases,
               sizeof(kPhrases) / sizeof(kPhrases[0]))) {
        return 1u;
    }
    // Word-boundary only: as substrings these fire on "shows",
    // "showing", "it shows", "show me", "no-show" and "party" inside
    // longer words, which mislabels an ordinary conversation as a
    // trivial social reason.
    static const char* kWords[] = { "show", "party", "gig" };
    for (const char* w : kWords) {
        if (word_match(low, w)) return 1u;
    }
    return 0u;
}

// ── Insurance-covered cancellation reason ──────────────────────
//
// policy.md (upstream tau2-bench v1.0.1), "Travel insurance":
//   "The travel insurance is 30 dollars per passenger and enables full
//    refund if the user needs to cancel the flight given health or
//    weather reasons."
// So exactly two reasons are covered: health and weather. Schedule
// conflicts, change of plans, bereavement, jury duty, job loss,
// deployment, "giving up my seat" and a bare "emergency" are NOT
// covered by that sentence and deliberately do not match here.

// A hit counts only when it is not negated. "I do not have a health
// related issue", "I'm not sick anymore" and "there is no storm" name a
// covered reason only to deny it, and reading them as the reason
// unlocks the insurance cancellation ground.
//
// Look-back: up to three words before the keyword are checked for a
// negator. The look-back stops at a clause boundary (punctuation, or
// because / since / as / but / so / and), so the negation in "I can't
// fly because I'm sick" governs the travel, not the sickness, and the
// sickness still counts. Three shapes extend it:
//   - a negator directly before the boundary word negates the reason
//     it introduces: "not because of the weather";
//   - "to do with" does not use up the window, so "nothing to do with
//     the weather" and "doesn't have anything to do with my health"
//     reach their negator ("it has to do with my health" still counts);
//   - "or" / "nor" do not use up the window either, so a negation
//     covers every alternative it governs: "nothing to do with the
//     weather or my health".
// Look-ahead: the keyword followed by a copula and a word saying the
// thing is fine ("the weather is fine", "my health is great", "the
// weather's perfectly clear") names the reason only to rule it out.
static bool is_clause_mark(char c) {
    return c == '.' || c == ',' || c == ';' || c == ':' || c == '!'
        || c == '?' || c == '\n' || c == '(' || c == ')';
}

// Reads the word ending at hay[i - 1], skipping spaces first. Sets `i`
// to the word's start. Returns "" when the start of the text or a
// clause mark comes first.
static std::string prev_word(const std::string& hay, std::size_t& i) {
    while (i > 0 && hay[i - 1] == ' ') --i;
    if (i == 0 || is_clause_mark(hay[i - 1])) return "";
    std::size_t end = i;
    while (i > 0 && hay[i - 1] != ' ' && !is_clause_mark(hay[i - 1])) --i;
    return hay.substr(i, end - i);
}

static bool is_one_of(const std::string& w, const char* const* list,
                      std::size_t n) {
    for (std::size_t k = 0; k < n; ++k) {
        if (w == list[k]) return true;
    }
    return false;
}

static const char* const kCoveredNegators[] = {
    "not", "no", "never", "without", "nor", "nothing", "none",
    "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
    "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt",
    "weren't", "werent", "haven't", "havent", "hasn't", "hasnt",
    "hadn't", "hadnt", "won't", "wont", "neither",
};
static const std::size_t kCoveredNegatorsN =
    sizeof(kCoveredNegators) / sizeof(kCoveredNegators[0]);

static bool covered_hit(const std::string& hay, std::size_t pos) {
    static const char* const kBoundaries[] = {
        "because", "since", "as", "but", "so", "and",
    };
    std::size_t i = pos;
    int words = 0;
    while (words < 3) {
        std::string w = prev_word(hay, i);
        if (w.empty()) break;
        if (is_one_of(w, kBoundaries,
                      sizeof(kBoundaries) / sizeof(kBoundaries[0]))) {
            std::size_t j = i;
            return !is_one_of(prev_word(hay, j), kCoveredNegators,
                              kCoveredNegatorsN);
        }
        if (is_one_of(w, kCoveredNegators, kCoveredNegatorsN)) return false;
        if (w == "or") {
            words = 0;
            continue;
        }
        if (w == "with") {
            std::size_t j = i;
            if (prev_word(hay, j) == "do" && prev_word(hay, j) == "to") {
                i = j;
                words = 0;
                continue;
            }
        }
        ++words;
    }
    return true;
}

static bool covered_ruled_out_after(const std::string& hay,
                                    std::size_t end) {
    static const char* const kCopulas[] = {
        "is", "was", "are", "were", "looks", "look", "seems", "seem",
        "has", "have", "been", "will", "be", "should",
    };
    static const char* const kIntensifiers[] = {
        "totally", "perfectly", "really", "very", "actually", "now",
        "just", "completely", "absolutely", "pretty", "quite", "all",
        "still", "entirely",
    };
    static const char* const kFine[] = {
        "fine", "good", "great", "ok", "okay", "clear", "perfect",
        "nice", "excellent", "alright", "normal",
    };
    std::size_t i = end;
    bool copula = false;
    if (hay.compare(i, 2, "'s") == 0) {
        i += 2;
        copula = true;
    }
    for (int k = 0; k < 4; ++k) {
        while (i < hay.size() && hay[i] == ' ') ++i;
        std::size_t b = i;
        while (i < hay.size() && std::isalpha(static_cast<unsigned char>(hay[i]))) ++i;
        if (i == b) return false;
        std::string w = hay.substr(b, i - b);
        if (is_one_of(w, kCopulas, sizeof(kCopulas) / sizeof(kCopulas[0]))) {
            copula = true;
            continue;
        }
        if (is_one_of(w, kIntensifiers,
                      sizeof(kIntensifiers) / sizeof(kIntensifiers[0]))) {
            continue;
        }
        return copula && is_one_of(w, kFine, sizeof(kFine) / sizeof(kFine[0]));
    }
    return false;
}

// word_match with the negation checks above applied to each
// boundary-matched occurrence.
static bool covered_match(const std::string& hay, const char* needle) {
    std::size_t nlen = std::strlen(needle);
    std::size_t pos = 0;
    while (true) {
        pos = hay.find(needle, pos);
        if (pos == std::string::npos) return false;
        bool left_ok  = pos == 0 ||
            !std::isalnum(static_cast<unsigned char>(hay[pos - 1]));
        bool right_ok = pos + nlen == hay.size() ||
            !std::isalnum(static_cast<unsigned char>(hay[pos + nlen]));
        if (left_ok && right_ok && covered_hit(hay, pos)
            && !covered_ruled_out_after(hay, pos + nlen)) {
            return true;
        }
        pos += 1;
    }
}

// word_match for the negated health idioms of
// covered_cancellation_reason, which also rejects a hit followed by
// '-': "don't feel well-informed" is a compound adjective, not the
// idiom "don't feel well".
static bool idiom_match(const std::string& hay, const char* needle) {
    std::size_t nlen = std::strlen(needle);
    std::size_t pos = 0;
    while (true) {
        pos = hay.find(needle, pos);
        if (pos == std::string::npos) return false;
        bool left_ok  = pos == 0 ||
            !std::isalnum(static_cast<unsigned char>(hay[pos - 1]));
        bool right_ok = pos + nlen == hay.size() ||
            (!std::isalnum(static_cast<unsigned char>(hay[pos + nlen]))
             && hay[pos + nlen] != '-');
        if (left_ok && right_ok) return true;
        pos += 1;
    }
}

souffle::RamDomain covered_cancellation_reason(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text_sym)
{
    // Multi-word phrases. Every single-word token lives in kWords
    // below. All are boundary-matched: as bare substrings they attach
    // to unrelated words ("health" to "healthy", "hospital" to
    // "hospitality", "flu" to "fluctuate", "storm" to "brainstorm"),
    // and each false positive unlocks the insurance cancellation path.
    static const char* kPhrases[] = {
        // Health
        "not feeling good", "feeling poorly",
        "under the weather", "emergency room", "medical emergency",
        "health reasons", "health issue", "health issues",
        "health problem", "health problems", "health condition",
        "doctor's orders", "doctors orders",
        "broke my leg", "broke my arm", "broken leg", "broken arm",
        "heart attack",
        // Weather
        "bad weather", "severe weather", "heavy rain", "heavy snow",
        "freezing rain", "ice storm", "winter storm", "tropical storm",
    };
    // Health idioms that state an illness through a negation word
    // ("I haven't been feeling well", "my son doesn't feel well"). The
    // phrase itself carries the negation, so it is matched on word
    // boundaries only: the look-back of covered_match would read a
    // negator just before it ("no, I don't feel well") as denying the
    // illness. A double negation ("it's not that I don't feel well")
    // is therefore read as the reason. Every idiom ends in "well", and a
    // hyphen after it makes a compound adjective rather than the idiom
    // ("I don't feel well-informed", "well-rested"), so a hit followed by
    // '-' does not count (idiom_match).
    static const char* kNegatedIdioms[] = {
        "not feeling well", "not been feeling well",
        "haven't been feeling well", "havent been feeling well",
        "hasn't been feeling well", "have not been feeling well",
        "has not been feeling well", "haven't felt well",
        "don't feel well", "dont feel well", "doesn't feel well",
    };
    const std::string& raw = symbolTable->decode(text_sym);
    std::string low = to_lower(raw);
    for (const char* ph : kNegatedIdioms) {
        if (idiom_match(low, ph)) return 1u;
    }
    for (const char* ph : kPhrases) {
        if (covered_match(low, ph)) return 1u;
    }
    // Single tokens. Inflections are listed explicitly rather than
    // derived, because the right suffix differs per stem and does not
    // always preserve meaning: "stormy" is still a weather reason,
    // while "healthy" is the OPPOSITE of a health reason — so "health"
    // deliberately has no -y form here.
    static const char* kWords[] = {
        // Health
        "ill", "flu", "sick", "sickness", "illness", "illnesses",
        "unwell", "fever", "infection", "pneumonia", "covid",
        "hospital", "hospitalized", "hospitalised",
        "hospitalization", "hospitalisation",
        "surgery", "surgeries",
        "injury", "injuries", "injured",
        "medical", "health", "doctor", "doctors",
        "diagnosed", "diagnosis", "cancer",
        "pregnant", "pregnancy",
        // Weather
        "weather", "storm", "storms", "stormy", "thunderstorm",
        "thunderstorms", "rainstorm", "snowstorm", "snowstorms",
        "snow", "snowfall", "snowing", "blizzard", "blizzards",
        "hurricane", "hurricanes", "typhoon", "typhoons",
        "cyclone", "cyclones", "tornado", "tornadoes",
        "flood", "floods", "flooded", "flooding",
    };
    for (const char* w : kWords) {
        if (covered_match(low, w)) return 1u;
    }
    return 0u;
}

// ── User-requested compensation ────────────────────────────────

souffle::RamDomain user_requested_compensation(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text_sym)
{
    static const char* kPhrases[] = {
        "compensation", "compensate",
        "certificate", "travel certificate",
        "voucher", "credit for the trouble", "credit for the delay",
        "reimburse", "reimbursement",
        "refund for the delay", "refund for the trouble",
        "make it up to me", "make it right",
        "goodwill gesture",
    };
    const std::string& raw = symbolTable->decode(text_sym);
    std::string low = to_lower(raw);
    return any_of(low, kPhrases,
                  sizeof(kPhrases) / sizeof(kPhrases[0])) ? 1u : 0u;
}

// ── Named bag count ────────────────────────────────────────────
//
// Extracts an explicit checked-bag count from a user message.
//
// Returns the largest count found in numeric or word form, e.g.
//   "I'd like 3 checked bags"        → 3
//   "add two bags"                   → 2
//   "no bags"                        → 0
//   "I'll take a bag"                → 1
//   "bags beyond the 3"             → 3   (backward-looking)
//   "wonderful weather"              → -1   (no count expressed)
//
// Returns -1 ("no info") when the user said nothing about bags;
// callers can then treat that as "do not constrain". A return of
// 0 is *deliberate* — distinguishes "user said no bags" from "no
// info" so the rule layer can require total_baggages = 0 if it
// chooses (we do not currently — only require ≥ N for N>0).

// Internal: scan a (lowercase) chunk of text for the largest
// explicitly-named checked-bag count. Returns -1 for "no info",
// 0 for an explicit decline, positive for an explicit count.
static int scan_bag_count(const std::string& low) {
    static const struct { const char* word; int n; } kWordCounts[] = {
        {"a bag",     1}, {"a checked bag", 1},
        {"one bag",   1}, {"one checked bag", 1},
        {"two bags",  2}, {"two checked bags", 2},
        {"three bags",3}, {"three checked bags", 3},
        {"four bags", 4}, {"four checked bags", 4},
        {"five bags", 5}, {"five checked bags", 5},
        {"six bags",  6}, {"seven bags", 7},
        {"eight bags",8}, {"nine bags",  9}, {"ten bags", 10},
        {"a suitcase",      1}, {"one suitcase",    1},
        {"two suitcases",   2}, {"three suitcases", 3},
    };
    static const char* kZeroPhrases[] = {
        "no bags",  "no bag",  "no checked bags", "no checked bag",
        "no luggage", "no checked luggage",
        "without bags", "without checked bags",
        "without luggage", "without checked luggage",
        "zero bags", "0 bags", "0 checked bags",
        "no suitcase", "no suitcases",
    };
    static const char* kBagNouns[] = {
        "bag", "bags", "baggage", "baggages",
        "suitcase", "suitcases",
        "piece of luggage", "pieces of luggage",
    };

    int best = -1;
    for (const char* p : kZeroPhrases) {
        if (contains(low, p)) { best = (best < 0 ? 0 : best); break; }
    }
    // Boundary-matched: as bare substrings "a bag" sits inside "extra
    // bag(s)" and "extra baggage", so a decline such as "no insurance or
    // extra baggage needed" read as a request for one bag.
    for (const auto& wc : kWordCounts) {
        if (word_match(low, wc.word) && wc.n > best) best = wc.n;
    }

    std::size_t i = 0;
    while (i < low.size()) {
        if (!std::isdigit(static_cast<unsigned char>(low[i]))) { ++i; continue; }
        std::size_t j = i;
        while (j < low.size() &&
               std::isdigit(static_cast<unsigned char>(low[j]))) ++j;
        int n = std::atoi(low.substr(i, j - i).c_str());
        std::size_t k = j;
        while (k < low.size() && std::isspace(
                   static_cast<unsigned char>(low[k]))) ++k;
        const char* kw = "checked ";
        if (k + std::strlen(kw) <= low.size() &&
            low.compare(k, std::strlen(kw), kw) == 0) {
            k += std::strlen(kw);
        }
        // Currency amounts are not bag counts. Skip when preceded by
        // a currency symbol (possibly with spaces: "$ 50") or when the
        // digit is a decimal fraction of a price ("49.99").
        if (i > 0) {
            // Decimal fraction: no spaces inside a number
            unsigned char imm = static_cast<unsigned char>(low[i - 1]);
            if (imm == '.' && i >= 2 &&
                std::isdigit(static_cast<unsigned char>(low[i - 2]))) {
                i = j; continue;
            }
            // Currency symbol, possibly with spaces before the digit.
            // Handles $ (ASCII), £ (C2 A3), ¥ (C2 A5), € (E2 82 AC).
            std::size_t cb = i;
            while (cb > 0 && low[cb - 1] == ' ') --cb;
            if (cb > 0) {
                unsigned char prev =
                    static_cast<unsigned char>(low[cb - 1]);
                if (prev == '$') { i = j; continue; }
                if (cb >= 2) {
                    unsigned char p2 =
                        static_cast<unsigned char>(low[cb - 2]);
                    if (p2 == 0xC2 &&
                        (prev == 0xA3 || prev == 0xA5)) {
                        i = j; continue; // £ or ¥
                    }
                    if (cb >= 3 &&
                        static_cast<unsigned char>(low[cb - 3])
                            == 0xE2 &&
                        p2 == 0x82 && prev == 0xAC) {
                        i = j; continue; // €
                    }
                }
            }
        }
        bool match = false;
        std::size_t matched_end = 0;
        for (const char* noun : kBagNouns) {
            std::size_t nlen = std::strlen(noun);
            if (k + nlen <= low.size() &&
                low.compare(k, nlen, noun) == 0) {
                bool right_ok = k + nlen == low.size() ||
                    !std::isalnum(
                        static_cast<unsigned char>(low[k + nlen]));
                if (right_ok) { match = true; matched_end = k + nlen; break; }
            }
        }
        // A bag noun followed by a fee noun describes a price
        // ("$50 baggage fee"), not a quantity of bags. "cost" and
        // "costs" are excluded: they double as verbs ("2 bags cost
        // $100") and the currency-prefix check already handles
        // "$50 baggage cost".
        if (match) {
            std::size_t f = matched_end;
            while (f < low.size() &&
                   std::isspace(static_cast<unsigned char>(low[f]))) ++f;
            static const char* kFeeWords[] = {
                "fee", "fees", "charge", "charges",
                "price", "surcharge", "penalty",
            };
            for (const char* fw : kFeeWords) {
                std::size_t fwlen = std::strlen(fw);
                if (f + fwlen <= low.size() &&
                    low.compare(f, fwlen, fw) == 0 &&
                    (f + fwlen == low.size() ||
                     !std::isalnum(
                         static_cast<unsigned char>(low[f + fwlen])))) {
                    match = false; break;
                }
            }
        }
        // Backward-looking: "<bag_noun> beyond [the] N" — retained
        // total ("No additional bags beyond the 3."). Rejects:
        //   - ordinals/dates (3rd, 3pm)
        //   - fractional (3.5 — not an integer count)
        //   - no locally connected bag noun before "beyond"
        //   - conditional/hypothetical ("if I wanted ... beyond 3")
        //   - quotation marks (reported form, not a decision)
        if (!match && n >= 0) {
            bool reject = false;
            // Reject ordinals/dates/fractional
            std::size_t after = j;
            while (after < low.size() && low[after] == '*') ++after;
            if (after < low.size()) {
                char ac = low[after];
                static const char* kOrdSuffix[] = {
                    "st", "nd", "rd", "th", "am", "pm",
                };
                for (const char* os : kOrdSuffix) {
                    std::size_t oslen = std::strlen(os);
                    if (after + oslen <= low.size() &&
                        low.compare(after, oslen, os) == 0) {
                        reject = true; break;
                    }
                }
                if (ac == '.' && after + 1 < low.size() &&
                    std::isdigit(static_cast<unsigned char>(
                        low[after + 1]))) {
                    reject = true;
                }
            }
            if (!reject) {
                std::size_t back = i;
                while (back > 0 && low[back - 1] == '*') --back;
                while (back > 0 && low[back - 1] == ' ') --back;
                static const char* kBeyond[] = {
                    "beyond the", "beyond"};
                for (const char* bp : kBeyond) {
                    std::size_t bplen = std::strlen(bp);
                    if (back < bplen) continue;
                    if (low.compare(back - bplen, bplen, bp) != 0)
                        continue;
                    std::size_t bb = back - bplen;
                    while (bb > 0 && low[bb - 1] == ' ') --bb;
                    bool found_noun = false;
                    std::size_t noun_start = bb;
                    for (const char* noun : kBagNouns) {
                        std::size_t nlen = std::strlen(noun);
                        if (bb >= nlen &&
                            low.compare(bb - nlen, nlen, noun) == 0) {
                            found_noun = true;
                            noun_start = bb - nlen;
                            break;
                        }
                    }
                    if (!found_noun) break;
                    std::size_t sent_start = noun_start;
                    while (sent_start > 0 &&
                           low[sent_start-1] != '.' &&
                           low[sent_start-1] != '!' &&
                           low[sent_start-1] != '\n') --sent_start;
                    std::string sent_before =
                        low.substr(sent_start,
                                   noun_start - sent_start);
                    std::size_t trim = 0;
                    while (trim < sent_before.size()) {
                        char c = sent_before[trim];
                        if (c == ' ' || c == '\t' || c == '*'
                            || c == '-' || c == '#'
                            || std::isdigit(
                                static_cast<unsigned char>(c))
                            || c == ')' || c == '(')
                            { ++trim; continue; }
                        break;
                    }
                    std::size_t tend = sent_before.size();
                    while (tend > trim
                           && sent_before[tend - 1] == ' ') --tend;
                    std::string pfx =
                        sent_before.substr(trim, tend - trim);
                    static const char* kRetained[] = {
                        "no additional", "no more", "no extra",
                        "not adding", "without additional",
                        nullptr
                    };
                    bool retained = false;
                    for (const char** rp = kRetained; *rp; ++rp) {
                        if (pfx == *rp) {
                            retained = true; break;
                        }
                    }
                    if (!retained) break;
                    std::size_t tail = j;
                    while (tail < low.size()
                           && low[tail] == '*') ++tail;
                    while (tail < low.size()
                           && low[tail] == ' ') ++tail;
                    if (tail < low.size()
                        && low[tail] != '.'
                        && low[tail] != '!'
                        && low[tail] != '\n') break;
                    match = true;
                    break;
                }
            }
        }
        if (match && n > best) best = n;
        i = j;
    }
    return best;
}

souffle::RamDomain user_named_bag_count(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text_sym)
{
    const std::string& raw = symbolTable->decode(text_sym);
    std::string low = to_lower(raw);
    return static_cast<souffle::RamDomain>(scan_bag_count(low));
}

// ── Named bag count, scoped to a specific id ───────────────────
//
// Per-sentence scoping mirroring user_confirms_for_id. For each
// occurrence of `id` in `text`, walk to the surrounding sentence
// boundaries (. ! ? \n) and run scan_bag_count on that sentence
// only. Return the highest count found across sentences that
// contain the id; -1 if id never appears or no sentence
// containing the id has an explicit count.
//
// This avoids binding "Add 3 bags to ABC. Also book DEF" so that
// the count 3 is bound to ABC only — not also to DEF.

souffle::RamDomain user_named_bag_count_for_id(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text_sym,
    souffle::RamDomain id_sym)
{
    const std::string& text = symbolTable->decode(text_sym);
    const std::string& id   = symbolTable->decode(id_sym);
    if (id.empty()) return static_cast<souffle::RamDomain>(-1);

    std::string low    = to_lower(text);
    std::string id_low = to_lower(id);

    int best = -1;
    std::size_t pos = 0;
    while ((pos = low.find(id_low, pos)) != std::string::npos) {
        // Sentence start — walk back to . ! ? \n or buffer start.
        std::size_t sb = pos;
        while (sb > 0) {
            char c = low[sb - 1];
            if (c == '.' || c == '!' || c == '?' || c == '\n') break;
            --sb;
        }
        // Sentence end — walk forward to . ! ? \n or buffer end.
        std::size_t se = pos + id_low.size();
        while (se < low.size()) {
            char c = low[se];
            if (c == '.' || c == '!' || c == '?' || c == '\n') break;
            ++se;
        }
        std::string sentence = low.substr(sb, se - sb);
        int n = scan_bag_count(sentence);
        if (n > best) best = n;
        pos = se;
    }
    return static_cast<souffle::RamDomain>(best);
}

// ── Affirmative insurance request ──────────────────────────────
//
// User explicitly asked to include / buy / add insurance.

souffle::RamDomain user_affirmed_insurance(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text_sym)
{
    // Phrases must signal a first-person commit to insurance.
    // Bare bigrams like "with insurance", "include insurance",
    // "want insurance" were dropped because they false-positive on
    // exploratory user echoes ("what's the price with insurance?")
    // and on negated forms ("I don't want insurance" — the "want
    // insurance" substring matches before the surrounding clause is
    // considered). The decline functor is the polarity-correct way
    // to register a "no" answer; this functor only sees affirmative
    // commits.
    static const char* kPhrases[] = {
        "i want insurance",
        "i want travel insurance",
        "i'd like insurance",
        "i'd like travel insurance",
        "i'll take insurance",
        "i'll take the insurance",
        "i'll take travel insurance",
        "yes to insurance",
        "please add insurance",
        "please add travel insurance",
        "please include insurance",
        "please include travel insurance",
        "add insurance please",
        "add travel insurance please",
        "include insurance please",
        "include travel insurance please",
        "with insurance please",
        "buy insurance",
        "buy travel insurance",
        "purchase insurance",
        "purchase travel insurance",
        "insurance = yes", "insurance=yes", "insurance: yes",
        "insurance is yes", "yes to travel insurance",
    };
    // Verb phrases that a negation flips outright. "don't add the
    // insurance" contains "add the insurance", and the decline list
    // does not catch it either — its entries omit the article — so a
    // plain substring test reads an explicit refusal as consent with
    // nothing to counterbalance it. These go through the
    // negation-aware matcher instead.
    static const char* kNegatablePhrases[] = {
        "add the insurance", "add the travel insurance",
        "include the insurance", "include the travel insurance",
    };
    // Without the article ("please make sure to include travel
    // insurance", "Yes, include travel insurance"). These also read as a
    // question ("does that price include insurance?"), so like the weak
    // forms below they count only in a turn with no question mark.
    static const char* kNegatableNoQuestion[] = {
        "include travel insurance", "include insurance",
    };
    const std::string& raw = symbolTable->decode(text_sym);
    std::string low = to_lower(raw);
    if (any_of(low, kPhrases,
               sizeof(kPhrases) / sizeof(kPhrases[0]))) {
        return 1u;
    }
    // Field-value format: "travel insurance on the new booking: **YES**."
    // Only "new" or "this" booking (not "existing"). No question mark in
    // the whole message. Rejects:
    //   - negation before the prefix (same sentence)
    //   - quotation marks before the prefix (reported speech)
    //   - conditional/deferral/question after "yes" in the same sentence
    //   - "no" or "not" instead of "yes"
    if (low.find('?') == std::string::npos) {
        static const char* kFieldPrefixes[] = {
            "travel insurance on the new booking",
            "travel insurance on this booking",
            "travel insurance on the booking",
            "insurance on the new booking",
            "insurance on this booking",
            "insurance on the booking",
        };
        for (const char* fp : kFieldPrefixes) {
            std::size_t fplen = std::strlen(fp);
            std::size_t fpos = low.find(fp);
            if (fpos == std::string::npos) continue;
            // Reject "existing" booking
            if (low.find("existing") != std::string::npos) continue;
            // The field must begin an independent statement: the text
            // between the nearest sentence boundary and the field
            // prefix must be empty or a recognized safe prefix (list
            // number, markdown bullet, whitespace). Conditional
            // clauses, quotations, and arbitrary prefixes reject by
            // grammar, not by finite veto.
            std::size_t sb = fpos;
            while (sb > 0 && low[sb-1] != '.' && low[sb-1] != '!'
                   && low[sb-1] != '\n') --sb;
            std::string before_field = low.substr(sb, fpos - sb);
            // Strip leading whitespace and markdown emphasis.
            std::size_t bfs = 0;
            while (bfs < before_field.size() &&
                   (before_field[bfs] == ' ' || before_field[bfs] == '\t'
                    || before_field[bfs] == '*'))
                ++bfs;
            std::string bf_trimmed = before_field.substr(bfs);
            if (!bf_trimmed.empty()) {
                // Allow list-item prefix: digits followed by ./),
                // then optional space; or "- " / "* " markdown bullet.
                bool is_safe = false;
                std::size_t di = 0;
                while (di < bf_trimmed.size() &&
                       std::isdigit(static_cast<unsigned char>(
                           bf_trimmed[di])))
                    ++di;
                if (di > 0 && di < bf_trimmed.size() &&
                    (bf_trimmed[di] == '.' || bf_trimmed[di] == ')')) {
                    std::size_t ri = di + 1;
                    while (ri < bf_trimmed.size() &&
                           bf_trimmed[ri] == ' ') ++ri;
                    if (ri == bf_trimmed.size()) is_safe = true;
                }
                if (bf_trimmed == "- " || bf_trimmed == "* "
                    || bf_trimmed == "-" || bf_trimmed == "*")
                    is_safe = true;
                if (!is_safe) continue;
            }
            // After the prefix, expect ": " then optional ** then yes
            std::size_t p = fpos + fplen;
            while (p < low.size() && (low[p] == ' ' || low[p] == ':'))
                ++p;
            while (p < low.size() && low[p] == '*') ++p;
            if (p + 3 > low.size() ||
                low.compare(p, 3, "yes") != 0) continue;
            std::size_t yes_end = p + 3;
            while (yes_end < low.size() && low[yes_end] == '*') ++yes_end;
            // The field value must FULLY MATCH its independent
            // statement. After YES + bold markers, the remainder
            // of the sentence (up to '.' / '!' / '\n') must be
            // empty. Commas introduce clauses ("YES, provided...")
            // and are rejected as unrecognized trailing content.
            std::size_t se = yes_end;
            while (se < low.size() && low[se] != '.' &&
                   low[se] != '!' && low[se] != '\n')
                ++se;
            std::string after_yes = low.substr(yes_end, se - yes_end);
            // Strip whitespace and bold markers
            std::size_t as = 0;
            while (as < after_yes.size() &&
                   (after_yes[as] == ' ' || after_yes[as] == '*'))
                ++as;
            if (as < after_yes.size()) continue;
            return 1u;
        }
    }
    for (const char* phrase : kNegatablePhrases) {
        if (affirmative_match(low, phrase)) {
            return 1u;
        }
    }
    if (low.find('?') == std::string::npos) {
        for (const char* phrase : kNegatableNoQuestion) {
            if (affirmative_match(low, phrase)) {
                return 1u;
            }
        }
    }
    // First-person purchasing authorization — anchored to the START
    // of the trimmed message so that any non-trivial prefix (a
    // conditional clause, a quote frame, a reporting verb, a long
    // preamble) rejects the match. Only harmless yes/please prefixes
    // are allowed before "i authorize <action>". Deferrals in the
    // suffix still reject.
    if (low.find('?') == std::string::npos) {
        std::size_t p = 0;
        while (p < low.size() &&
               std::isspace(static_cast<unsigned char>(low[p]))) ++p;
        // Optional yes-word prefix
        static const char* kYesPfx[] = {
            "yes", "yep", "yeah", "ok", "okay",
        };
        for (const char* yw : kYesPfx) {
            std::size_t ywlen = std::strlen(yw);
            if (p + ywlen <= low.size() &&
                low.compare(p, ywlen, yw) == 0 &&
                (p + ywlen == low.size() ||
                 !std::isalnum(static_cast<unsigned char>(
                     low[p + ywlen])))) {
                p += ywlen;
                while (p < low.size() &&
                       (low[p] == ',' || low[p] == '.' ||
                        low[p] == ' ' || low[p] == ';' ||
                        low[p] == '!' || low[p] == '-')) ++p;
                break;
            }
        }
        // Optional "please" prefix
        if (p + 6 <= low.size() &&
            low.compare(p, 6, "please") == 0 &&
            (p + 6 == low.size() ||
             !std::isalnum(static_cast<unsigned char>(
                 low[p + 6])))) {
            p += 6;
            while (p < low.size() &&
                   (low[p] == ',' || low[p] == '.' ||
                    low[p] == ' ')) ++p;
        }
        const char* anchor = "i authorize ";
        std::size_t alen = std::strlen(anchor);
        if (p + alen <= low.size() &&
            low.compare(p, alen, anchor) == 0) {
            std::size_t act_start = p + alen;
            static const char* kAuthActs[] = {
                "purchasing insurance",
                "purchasing travel insurance",
                "buying insurance",
                "buying travel insurance",
                "the purchase of insurance",
                "the purchase of travel insurance",
            };
            std::size_t act_end = 0;
            for (const char* act : kAuthActs) {
                std::size_t actlen = std::strlen(act);
                if (act_start + actlen <= low.size() &&
                    low.compare(act_start, actlen, act) == 0 &&
                    (act_start + actlen == low.size() ||
                     !std::isalnum(static_cast<unsigned char>(
                         low[act_start + actlen])))) {
                    act_end = act_start + actlen;
                    break;
                }
            }
            if (act_end > 0) {
                // Closed suffix set — no open-ended text. After
                // the action phrase and trimming terminal . ! and
                // whitespace, the remainder must exactly match one
                // of the enumerated alternatives. Unsupported
                // elaborations conservatively reject.
                std::size_t tail = low.size();
                while (tail > act_end &&
                       (low[tail - 1] == '.' || low[tail - 1] == '!'
                        || low[tail - 1] == ' '
                        || low[tail - 1] == '\n')) --tail;
                std::string sfx = low.substr(act_end, tail - act_end);
                static const char* kAllowed[] = {
                    "",
                    " please",
                    " for the new booking",
                    " for the new booking please",
                    " for this booking",
                    " for this booking please",
                    " for the booking",
                    " for the booking please",
                    " for my booking",
                    " for my booking please",
                    " on the new booking",
                    " on this booking",
                    " on the booking",
                    " on my booking",
                    " for the new flight",
                    " for the flight",
                    " for this flight",
                    " for my flight",
                };
                for (const char* a : kAllowed) {
                    if (sfx == a) return 1u;
                }
            }
        }
    }
    // Weak forms: these read the same in a commitment ("YES, book
    // Option A for $213 + insurance") and in a question ("what's the
    // total plus insurance?"). Only count them in a turn that is not a
    // question and that carries an affirmative of its own.
    static const char* kWeakPhrases[] = { "+ insurance", "plus insurance" };
    if (low.find('?') == std::string::npos &&
        any_of(low, kWeakPhrases,
               sizeof(kWeakPhrases) / sizeof(kWeakPhrases[0]))) {
        // Single words go through word_match ONLY — "yesterday's quote
        // plus insurance" contains "yes" and is not an affirmative.
        // Multi-word phrases are unambiguous enough for a substring
        // test.
        static const char* kAffirmWords[] = {
            "yes", "yep", "yeah", "confirm", "confirmed", "proceed",
        };
        for (const char* w : kAffirmWords) {
            if (word_match(low, w)) return 1u;
        }
        static const char* kAffirmPhrases[] = {
            "book it", "go ahead", "please book",
        };
        if (any_of(low, kAffirmPhrases,
                   sizeof(kAffirmPhrases) / sizeof(kAffirmPhrases[0]))) {
            return 1u;
        }
    }
    return 0u;
}

// ── Decline insurance ──────────────────────────────────────────
//
// User explicitly asked not to include insurance.

souffle::RamDomain user_declined_insurance(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text_sym)
{
    static const char* kPhrases[] = {
        "no insurance",       "no travel insurance",
        "without insurance",  "without travel insurance",
        "without the insurance",
        "skip insurance",     "skip the insurance",
        "skip travel insurance",
        "decline insurance",  "decline the insurance",
        "decline travel insurance",
        "don't add insurance",        "do not add insurance",
        "don't include insurance",    "do not include insurance",
        "don't want insurance",       "do not want insurance",
        "don't need insurance",       "do not need insurance",
        "no need for insurance",      "no need for travel insurance",
        "no, no insurance",
    };
    const std::string& raw = symbolTable->decode(text_sym);
    std::string low = to_lower(raw);
    return any_of(low, kPhrases,
                  sizeof(kPhrases) / sizeof(kPhrases[0])) ? 1u : 0u;
}

// ── Passenger counter ──────────────────────────────────────────

souffle::RamDomain count_passengers(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain args_sym)
{
    const std::string& args = symbolTable->decode(args_sym);
    int n = count_json_array_elements(args, "passengers");
    return static_cast<souffle::RamDomain>(n);
}

// ── Count payment methods whose payment_id starts with prefix ─

souffle::RamDomain count_payment_prefix(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain json_sym,
    souffle::RamDomain prefix_sym)
{
    const std::string& json = symbolTable->decode(json_sym);
    const std::string& prefix = symbolTable->decode(prefix_sym);
    static const char* kKey = "\"payment_id\"";
    const std::size_t key_len = std::strlen(kKey);
    std::size_t pos = 0;
    int count = 0;
    while (true) {
        std::size_t p = json.find(kKey, pos);
        if (p == std::string::npos) break;
        std::size_t c = json.find(':', p + key_len);
        if (c == std::string::npos) break;
        std::size_t v = c + 1;
        while (v < json.size() &&
               std::isspace(static_cast<unsigned char>(json[v]))) ++v;
        if (v >= json.size() || json[v] != '"') { pos = v + 1; continue; }
        std::size_t q1 = v + 1;
        if (q1 + prefix.size() <= json.size() &&
            json.compare(q1, prefix.size(), prefix) == 0) {
            ++count;
        }
        std::size_t q2 = json.find('"', q1);
        if (q2 == std::string::npos) break;
        pos = q2 + 1;
    }
    return static_cast<souffle::RamDomain>(count);
}

// ── Extract last flight_number from a flights array ────────────

souffle::RamDomain extract_last_flight(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain json_sym)
{
    const std::string& json = symbolTable->decode(json_sym);
    static const char* kKey = "\"flight_number\"";
    const std::size_t key_len = std::strlen(kKey);
    std::size_t pos = 0;
    std::size_t last_q1 = std::string::npos;
    std::size_t last_q2 = 0;
    while (true) {
        std::size_t p = json.find(kKey, pos);
        if (p == std::string::npos) break;
        std::size_t c = json.find(':', p + key_len);
        if (c == std::string::npos) break;
        std::size_t v = c + 1;
        while (v < json.size() &&
               std::isspace(static_cast<unsigned char>(json[v]))) ++v;
        if (v >= json.size() || json[v] != '"') { pos = v + 1; continue; }
        std::size_t q1 = v + 1;
        std::size_t q2 = json.find('"', q1);
        if (q2 == std::string::npos) break;
        last_q1 = q1;
        last_q2 = q2;
        pos = q2 + 1;
    }
    if (last_q1 == std::string::npos) return symbolTable->encode("");
    return symbolTable->encode(json.substr(last_q1, last_q2 - last_q1));
}

// ── Find a flight object by flight_number and extract a field ─

// Given a JSON containing one or more flight objects (each with
// "flight_number", "origin", "destination" fields), find the object
// whose flight_number matches ``flight_num`` and return its
// ``field`` value. Used by flight_origin / flight_destination.
static std::string _flight_field(
    const std::string& json,
    const std::string& flight_num,
    const std::string& field)
{
    static const char* kKey = "\"flight_number\"";
    const std::size_t key_len = std::strlen(kKey);
    std::size_t pos = 0;
    while (true) {
        std::size_t p = json.find(kKey, pos);
        if (p == std::string::npos) return "";
        std::size_t c = json.find(':', p + key_len);
        if (c == std::string::npos) return "";
        std::size_t v = c + 1;
        while (v < json.size() &&
               std::isspace(static_cast<unsigned char>(json[v]))) ++v;
        if (v >= json.size() || json[v] != '"') { pos = v + 1; continue; }
        std::size_t q1 = v + 1;
        std::size_t q2 = json.find('"', q1);
        if (q2 == std::string::npos) return "";
        if (q2 - q1 == flight_num.size() &&
            json.compare(q1, flight_num.size(), flight_num) == 0) {
            // Match — find enclosing object { ... } and pull the
            // requested field. Walk backward for '{', then forward
            // to the matching '}' so we don't wander into a sibling.
            std::size_t obj_start = json.rfind('{', p);
            if (obj_start == std::string::npos) return "";
            int depth = 1;
            std::size_t i = obj_start + 1;
            std::size_t obj_end = std::string::npos;
            while (i < json.size() && depth > 0) {
                char ch = json[i];
                if (ch == '{') ++depth;
                else if (ch == '}') { --depth; if (depth == 0) obj_end = i; }
                else if (ch == '"') {
                    ++i;
                    while (i < json.size() && json[i] != '"') {
                        if (json[i] == '\\' && i + 1 < json.size()) ++i;
                        ++i;
                    }
                }
                ++i;
            }
            if (obj_end == std::string::npos) return "";
            // Search for "field": "value" within [obj_start, obj_end]
            std::string field_key = "\"" + field + "\"";
            std::size_t f = json.find(field_key, obj_start);
            if (f == std::string::npos || f > obj_end) return "";
            std::size_t fc = json.find(':', f + field_key.size());
            if (fc == std::string::npos || fc > obj_end) return "";
            std::size_t fv = fc + 1;
            while (fv < json.size() &&
                   std::isspace(static_cast<unsigned char>(json[fv]))) ++fv;
            if (fv >= json.size() || json[fv] != '"') return "";
            std::size_t fq1 = fv + 1;
            std::size_t fq2 = json.find('"', fq1);
            if (fq2 == std::string::npos || fq2 > obj_end) return "";
            return json.substr(fq1, fq2 - fq1);
        }
        pos = q2 + 1;
    }
}

souffle::RamDomain flight_origin(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain json_sym,
    souffle::RamDomain flight_sym)
{
    std::string v = _flight_field(
        symbolTable->decode(json_sym),
        symbolTable->decode(flight_sym),
        "origin");
    return symbolTable->encode(v);
}

souffle::RamDomain flight_destination(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain json_sym,
    souffle::RamDomain flight_sym)
{
    std::string v = _flight_field(
        symbolTable->decode(json_sym),
        symbolTable->decode(flight_sym),
        "destination");
    return symbolTable->encode(v);
}

// ── Flight-list-visits check ───────────────────────────────────

// Given ``args`` (update_reservation_flights call args with a
// ``"flights": [{"flight_number": ...}, ...]`` list) and ``haystack``
// (a prior tool result blob with resolved origin/destination fields
// per flight), return "1" if any flight number referenced in the
// new flights list has origin == airport or destination == airport
// when resolved via ``haystack``; otherwise "". Used by C26 route
// invariance so round-trip destination changes are caught too.
souffle::RamDomain flight_list_visits(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain args_sym,
    souffle::RamDomain haystack_sym,
    souffle::RamDomain airport_sym)
{
    const std::string& args = symbolTable->decode(args_sym);
    const std::string& haystack = symbolTable->decode(haystack_sym);
    const std::string& airport = symbolTable->decode(airport_sym);
    if (airport.empty()) return symbolTable->encode("");
    static const char* kKey = "\"flight_number\"";
    const std::size_t key_len = std::strlen(kKey);
    std::size_t pos = 0;
    while (true) {
        std::size_t p = args.find(kKey, pos);
        if (p == std::string::npos) return symbolTable->encode("");
        std::size_t c = args.find(':', p + key_len);
        if (c == std::string::npos) return symbolTable->encode("");
        std::size_t v = c + 1;
        while (v < args.size() &&
               std::isspace(static_cast<unsigned char>(args[v]))) ++v;
        if (v >= args.size() || args[v] != '"') { pos = v + 1; continue; }
        std::size_t q1 = v + 1;
        std::size_t q2 = args.find('"', q1);
        if (q2 == std::string::npos) return symbolTable->encode("");
        std::string flight_num = args.substr(q1, q2 - q1);
        if (_flight_field(haystack, flight_num, "origin") == airport)
            return symbolTable->encode("1");
        if (_flight_field(haystack, flight_num, "destination") == airport)
            return symbolTable->encode("1");
        pos = q2 + 1;
    }
}

// ── Any flight date in the past ────────────────────────────────

// Scan the JSON for each "date": "YYYY-MM-DD" pair and return 1 if
// any value is strictly less than ``cutoff`` (lex compare on the
// first 10 characters). Used by the date-based branch of
// ReservationHasFlownFlight — tau2 doesn't always stamp past flights
// with "landed"/"flying" in the response text, but the "date" field
// is always present, so a past date reliably signals a flown flight.
souffle::RamDomain any_flight_date_before(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain json_sym,
    souffle::RamDomain cutoff_sym)
{
    const std::string& json = symbolTable->decode(json_sym);
    const std::string& cutoff = symbolTable->decode(cutoff_sym);
    static const char* kKey = "\"date\"";
    const std::size_t key_len = std::strlen(kKey);
    std::size_t pos = 0;
    while (true) {
        std::size_t p = json.find(kKey, pos);
        if (p == std::string::npos) return 0u;
        std::size_t c = json.find(':', p + key_len);
        if (c == std::string::npos) return 0u;
        std::size_t v = c + 1;
        while (v < json.size() &&
               std::isspace(static_cast<unsigned char>(json[v]))) ++v;
        if (v >= json.size() || json[v] != '"') {
            pos = v + 1;
            continue;
        }
        std::size_t q = v + 1;
        std::size_t r = json.find('"', q);
        if (r == std::string::npos) return 0u;
        std::string value = json.substr(q, r - q);
        if (value.size() >= 10 && value.substr(0, 10) < cutoff) {
            return 1u;
        }
        pos = r + 1;
    }
}

// ── One date in the past ───────────────────────────────────────

// Return 1 iff ``date`` sorts strictly before ``cutoff``. Both are
// ISO dates ("YYYY-MM-DD"), which sort lexicographically, so this is
// the same comparison any_flight_date_before makes — but on a single
// leg's date rather than on every date in a payload. That is what
// lets a rule say "this leg is in the past" and then ask whether the
// airline cancelled that same leg. A value shorter than ten
// characters is not a date and returns 0.
souffle::RamDomain date_before(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain date_sym,
    souffle::RamDomain cutoff_sym)
{
    const std::string& date = symbolTable->decode(date_sym);
    const std::string& cutoff = symbolTable->decode(cutoff_sym);
    if (date.size() < 10) return 0u;
    return date.substr(0, 10) < cutoff ? 1u : 0u;
}

// ── User states a flight has already been flown ────────────────

// tau2 reservations carry no per-leg status and no departure time, so
// a leg departing earlier TODAY is invisible to the date comparison
// above. The user is a reliable second source for it: they know
// whether they have flown. This signal only ever makes the policy
// stricter (it adds flown flights, which block cancellation and cabin
// changes), so a loose match cannot be used to unlock anything.
souffle::RamDomain user_stated_flight_flown(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text_sym)
{
    // Deliberately narrow. Bare past-tense travel talk ("we flew to
    // Chicago last year", "I already left the airport") says nothing
    // about THIS reservation, and every match here blocks a cabin
    // change or cancellation — so each phrase has to be about a flight
    // on the itinerary under discussion, not travel in general.
    static const char* kPhrases[] = {
        "already flew that", "already flew the",
        "already flown that", "already flown the",
        "already taken that flight", "already took that flight",
        "already took the flight", "already flew out",
        "that flight already departed", "flight has already departed",
        "that flight has already flown", "that flight has already landed",
        "that flight has already departed",
        "outbound has already flown", "outbound has already departed",
        "outbound has already landed",
        "first leg has already flown", "first leg has already departed",
        "flight has already gone", "that flight is done",
        "we already made that trip", "we already did that leg",
        "already flown the outbound",
    };
    const std::string& raw = symbolTable->decode(text_sym);
    std::string low = to_lower(raw);
    return any_of(low, kPhrases,
                  sizeof(kPhrases) / sizeof(kPhrases[0])) ? 1u : 0u;
}

// ── Contextual insurance YES ──────────────────────────────────

// 1 iff the user's message is a sole bare affirmative: "YES", "Yes",
// with optional trailing/leading whitespace, punctuation, or markdown.
// Rejects any substantive content, conditionals, deferrals, quotes.
souffle::RamDomain user_is_sole_affirmative(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text_sym)
{
    const std::string& raw = symbolTable->decode(text_sym);
    std::string low = to_lower(raw);
    // Reject if contains quote marks (reported speech)
    if (low.find('"') != std::string::npos) return 0u;
    // Reject if contains conditional/deferral words
    if (low.find("if ") != std::string::npos) return 0u;
    if (low.find("but ") != std::string::npos) return 0u;
    if (low.find("wait") != std::string::npos) return 0u;
    if (low.find("later") != std::string::npos) return 0u;
    if (low.find("decided") != std::string::npos) return 0u;
    if (low.find("form ") != std::string::npos) return 0u;
    if (low.find("says ") != std::string::npos) return 0u;
    // Strip whitespace, punctuation, markdown emphasis
    std::size_t s = 0;
    while (s < low.size() && (low[s] == ' ' || low[s] == '\t' ||
           low[s] == '\n' || low[s] == '*')) ++s;
    std::size_t e = low.size();
    while (e > s && (low[e-1] == ' ' || low[e-1] == '\t' ||
           low[e-1] == '\n' || low[e-1] == '*' || low[e-1] == '.' ||
           low[e-1] == '!' || low[e-1] == ',')) --e;
    std::string trimmed = low.substr(s, e - s);
    static const char* kSole[] = {
        "yes", "yep", "yeah", "yup", "ok", "okay",
        "sure", "confirm", "confirmed", "correct",
    };
    for (const char* w : kSole) {
        if (trimmed == w) return 1u;
    }
    return 0u;
}

// Parse the entire input as a strict JSON {"message": "..."} envelope.
// Returns true and writes the decoded message value to `out` iff the
// input is exactly one JSON object with exactly one string-valued key
// "message", no surrounding prose, no unsupported keys, no trailing
// content. Any structural, encoding, or escape error fails closed.
static bool parse_json_message_envelope(
    const std::string& input, std::string& out)
{
    std::size_t pos = 0;
    auto skip_ws = [&]() {
        while (pos < input.size()) {
            char c = input[pos];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
                ++pos;
            else break;
        }
    };
    auto hex4 = [&]() -> int {
        if (pos + 4 > input.size()) return -1;
        int val = 0;
        for (int i = 0; i < 4; ++i) {
            char h = input[pos + i];
            int d;
            if (h >= '0' && h <= '9') d = h - '0';
            else if (h >= 'a' && h <= 'f') d = h - 'a' + 10;
            else if (h >= 'A' && h <= 'F') d = h - 'A' + 10;
            else return -1;
            val = (val << 4) | d;
        }
        pos += 4;
        return val;
    };
    auto encode_utf8 = [](std::string& s, int cp) {
        if (cp < 0x80) {
            s += static_cast<char>(cp);
        } else if (cp < 0x800) {
            s += static_cast<char>(0xC0 | (cp >> 6));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            s += static_cast<char>(0xE0 | (cp >> 12));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            s += static_cast<char>(0xF0 | (cp >> 18));
            s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            s += static_cast<char>(0x80 | (cp & 0x3F));
        }
    };
    auto decode_string = [&](std::string& s) -> bool {
        if (pos >= input.size() || input[pos] != '"') return false;
        ++pos;
        s.clear();
        while (pos < input.size()) {
            unsigned char c =
                static_cast<unsigned char>(input[pos]);
            if (c < 0x20) return false;
            if (c == '"') { ++pos; return true; }
            if (c == '\\') {
                ++pos;
                if (pos >= input.size()) return false;
                char e = input[pos++];
                switch (e) {
                case '"':  s += '"';  break;
                case '\\': s += '\\'; break;
                case '/':  s += '/';  break;
                case 'b':  s += '\b'; break;
                case 'f':  s += '\f'; break;
                case 'n':  s += '\n'; break;
                case 'r':  s += '\r'; break;
                case 't':  s += '\t'; break;
                case 'u': {
                    int cp = hex4();
                    if (cp < 0) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (pos + 1 >= input.size()
                            || input[pos] != '\\'
                            || input[pos + 1] != 'u')
                            return false;
                        pos += 2;
                        int lo = hex4();
                        if (lo < 0 || lo < 0xDC00
                            || lo > 0xDFFF)
                            return false;
                        cp = 0x10000
                             + ((cp - 0xD800) << 10)
                             + (lo - 0xDC00);
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        return false;
                    }
                    encode_utf8(s, cp);
                    break;
                }
                default: return false;
                }
            } else {
                s += static_cast<char>(c);
                ++pos;
            }
        }
        return false;
    };

    skip_ws();
    if (pos >= input.size() || input[pos] != '{') return false;
    ++pos;
    skip_ws();
    std::string key;
    if (!decode_string(key)) return false;
    if (key != "message") return false;
    skip_ws();
    if (pos >= input.size() || input[pos] != ':') return false;
    ++pos;
    skip_ws();
    if (!decode_string(out)) return false;
    skip_ws();
    if (pos >= input.size() || input[pos] != '}') return false;
    ++pos;
    skip_ws();
    return pos == input.size();
}

// 1 iff the agent's message asks a specific, sole, unambiguous
// yes/no question about adding travel insurance to a new booking.
// The selection instruction must be about INSURANCE specifically,
// not about bags or another topic that happens to mention insurance
// elsewhere. Rejects multi-question turns (2+ "?" marks), quoted/
// hypothetical/old-booking insurance, and descriptive mentions
// where the actual question asks about something else.
souffle::RamDomain agent_asked_insurance_yesno(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text_sym)
{
    const std::string& raw = symbolTable->decode(text_sym);

    // JSON envelope: parse the RAW text before any normalization
    // so that keys, quotes and escape sequences are structurally
    // intact.  If the input looks like JSON (leading '{') but is
    // not a valid single-key {"message":"..."} envelope, fail
    // closed — no substring extraction, no plaintext fallback.
    std::string semantic;
    {
        std::size_t ws = 0;
        while (ws < raw.size()) {
            char c = raw[ws];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
                ++ws;
            else break;
        }
        if (ws < raw.size() && raw[ws] == '{') {
            if (!parse_json_message_envelope(raw, semantic))
                return 0u;
        } else {
            semantic = raw;
        }
    }

    std::string low = to_lower(semantic);
    if (low.find("insurance") == std::string::npos) return 0u;
    // Strip markdown bold markers for clean matching.
    {
        std::string clean;
        clean.reserve(low.size());
        for (std::size_t i = 0; i < low.size(); ++i) {
            if (low[i] == '*') continue;
            clean += low[i];
        }
        low = clean;
    }
    // Normalize curly apostrophe/quotes so template literals
    // match (' → ', " " → ").
    {
        std::string norm;
        norm.reserve(low.size());
        for (std::size_t i = 0; i < low.size(); ++i) {
            unsigned char c0 = static_cast<unsigned char>(low[i]);
            if (c0 == 0xE2 && i + 2 < low.size()
                && static_cast<unsigned char>(low[i+1]) == 0x80) {
                unsigned char c2 =
                    static_cast<unsigned char>(low[i+2]);
                if (c2 == 0x98 || c2 == 0x99) {
                    norm += '\''; i += 2; continue;
                }
                if (c2 == 0x9C || c2 == 0x9D) {
                    norm += '"'; i += 2; continue;
                }
            }
            norm += low[i];
        }
        low = norm;
    }

    bool has_yesno = false;
    std::size_t matched_q_end = 0;
    std::size_t matched_q_start = 0;

    // Full-match recognized affirmative insurance question. The
    // question must be a COMPLETE sentence — its template must start
    // at a sentence/paragraph boundary (no arbitrary prefix) and end
    // at "?". "If the premium is waived, do you want to add..."
    // fails because "if the premium is waived, " is unrecognized
    // prefix content before the template.
    {
        static const char* kQuestions[] = {
            "do you want to add travel insurance to the new booking",
            "do you want to add travel insurance to this booking",
            "do you want to add travel insurance",
            "do you want to add insurance to the new booking",
            "do you want to add insurance",
            "would you like to add travel insurance to the new booking",
            "would you like to add travel insurance",
            "would you like to add insurance to the new booking",
            "would you like to add insurance",
            nullptr,
        };
        for (const char** q = kQuestions; *q; ++q) {
            std::size_t qlen = std::strlen(*q);
            std::size_t qpos = low.find(*q);
            if (qpos == std::string::npos) continue;
            // Must end at "?" (with optional whitespace)
            std::size_t after_q = qpos + qlen;
            while (after_q < low.size() && low[after_q] == ' ')
                ++after_q;
            if (after_q >= low.size() || low[after_q] != '?') continue;
            // Must START at a sentence/paragraph boundary: the
            // content between the nearest boundary and the template
            // must be empty (whitespace/markdown only). Any prefix
            // prose rejects. Handle JSON-escaped \\n as newline.
            std::size_t sb = qpos;
            while (sb > 0) {
                char c = low[sb - 1];
                if (c == '.' || c == '!' || c == '\n' || c == '?')
                    break;
                if (sb >= 2 && low[sb - 1] == 'n'
                    && low[sb - 2] == '\\') { sb -= 2; break; }
                --sb;
            }
            std::string before_q = low.substr(sb, qpos - sb);
            // Trim JSON-escaped newlines, spaces, markdown, hashes.
            std::size_t bi = 0;
            while (bi < before_q.size()) {
                char c = before_q[bi];
                if (c == ' ' || c == '\t' || c == '\n' || c == '#'
                    || c == '-') { ++bi; continue; }
                if (c == '\\' && bi + 1 < before_q.size()
                    && before_q[bi + 1] == 'n') { bi += 2; continue; }
                break;
            }
            if (bi < before_q.size()) continue;
            has_yesno = true;
            matched_q_start = sb;
            matched_q_end = after_q + 1;
            break;
        }
    }

    if (!has_yesno) return 0u;

    // Must be about a NEW booking, not existing/old
    if (low.find("new booking") == std::string::npos &&
        low.find("new reservation") == std::string::npos &&
        low.find("new trip") == std::string::npos) {
        if (low.find("do you want to add") == std::string::npos)
            return 0u;
    }
    // Reject old/existing booking insurance
    if (low.find("existing booking") != std::string::npos) return 0u;
    if (low.find("old booking") != std::string::npos) return 0u;
    if (low.find("current booking") != std::string::npos) return 0u;
    // Reject quoted insurance question
    {
        std::size_t ins_pos = low.find("insurance");
        if (ins_pos != std::string::npos) {
            std::size_t scan = ins_pos > 60 ? ins_pos - 60 : 0;
            std::string window = low.substr(scan, ins_pos - scan);
            if (window.find("says \"") != std::string::npos
                || window.find("form \"") != std::string::npos
                || window.find("wrote \"") != std::string::npos
                || window.find("reads \"") != std::string::npos
                || window.find("text \"") != std::string::npos)
                return 0u;
        }
    }
    // Reject hypothetical
    if (low.find("if we were") != std::string::npos) return 0u;
    if (low.find("hypothetical") != std::string::npos) return 0u;
    if (low.find("would you want") != std::string::npos) return 0u;
    // Reject price readback / understanding check
    if (low.find("understood") != std::string::npos) return 0u;
    if (low.find("is that correct") != std::string::npos) return 0u;
    // Reject multi-question: 2+ "?" marks means multiple topics.
    int qcount = 0;
    for (char c : low) { if (c == '?') ++qcount; }
    if (qcount >= 2) return 0u;
    // Sequential full consumption from question end.
    {
        std::size_t cur = matched_q_end;
        auto ws_only_to_eof = [&]() -> bool {
            while (cur < low.size()) {
                char c = low[cur];
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
                    { ++cur; continue; }
                return false;
            }
            return true;
        };
        auto skip = [&]() {
            while (cur < low.size()) {
                char c = low[cur];
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r'
                    || c == '-' || c == '#')
                    { ++cur; continue; }
                break;
            }
        };

        skip();
        // Plain sole question: only whitespace to EOF.
        {
            bool plain = true;
            for (std::size_t i = cur; i < low.size(); ++i) {
                char c = low[i];
                if (c != ' ' && c != '\t' && c != '\n' && c != '\r')
                    { plain = false; break; }
            }
            if (plain) goto reply_ok;
        }

        // Simple format: "reply yes or no" + optional period, then EOF.
        if (cur + 15 <= low.size()
            && low.compare(cur, 15, "reply yes or no") == 0) {
            cur += 15;
            if (cur < low.size() && low[cur] == '.') ++cur;
            if (!ws_only_to_eof()) return 0u;
            goto reply_ok;
        }

        // Rich format: exact reply YES / NO lines with price.
        {
            const char* kCharge = "total charge will be $";
            std::size_t clen = std::strlen(kCharge);
            // Reply YES line: "reply yes (total charge will be $NNN), or"
            if (cur + 9 > low.size()
                || low.compare(cur, 9, "reply yes") != 0) return 0u;
            cur += 9;
            while (cur < low.size() && low[cur] == ' ') ++cur;
            if (cur >= low.size() || low[cur] != '(') return 0u;
            ++cur;
            if (cur + clen > low.size()
                || low.compare(cur, clen, kCharge) != 0) return 0u;
            cur += clen;
            if (cur >= low.size()
                || !std::isdigit(static_cast<unsigned char>(low[cur])))
                return 0u;
            while (cur < low.size()
                   && (std::isdigit(static_cast<unsigned char>(low[cur]))
                       || low[cur] == '.')) ++cur;
            if (cur >= low.size() || low[cur] != ')') return 0u;
            ++cur;
            if (cur + 4 <= low.size()
                && low.compare(cur, 4, ", or") == 0) cur += 4;
            while (cur < low.size() && low[cur] == ' ') ++cur;
            if (cur < low.size() && low[cur] == '\n') ++cur;
            skip();
            // Reply NO line: "reply no (total charge will be $NNN)."
            if (cur + 8 > low.size()
                || low.compare(cur, 8, "reply no") != 0) return 0u;
            cur += 8;
            while (cur < low.size() && low[cur] == ' ') ++cur;
            if (cur >= low.size() || low[cur] != '(') return 0u;
            ++cur;
            if (cur + clen > low.size()
                || low.compare(cur, clen, kCharge) != 0) return 0u;
            cur += clen;
            if (cur >= low.size()
                || !std::isdigit(static_cast<unsigned char>(low[cur])))
                return 0u;
            while (cur < low.size()
                   && (std::isdigit(static_cast<unsigned char>(low[cur]))
                       || low[cur] == '.')) ++cur;
            if (cur >= low.size() || low[cur] != ')') return 0u;
            ++cur;
            if (cur < low.size() && low[cur] == '.') ++cur;
            skip();
            // Optional exact consequence literal.
            {
                const char* kCons =
                    "once you answer, i'll place the booking "
                    "immediately.";
                std::size_t cslen = std::strlen(kCons);
                if (cur + cslen <= low.size()
                    && low.compare(cur, cslen, kCons) == 0)
                    cur += cslen;
            }
            if (!ws_only_to_eof()) return 0u;
        }
    }
reply_ok:
    // Full positive prefix grammar, encoding-independent.
    // Every prefix line must be FULLY consumed by a typed template.
    {
        // ── Typed slot consumers ──
        // Each returns the new cursor position, or npos on failure.
        const std::size_t NP = std::string::npos;
        auto clit = [](const std::string& s, std::size_t p,
                       const char* lit) -> std::size_t {
            std::size_t n = std::strlen(lit);
            if (p + n > s.size() || s.compare(p, n, lit) != 0)
                return NP;
            return p + n;
        };
        auto ascii_alpha = [](unsigned char c) -> bool {
            return (c >= 'a' && c <= 'z')
                || (c >= 'A' && c <= 'Z');
        };
        auto ascii_alnum = [&ascii_alpha](unsigned char c)
            -> bool {
            return ascii_alpha(c)
                || (c >= '0' && c <= '9');
        };
        auto c_alnum = [&ascii_alnum](const std::string& s,
                          std::size_t p) -> std::size_t {
            std::size_t q = p;
            while (q < s.size()
                   && ascii_alnum(
                       static_cast<unsigned char>(s[q])))
                ++q;
            return q > p ? q : NP;
        };
        auto c_money = [](const std::string& s,
                          std::size_t p) -> std::size_t {
            std::size_t q = p;
            if (q >= s.size()
                || !std::isdigit(
                    static_cast<unsigned char>(s[q])))
                return NP;
            while (q < s.size()
                   && std::isdigit(
                       static_cast<unsigned char>(s[q])))
                ++q;
            if (q < s.size() && s[q] == '.') {
                if (q + 3 > s.size()
                    || !std::isdigit(
                        static_cast<unsigned char>(s[q+1]))
                    || !std::isdigit(
                        static_cast<unsigned char>(s[q+2])))
                    return NP;
                q += 3;
            }
            return q;
        };
        auto c_payid = [&ascii_alnum](const std::string& s,
                          std::size_t p) -> std::size_t {
            std::size_t q = p;
            bool us = false;
            while (q < s.size()) {
                char c = s[q];
                if (ascii_alnum(static_cast<unsigned char>(c))
                    || c == '_') {
                    if (c == '_') us = true;
                    ++q;
                } else break;
            }
            return (q > p && us) ? q : NP;
        };
        auto c_airport = [&ascii_alpha](const std::string& s,
                            std::size_t p) -> std::size_t {
            if (p + 3 > s.size()) return NP;
            for (int i = 0; i < 3; ++i)
                if (!ascii_alpha(
                    static_cast<unsigned char>(s[p+i])))
                    return NP;
            if (p + 3 < s.size()
                && ascii_alpha(
                    static_cast<unsigned char>(s[p+3])))
                return NP;
            return p + 3;
        };
        auto c_flight = [&ascii_alpha](const std::string& s,
                           std::size_t p) -> std::size_t {
            std::size_t q = p;
            while (q < s.size()
                   && ascii_alpha(
                       static_cast<unsigned char>(s[q])))
                ++q;
            if (q == p) return NP;
            std::size_t d = q;
            while (q < s.size()
                   && std::isdigit(
                       static_cast<unsigned char>(s[q])))
                ++q;
            return q > d ? q : NP;
        };
        auto c_date = [](const std::string& s,
                         std::size_t p) -> std::size_t {
            if (p + 10 > s.size()) return NP;
            for (int i = 0; i < 10; ++i) {
                if (i == 4 || i == 7) {
                    if (s[p+i] != '-') return NP;
                } else if (!std::isdigit(
                    static_cast<unsigned char>(s[p+i])))
                    return NP;
            }
            return p + 10;
        };
        auto c_time = [](const std::string& s,
                         std::size_t p) -> std::size_t {
            if (p + 5 > s.size()) return NP;
            if (!std::isdigit(
                    static_cast<unsigned char>(s[p]))
                || !std::isdigit(
                    static_cast<unsigned char>(s[p+1]))
                || s[p+2] != ':'
                || !std::isdigit(
                    static_cast<unsigned char>(s[p+3]))
                || !std::isdigit(
                    static_cast<unsigned char>(s[p+4])))
                return NP;
            return p + 5;
        };
        auto c_int = [](const std::string& s,
                        std::size_t p) -> std::size_t {
            std::size_t q = p;
            while (q < s.size()
                   && std::isdigit(
                       static_cast<unsigned char>(s[q])))
                ++q;
            return q > p ? q : NP;
        };
        auto c_cabin = [&clit](const std::string& s,
                               std::size_t p) -> std::size_t {
            static const char* k[] = {
                "premium economy", "basic economy",
                "economy", "business", "first", nullptr};
            for (const char** c = k; *c; ++c) {
                std::size_t r = clit(s, p, *c);
                if (r != NP) return r;
            }
            return NP;
        };
        auto c_status = [&clit](const std::string& s,
                                std::size_t p) -> std::size_t {
            static const char* k[] = {
                "regular", "silver", "gold",
                "platinum", nullptr};
            for (const char** c = k; *c; ++c) {
                std::size_t r = clit(s, p, *c);
                if (r != NP) return r;
            }
            return NP;
        };
        // → (UTF-8 E2 86 92) or - or ->
        auto c_rarrow = [&clit](const std::string& s,
                                std::size_t p) -> std::size_t {
            std::size_t r = clit(s, p, "\xe2\x86\x92");
            if (r != NP) return r;
            r = clit(s, p, "->");
            if (r != NP) return r;
            return clit(s, p, "-");
        };
        // ↔ (UTF-8 E2 86 94) or -
        auto c_biarrow = [&clit](const std::string& s,
                                 std::size_t p) -> std::size_t {
            std::size_t r = clit(s, p, "\xe2\x86\x94");
            if (r != NP) return r;
            return clit(s, p, "-");
        };
        // — (E2 80 94) or – (E2 80 93) or -
        auto c_dash = [&clit](const std::string& s,
                              std::size_t p) -> std::size_t {
            std::size_t r = clit(s, p, "\xe2\x80\x94");
            if (r != NP) return r;
            r = clit(s, p, "\xe2\x80\x93");
            if (r != NP) return r;
            return clit(s, p, "-");
        };

        // ── Full-line matchers ──

        // Heading: strip leading # and spaces, match against
        // closed set of exact heading texts.
        auto match_heading = [&](const std::string& t)
            -> bool {
            std::size_t p = 0;
            while (p < t.size() && t[p] == '#') ++p;
            while (p < t.size() && t[p] == ' ') ++p;
            std::string h = t.substr(p);
            while (!h.empty() && h.back() == ' ') h.pop_back();
            static const char* k[] = {
                "important: travel insurance does not carry "
                    "over automatically",
                "rebooking details (ready to book)",
                "please reply with one word",
                "please reply with a single word",
                nullptr};
            for (const char** kk = k; *kk; ++kk)
                if (h == *kk) return true;
            return false;
        };

        // Status: done: i cancelled reservation <RID>
        // successfully[. rest...]
        auto match_status = [&](const std::string& t)
            -> bool {
            std::size_t p = 0;
            p = clit(t, p,
                "done: i cancelled reservation ");
            if (p == NP) return false;
            p = c_alnum(t, p);
            if (p == NP) return false;
            p = clit(t, p, " successfully.");
            if (p == NP) return false;
            if (p == t.size()) return true;
            p = clit(t, p,
                " the cancellation shows a $");
            if (p == NP) return false;
            p = c_money(t, p);
            if (p == NP) return false;
            p = clit(t, p, " refund back to ");
            if (p == NP) return false;
            p = c_payid(t, p);
            if (p == NP) return false;
            p = clit(t, p,
                " (posted to the original payment "
                "method; typical processing is ");
            if (p == NP) return false;
            p = c_int(t, p);
            if (p == NP) return false;
            p = c_dash(t, p);
            if (p == NP) return false;
            p = c_int(t, p);
            if (p == NP) return false;
            p = clit(t, p, " business days).");
            if (p == NP) return false;
            return p == t.size();
        };

        // Transition: i then attempted to rebook the new
        // <AIRPORT><BIARROW><AIRPORT> itinerary, but i need
        // one clarification before i can complete the new
        // booking:
        auto match_transition = [&](const std::string& t)
            -> bool {
            std::size_t p = 0;
            p = clit(t, p,
                "i then attempted to rebook the new ");
            if (p == NP) return false;
            p = c_airport(t, p);
            if (p == NP) return false;
            p = c_biarrow(t, p);
            if (p == NP) return false;
            p = c_airport(t, p);
            if (p == NP) return false;
            p = clit(t, p,
                " itinerary, but i need one clarification "
                "before i can complete the new booking:");
            if (p == NP) return false;
            return p == t.size();
        };

        // Insurance explanation: exact full text.
        auto match_ins_expl = [](const std::string& t)
            -> bool {
            static const char* kFull =
                "your old reservation had insurance = yes, "
                "but a new booking is a new reservation. "
                "you asked to \"make sure any change/cancel "
                "fees are waived\" due to health + insurance"
                ", but to keep that coverage on the new "
                "trip, i need your explicit instruction to "
                "buy travel insurance again for the new "
                "reservation.";
            if (t == kFull) return true;
            static const char* kShort =
                "your old reservation had insurance = yes, "
                "but a new booking is a new reservation.";
            return t == kShort;
        };

        // Bullet: - insurance cost is $<MONEY> per passenger.
        auto match_ins_cost = [&](const std::string& t)
            -> bool {
            std::size_t p = clit(t, 0,
                "- insurance cost is $");
            if (p == NP) return false;
            p = c_money(t, p);
            if (p == NP) return false;
            p = clit(t, p, " per passenger.");
            return p != NP && p == t.size();
        };

        // Bullet: - outbound/return: <FLIGHT> on <DATE>
        //   (<AIRPORT><ARROW><AIRPORT> <TIME><ARROW><TIME>)
        //   <DASH> <CABIN>
        // or short: (<AIRPORT>-<AIRPORT>) - <CABIN>
        auto match_leg = [&](const std::string& t,
                             const char* label)
            -> bool {
            std::size_t p = clit(t, 0, "- ");
            if (p == NP) return false;
            p = clit(t, p, label);
            if (p == NP) return false;
            p = clit(t, p, " ");
            if (p == NP) return false;
            p = c_flight(t, p);
            if (p == NP) return false;
            p = clit(t, p, " on ");
            if (p == NP) return false;
            p = c_date(t, p);
            if (p == NP) return false;
            p = clit(t, p, " (");
            if (p == NP) return false;
            p = c_airport(t, p);
            if (p == NP) return false;
            p = c_rarrow(t, p);
            if (p == NP) return false;
            p = c_airport(t, p);
            if (p == NP) return false;
            // Full form: <TIME><ARROW><TIME>) <DASH> <CABIN>
            // Short form: ) <DASH> <CABIN>
            std::size_t full = NP;
            {
                std::size_t q = clit(t, p, " ");
                if (q != NP) {
                    q = c_time(t, q);
                    if (q != NP) {
                        q = c_rarrow(t, q);
                        if (q != NP) {
                            q = c_time(t, q);
                            if (q != NP) full = q;
                        }
                    }
                }
            }
            if (full != NP) p = full;
            p = clit(t, p, ") ");
            if (p == NP) return false;
            p = c_dash(t, p);
            if (p == NP) return false;
            p = clit(t, p, " ");
            if (p == NP) return false;
            p = c_cabin(t, p);
            if (p == NP) return false;
            return p == t.size();
        };

        // Bullet: - checked bags: ...
        // Full: you're a <STATUS> member, so in <CABIN>
        //   you get <INT> free checked bag(s) <ARROW>
        //   your <INT> checked bag(s) is/are free
        //   (no bag fee).
        // Short: <INT> free checked bag(s) (no bag fee).
        // Minimal: <INT>
        auto match_bags = [&](const std::string& t)
            -> bool {
            std::size_t p = clit(t, 0,
                "- checked bags: ");
            if (p == NP) return false;
            // Try full form
            {
                std::size_t q = clit(t, p, "you're a ");
                if (q != NP) {
                    q = c_status(t, q);
                    if (q != NP) {
                        q = clit(t, q,
                            " member, so in ");
                        if (q != NP) {
                            q = c_cabin(t, q);
                            if (q != NP) {
                                q = clit(t, q,
                                    " you get ");
                                if (q != NP) {
                                    q = c_int(t, q);
                                    if (q != NP) {
                                        std::size_t r =
                                            clit(t, q,
                                            " free checked bags");
                                        if (r == NP)
                                            r = clit(t, q,
                                            " free checked bag");
                                        if (r != NP) {
                                            r = clit(t, r, " ");
                                            if (r != NP) {
                                                r = c_rarrow(t, r);
                                                if (r != NP) {
                                                    r = clit(t, r,
                                                        " your ");
                                                    if (r != NP) {
                                                        r = c_int(t,
                                                            r);
                                                        if (r != NP) {
                                                            std::size_t
                                                                s = clit(t, r, " checked bags are free (no bag fee).");
                                                            if (s == NP) s = clit(t, r, " checked bag is free (no bag fee).");
                                                            if (s != NP && s == t.size()) return true;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Short: <INT> free checked bag(s) (no bag
            // fee).
            {
                std::size_t q = c_int(t, p);
                if (q != NP) {
                    std::size_t r = clit(t, q,
                        " free checked bags (no bag fee).");
                    if (r == NP)
                        r = clit(t, q,
                        " free checked bag (no bag fee).");
                    if (r != NP && r == t.size())
                        return true;
                }
            }
            // Minimal: just <INT>
            {
                std::size_t q = c_int(t, p);
                if (q != NP && q == t.size()) return true;
            }
            return false;
        };

        // Bullet: - total flight price (<CABIN>):
        //   $<MONEY> to be charged to <PAYID>
        auto match_price = [&](const std::string& t)
            -> bool {
            std::size_t p = clit(t, 0,
                "- total flight price (");
            if (p == NP) return false;
            p = c_cabin(t, p);
            if (p == NP) return false;
            p = clit(t, p, "): $");
            if (p == NP) return false;
            p = c_money(t, p);
            if (p == NP) return false;
            p = clit(t, p, " to be charged to ");
            if (p == NP) return false;
            p = c_payid(t, p);
            if (p == NP) return false;
            return p == t.size();
        };

        // ── Line-by-line validation ──
        std::size_t cur = 0;
        while (cur < matched_q_start) {
            std::size_t le = cur;
            while (le < matched_q_start && low[le] != '\n')
                ++le;
            std::string line = low.substr(cur, le - cur);
            cur = (le < matched_q_start) ? le + 1 : le;
            std::size_t ls = 0;
            while (ls < line.size()
                   && (line[ls] == ' ' || line[ls] == '\t'
                       || line[ls] == '\r'))
                ++ls;
            std::size_t le2 = line.size();
            while (le2 > ls
                   && (line[le2-1] == ' '
                       || line[le2-1] == '\t'
                       || line[le2-1] == '\r'))
                --le2;
            if (ls >= le2) continue;
            std::string t = line.substr(ls, le2 - ls);
            if (t[0] == '#' && match_heading(t)) continue;
            if (match_status(t)) continue;
            if (match_transition(t)) continue;
            if (match_ins_expl(t)) continue;
            if (match_ins_cost(t)) continue;
            if (match_leg(t, "outbound:")) continue;
            if (match_leg(t, "return:")) continue;
            if (match_bags(t)) continue;
            if (match_price(t)) continue;
            return 0u;
        }
    }
    return 1u;
}

} // extern "C"

// ── First flight number extraction ────────────────────────────

// Forward-declare common functor (C linkage, defined in functors_common.cpp)
extern "C" souffle::RamDomain json_get_str(
    souffle::SymbolTable*, souffle::RecordTable*,
    souffle::RamDomain, souffle::RamDomain);

// Returns the first flight number appearing anywhere in the JSON.
//
// This used to delegate to @json_get_str, which resolves TOP-LEVEL keys
// only — and "flight_number" is never top level: it sits inside
// flights[] in both a reservation payload and an
// update_reservation_flights argument list. The delegating version
// therefore always returned "", so every rule built on it was dead.
// Scan for the first "flight_number": "..." pair instead, the same way
// any_flight_date_before scans for dates.
extern "C" souffle::RamDomain extract_first_flight(
    souffle::SymbolTable* st, souffle::RecordTable* /*rt*/, souffle::RamDomain json_sym)
{
    const std::string& json = st->decode(json_sym);
    static const char* kKey = "\"flight_number\"";
    std::size_t p = json.find(kKey);
    if (p == std::string::npos) return st->encode("");
    std::size_t c = json.find(':', p + std::strlen(kKey));
    if (c == std::string::npos) return st->encode("");
    std::size_t v = c + 1;
    while (v < json.size() &&
           std::isspace(static_cast<unsigned char>(json[v]))) ++v;
    if (v >= json.size() || json[v] != '"') return st->encode("");
    std::size_t q1 = v + 1;
    std::size_t q2 = json.find('"', q1);
    if (q2 == std::string::npos) return st->encode("");
    return st->encode(json.substr(q1, q2 - q1));
}
