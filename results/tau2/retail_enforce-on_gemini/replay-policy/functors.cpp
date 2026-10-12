/*
 * Custom C++ functors for the tau2 retail-agent policy.
 *
 * Compiled alongside the Souffle policy by the SASY evaluator.
 * Each functor performs a deterministic structural check; no LLM
 * oracle is invoked.
 *
 * Functors implemented:
 *   user_confirmed_check(text)  → 1 if the text contains affirmative
 *                                  confirmation language ("yes",
 *                                  "confirm", "go ahead", ...).
 *   is_blocked_marker(text)     → 1 if the text begins with the
 *                                  "[BLOCKED]" denial marker.
 *   is_handoff_message(text)    → 1 if the text contains the literal
 *                                  τ2 handoff sentence
 *                                  ("YOU ARE BEING TRANSFERRED ...").
 *   json_path_cents(json, path) → the JSON number at `path` in whole
 *                                  cents (rounded), -1 on a miss.
 *   json_path_is_true(json, path) → 1 if the value at `path` is the
 *                                  JSON literal true.
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <souffle/SouffleInterface.h>

namespace {

std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return out;
}

// Returns true iff `needle` appears in `hay` with non-letter (or
// boundary) characters on each side, and NOT preceded by a negation.
//
// Both halves matter. The boundary check is why "yes" does not match
// "eyes" or "yesterday". The negation window is why "No, that's not
// correct" is not agreement: the word "correct" is present either way,
// and reading a rejection as a confirmation settles the very
// cancellation reason the user just refused. Only the short run of text
// before the token is inspected, so "no problem — yes, go ahead" still
// reads as agreement.
//
// `whole_word` is false for multi-word phrases, which are long enough
// that a plain substring match does not over-match — but which still
// need the negation check ("don't go ahead").
bool affirmative_match(const std::string& hay, const char* needle,
                       bool whole_word) {
    const std::size_t n = std::strlen(needle);
    if (n == 0) return false;
    // Bare "no" is deliberately absent: see the "no problem" case above.
    static const char* const kNegations[] = {
        "not ", "n't ", "never ", "cannot ", "cant ",
        "rather not ", "instead of ", "no longer ",
    };
    std::size_t pos = 0;
    while ((pos = hay.find(needle, pos)) != std::string::npos) {
        bool boundary_ok = true;
        if (whole_word) {
            const bool left_ok =
                (pos == 0) || !std::isalpha(static_cast<unsigned char>(hay[pos - 1]));
            const std::size_t after = pos + n;
            const bool right_ok =
                (after >= hay.size()) ||
                !std::isalpha(static_cast<unsigned char>(hay[after]));
            boundary_ok = left_ok && right_ok;
        }
        if (boundary_ok) {
            const std::size_t win_start = pos > 24 ? pos - 24 : 0;
            std::string before = hay.substr(win_start, pos - win_start);
            // A negation only reaches across its own sentence: in
            // "No longer needed. Yes, cancel it." nothing before the
            // '.' can negate the "yes". Cut the window at the last
            // sentence boundary.
            const std::size_t cut = before.find_last_of(".!?;\n");
            if (cut != std::string::npos) before.erase(0, cut + 1);
            // "no longer needed" is one of the two cancellation reasons
            // the policy accepts. Stating it is not a refusal, so it
            // must not negate a following "yes" ("no longer needed,
            // yes"). Other "no longer ..." phrases still negate.
            static const std::string kReason = "no longer needed";
            for (std::size_t q; (q = before.find(kReason)) != std::string::npos;) {
                before.erase(q, kReason.size());
            }
            bool negated = false;
            for (const char* neg : kNegations) {
                if (before.find(neg) != std::string::npos) {
                    negated = true;
                    break;
                }
            }
            if (!negated) return true;
        }
        // Advance by one, not by `n`: overlapping occurrences matter
        // once a hit can be rejected for being negated.
        pos += 1;
    }
    return false;
}

// ── Minimal JSON path reader (for the numeric / boolean functors) ──
// Same path syntax as @json_get_str_path in functors_common.cpp:
// dotted object keys plus [n] array indices, e.g.
// "payment_methods.gift_card_1.balance" or "items[2].price". Those
// helpers are file-static there, so this file carries its own copy.

const char* skip_ws(const char* p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    return p;
}

const char* skip_value(const char* p) {
    p = skip_ws(p);
    if (*p == '"') {
        ++p;
        while (*p && *p != '"') {
            if (*p == '\\') ++p;
            if (*p) ++p;
        }
        if (*p == '"') ++p;
        return p;
    }
    if (*p == '{' || *p == '[') {
        const char open = *p;
        const char close = open == '{' ? '}' : ']';
        ++p;
        int depth = 1;
        while (*p && depth > 0) {
            if (*p == open) ++depth;
            else if (*p == close) --depth;
            else if (*p == '"') { p = skip_value(p); continue; }
            ++p;
        }
        return p;
    }
    while (*p && *p != ',' && *p != '}' && *p != ']' && *p != ' ' && *p != '\n') ++p;
    return p;
}

std::string read_key(const char* p) {
    std::string out;
    if (*p != '"') return out;
    ++p;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) ++p;
        out += *p++;
    }
    return out;
}

const char* find_field(const char* p, const std::string& field) {
    p = skip_ws(p);
    if (*p != '{') return nullptr;
    ++p;
    while (*p) {
        p = skip_ws(p);
        if (*p != '"') return nullptr;
        const std::string key = read_key(p);
        p = skip_ws(skip_value(p));
        if (*p != ':') return nullptr;
        p = skip_ws(p + 1);
        if (key == field) return p;
        p = skip_ws(skip_value(p));
        if (*p == ',') ++p;
    }
    return nullptr;
}

const char* find_elem(const char* p, std::size_t idx) {
    p = skip_ws(p);
    if (*p != '[') return nullptr;
    ++p;
    for (std::size_t k = 0; k < idx; ++k) {
        p = skip_ws(p);
        if (*p == ']' || *p == 0) return nullptr;
        p = skip_ws(skip_value(p));
        if (*p == ',') ++p;
    }
    p = skip_ws(p);
    if (*p == ']' || *p == 0) return nullptr;
    return p;
}

const char* find_path(const char* p, const std::string& path) {
    std::size_t i = 0;
    while (i < path.size() && p) {
        if (path[i] == '.') { ++i; continue; }
        if (path[i] == '[') {
            const std::size_t end = path.find(']', i);
            if (end == std::string::npos || end == i + 1) return nullptr;
            std::size_t idx = 0;
            for (std::size_t k = i + 1; k < end; ++k) {
                if (!std::isdigit(static_cast<unsigned char>(path[k]))) return nullptr;
                idx = idx * 10 + static_cast<std::size_t>(path[k] - '0');
            }
            p = find_elem(p, idx);
            i = end + 1;
        } else {
            std::size_t end = i;
            while (end < path.size() && path[end] != '.' && path[end] != '[') ++end;
            p = find_field(p, path.substr(i, end - i));
            i = end;
        }
    }
    return p ? skip_ws(p) : nullptr;
}

// ── Validated dollar-amount parser ───────────────────────────────
// Consumes $D+[.DD] at position p. Integer dollars (≥1 digit,
// ≤7 digits), optional single decimal point with EXACTLY two
// fractional digits. Leading markdown bold after '$' is skipped.
// Rejects malformed: $17.86.50, $17..86, $.86, $17.8, overflow.
bool consume_validated_amount(const std::string& s, std::size_t& p,
                              int& cents) {
    if (p >= s.size() || s[p] != '$') return false;
    std::size_t q = p + 1;
    while (q < s.size() && s[q] == '*') ++q;
    std::size_t int_start = q;
    while (q < s.size() && std::isdigit(static_cast<unsigned char>(s[q])))
        ++q;
    if (q == int_start) return false;
    if (q - int_start > 7) return false;
    long dollars = 0;
    for (std::size_t i = int_start; i < q; ++i)
        dollars = dollars * 10 + (s[i] - '0');
    int frac = 0;
    if (q < s.size() && s[q] == '.') {
        ++q;
        if (q + 2 > s.size()
            || !std::isdigit(static_cast<unsigned char>(s[q]))
            || !std::isdigit(static_cast<unsigned char>(s[q + 1])))
            return false;
        frac = (s[q] - '0') * 10 + (s[q + 1] - '0');
        q += 2;
        if (q < s.size()
            && (std::isdigit(static_cast<unsigned char>(s[q]))
                || s[q] == '.'))
            return false;
    }
    cents = static_cast<int>(dollars * 100 + frac);
    p = q;
    return true;
}

// ── Shared payment-selection parser ─────────────────────────────
// Parses fullmatch payment-selection clauses from lowercased text,
// returning coherent (card, cents) pairs. Supports proposal shapes
// (F1 question, F2a-c declarative) and user shapes (U-series).
// Used by proposal_payment_card_ending, proposal_payment_question_cents,
// extract_card_ending_from_text, and user_card_selection_ambiguous.

struct PaymentSelectionMatch {
    std::string card;
    int cents;
};

bool ps_consume(const std::string& s, std::size_t& p, const char* lit) {
    std::size_t n = std::strlen(lit);
    if (p + n > s.size()) return false;
    if (s.compare(p, n, lit) != 0) return false;
    p += n;
    return true;
}

bool ps_consume_4digits(const std::string& s, std::size_t& p,
                        std::string& out) {
    std::size_t q = p;
    while (q < s.size() && s[q] == '*') ++q;
    std::string digits;
    while (q < s.size()
           && std::isdigit(static_cast<unsigned char>(s[q])))
        digits += s[q++];
    if (digits.size() != 4) return false;
    while (q < s.size() && s[q] == '*') ++q;
    out = digits;
    p = q;
    return true;
}

bool ps_is_consumed(const std::string& s, std::size_t p) {
    while (p < s.size()
           && (s[p] == ' ' || s[p] == '*' || s[p] == '\t'))
        ++p;
    return p == s.size();
}

bool try_card_amount_shape(const std::string& clause, char terminal,
                           std::string& card, int& cents) {
    // F1: "if yes, would you like to use your original credit card
    //      ending in NNNN to cover the $X.XX price difference?"
    {
        std::size_t p = 0;
        std::string e; int c = -1;
        if (terminal == '?'
            && ps_consume(clause, p, "if yes, would you like to use your ")
            && ps_consume(clause, p, "original credit card ending in ")
            && ps_consume_4digits(clause, p, e)
            && ps_consume(clause, p, " to cover the ")
            && consume_validated_amount(clause, p, c)
            && ps_consume(clause, p, " price difference")
            && ps_is_consumed(clause, p)) {
            card = e; cents = c; return true;
        }
    }
    // F2a: "we will use your original credit card ending in NNNN
    //       to cover the $X.XX price difference."
    {
        std::size_t p = 0;
        std::string e; int c = -1;
        if (terminal == '.'
            && ps_consume(clause, p, "we will use your original credit card ending in ")
            && ps_consume_4digits(clause, p, e)
            && ps_consume(clause, p, " to cover the ")
            && consume_validated_amount(clause, p, c)
            && ps_consume(clause, p, " price difference")
            && ps_is_consumed(clause, p)) {
            card = e; cents = c; return true;
        }
    }
    // F2b: "we will use the card ending in NNNN to cover the $X.XX
    //       price difference."
    {
        std::size_t p = 0;
        std::string e; int c = -1;
        if (terminal == '.'
            && ps_consume(clause, p, "we will use the card ending in ")
            && ps_consume_4digits(clause, p, e)
            && ps_consume(clause, p, " to cover the ")
            && consume_validated_amount(clause, p, c)
            && ps_consume(clause, p, " price difference")
            && ps_is_consumed(clause, p)) {
            card = e; cents = c; return true;
        }
    }
    // F2c / U-core: "[prefix] use the card ending in NNNN
    //   {to cover the|for the} $X.XX {price difference|difference}"
    if (terminal == '.') {
        static const char* kPrefixes[] = {
            "", "and yes, ", "yes, ", nullptr
        };
        for (const char** pfx = kPrefixes; *pfx; ++pfx) {
            static const char* kMids[] = {
                " to cover the ", " for the ", nullptr
            };
            static const char* kTails[] = {
                " price difference", " difference", nullptr
            };
            for (const char** mid = kMids; *mid; ++mid) {
                for (const char** tail = kTails; *tail; ++tail) {
                    std::size_t p = 0;
                    std::string e; int c = -1;
                    if (ps_consume(clause, p, *pfx)
                        && ps_consume(clause, p, "use the card ending in ")
                        && ps_consume_4digits(clause, p, e)
                        && ps_consume(clause, p, *mid)
                        && consume_validated_amount(clause, p, c)
                        && ps_consume(clause, p, *tail)
                        && ps_is_consumed(clause, p)) {
                        card = e; cents = c; return true;
                    }
                }
            }
        }
    }
    return false;
}

std::vector<PaymentSelectionMatch> parse_payment_selections(
    const std::string& text)
{
    std::vector<PaymentSelectionMatch> matches;
    std::size_t clause_pos = 0;
    while (clause_pos < text.size()) {
        std::size_t clause_end = clause_pos;
        while (clause_end < text.size()) {
            char c = text[clause_end];
            if (c == '!' || c == '?' || c == '\n') break;
            if (c == '.') {
                bool in_number =
                    (clause_end > 0 && std::isdigit(static_cast<unsigned char>(
                         text[clause_end - 1])))
                    || (clause_end + 1 < text.size()
                        && std::isdigit(static_cast<unsigned char>(
                               text[clause_end + 1])));
                if (!in_number) break;
            }
            ++clause_end;
        }
        char terminal = (clause_end < text.size()) ? text[clause_end] : 0;
        std::string clause = text.substr(clause_pos,
            clause_end - clause_pos);
        clause_pos = clause_end + 1;
        std::size_t start = 0;
        while (start < clause.size()
               && (clause[start] == ' ' || clause[start] == '\t'
                   || clause[start] == '*'))
            ++start;
        clause = clause.substr(start);
        if (clause.empty()) continue;

        std::string card;
        int cents = -1;
        if (try_card_amount_shape(clause, terminal, card, cents)) {
            matches.push_back({card, cents});
        }
    }
    return matches;
}

struct PaymentSelectionSummary {
    std::string card;
    int cents;
    bool ambiguous;
};

PaymentSelectionSummary summarize_selections(
    const std::vector<PaymentSelectionMatch>& m)
{
    PaymentSelectionSummary s{"", -1, false};
    if (m.empty()) return s;
    s.card = m[0].card;
    s.cents = m[0].cents;
    for (std::size_t i = 1; i < m.size(); ++i) {
        if (m[i].card != s.card || m[i].cents != s.cents) {
            s.ambiguous = true;
            s.card = "";
            s.cents = -1;
            return s;
        }
    }
    return s;
}

}  // namespace

extern "C" {

souffle::RamDomain user_confirmed_check(souffle::SymbolTable* st,
                                        souffle::RecordTable*,
                                        souffle::RamDomain text_sym) {
    const std::string lower = to_lower(st->decode(text_sym));

    // Word-boundary phrases — short tokens that would over-match as
    // substrings of unrelated words.
    static const char* word_phrases[] = {
        "yes", "yep", "yeah", "yup",
        "ok", "okay",
        "sure", "confirm", "confirmed",
        "approve", "approved",
        "correct",
        nullptr,
    };
    for (const char** p = word_phrases; *p; ++p) {
        if (affirmative_match(lower, *p, /*whole_word=*/true)) return 1;
    }

    // Multi-word phrases — substring match is fine because they are
    // long enough not to false-match.
    static const char* multi_phrases[] = {
        "go ahead",
        "please do",
        "please proceed",
        "sounds good",
        "looks good",
        "do it",
        "let's do it",
        "i confirm",
        "that's correct",
        nullptr,
    };
    for (const char** p = multi_phrases; *p; ++p) {
        if (affirmative_match(lower, *p, /*whole_word=*/false)) return 1;
    }

    return 0;
}

souffle::RamDomain is_blocked_marker(souffle::SymbolTable* st,
                                     souffle::RecordTable*,
                                     souffle::RamDomain text_sym) {
    const std::string s = st->decode(text_sym);
    return (s.rfind("[BLOCKED]", 0) == 0) ? 1 : 0;
}

souffle::RamDomain is_handoff_message(souffle::SymbolTable* st,
                                      souffle::RecordTable*,
                                      souffle::RamDomain text_sym) {
    const std::string lower = to_lower(st->decode(text_sym));
    static const char* handoff =
        "you are being transferred to a human agent";
    return (lower.find(handoff) != std::string::npos) ? 1 : 0;
}

// The JSON number at `path`, in whole cents (rounded to nearest).
// -1 on any miss, a non-number, or a negative value.
souffle::RamDomain json_path_cents(souffle::SymbolTable* st,
                                   souffle::RecordTable*,
                                   souffle::RamDomain json_sym,
                                   souffle::RamDomain path_sym) {
    const std::string json = st->decode(json_sym);
    const std::string path = st->decode(path_sym);
    const char* p = find_path(json.c_str(), path);
    if (!p) return -1;
    if (*p != '-' && !std::isdigit(static_cast<unsigned char>(*p))) return -1;
    char* end = nullptr;
    const double v = std::strtod(p, &end);
    if (end == p || !std::isfinite(v) || v < 0 || v > 20000000.0) return -1;
    return static_cast<souffle::RamDomain>(std::llround(v * 100.0));
}

// 1 iff the value at `path` is the JSON literal `true`.
souffle::RamDomain json_path_is_true(souffle::SymbolTable* st,
                                     souffle::RecordTable*,
                                     souffle::RamDomain json_sym,
                                     souffle::RamDomain path_sym) {
    const std::string json = st->decode(json_sym);
    const std::string path = st->decode(path_sym);
    const char* p = find_path(json.c_str(), path);
    if (!p) return 0;
    if (std::strncmp(p, "true", 4) != 0) return 0;
    const char after = p[4];
    return (after == 0 || after == ',' || after == '}' || after == ']' ||
            after == ' ' || after == '\n' || after == '\r' || after == '\t')
               ? 1
               : 0;
}

// 1 iff `text` positively selects `id` as a payment method.
//
// Full-message anchored grammar. The ENTIRE message must be one
// recognized authorization sentence (or one authorization plus
// harmless courtesies like "Thank you."). Multi-sentence messages
// with unrecognized fragments are rejected.
//
// Pre-processing: reject whole-message questions (trailing '?'),
// strip Markdown emphasis, normalize em/en-dash, reject any
// explicit negation of the id anywhere in the message.
//
// Single-sentence grammar (case-insensitive, trimmed):
//   [repeatable-prefix]* <verb> <id> [exact-tail]
//
// Compound: [prefix]* "exchanging both items on order #<ID>"
//   " and " <verb> <id> [tail]
// where the text before "and" matches that exact pattern with no
// extra words (observed in actual GPT task58 trial2 authorization).
//
// Conservative false negatives are acceptable.
souffle::RamDomain user_selected_payment_id(souffle::SymbolTable* st,
                                            souffle::RecordTable*,
                                            souffle::RamDomain text_sym,
                                            souffle::RamDomain id_sym) {
    std::string text = to_lower(st->decode(text_sym));
    const std::string id_raw = to_lower(st->decode(id_sym));
    if (id_raw.empty()) return 0;

    auto is_id_char = [](unsigned char c) -> bool {
        return std::isalnum(c) || c == '_';
    };

    // Whole-message question rejection (before any processing).
    {
        std::size_t last_nonws = text.find_last_not_of(" \t\n\r");
        if (last_nonws != std::string::npos && text[last_nonws] == '?')
            return 0;
    }

    // Strip Markdown emphasis: remove all occurrences of "**"
    {
        std::string clean;
        clean.reserve(text.size());
        for (std::size_t i = 0; i < text.size(); ++i) {
            if (i + 1 < text.size() && text[i] == '*' && text[i+1] == '*') {
                ++i;
            } else {
                clean += text[i];
            }
        }
        text = clean;
    }

    // Normalize em-dash (UTF-8: 0xE2 0x80 0x93/0x94) and ASCII
    // double-dash to ", " so they act as clause separators.
    {
        std::string norm;
        norm.reserve(text.size());
        for (std::size_t i = 0; i < text.size(); ++i) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            if (c == 0xe2 && i + 2 < text.size() &&
                static_cast<unsigned char>(text[i+1]) == 0x80 &&
                (static_cast<unsigned char>(text[i+2]) == 0x93 ||
                 static_cast<unsigned char>(text[i+2]) == 0x94)) {
                norm += ", ";
                i += 2;
            } else if (c == '-' && i + 1 < text.size() && text[i+1] == '-') {
                norm += ", ";
                ++i;
            } else {
                norm += text[i];
            }
        }
        text = norm;
    }

    // Check for explicit negation of this id ANYWHERE in the message.
    // "not use <id>", "n't use <id>", "do not use <id>"
    {
        static const char* const kNegVerbs[] = {
            "not use ", "n't use ", "not charge ", "n't charge ",
            nullptr,
        };
        for (const char* const* nv = kNegVerbs; *nv; ++nv) {
            std::size_t np = text.find(*nv);
            while (np != std::string::npos) {
                std::size_t after_nv = np + std::strlen(*nv);
                if (text.compare(after_nv, id_raw.size(), id_raw) == 0) {
                    std::size_t id_end = after_nv + id_raw.size();
                    if (id_end >= text.size() ||
                        !is_id_char(static_cast<unsigned char>(text[id_end])))
                        return 0;
                }
                np = text.find(*nv, np + 1);
            }
        }
    }

    // Split into sentences at . ! ; \n
    // A '.' preceded by a digit is a decimal point, not a sentence end.
    std::vector<std::string> sentences;
    {
        std::string cur;
        for (std::size_t i = 0; i < text.size(); ++i) {
            const char ch = text[i];
            bool is_sent_end = (ch == '!' || ch == ';' || ch == '\n');
            if (ch == '.' && (i == 0 ||
                !std::isdigit(static_cast<unsigned char>(text[i-1])))) {
                is_sent_end = true;
            }
            if (is_sent_end) {
                if (!cur.empty()) sentences.push_back(cur);
                cur.clear();
            } else {
                cur += ch;
            }
        }
        if (!cur.empty()) sentences.push_back(cur);
    }

    // Selection verbs (longer first for greedy match)
    static const char* const kVerbs[] = {
        "proceed using ", "pay with ", "pay using ",
        "charge to ", "charge ",
        "use ", "switch to ", "change to ",
        "apply ", "select ",
        nullptr,
    };

    // Safe tail: the ENTIRE tail after the id must match a recognized
    // pattern. No prefix-only matching — the whole suffix is consumed.
    //
    // Recognized tail patterns (after trimming punctuation/whitespace):
    //   - empty
    //   - exact: "please", "thanks", "thank you", "thx"
    //   - "as the <noun-phrase>" where noun-phrase is one of a closed
    //     set: "payment method", "single payment method", "payment",
    //     "refund method"
    //   - "for the <$amount-phrase>" where amount-phrase matches
    //     "$NNN.NN difference" / "$NNN.NN price difference" /
    //     "$NNN.NN exchange"
    auto tail_is_safe = [](const std::string& tail) -> bool {
        if (tail.empty()) return true;
        static const char* const kExact[] = {
            "please", "thanks", "thank you", "thx",
            nullptr,
        };
        for (const char* const* e = kExact; *e; ++e) {
            if (tail == *e) return true;
        }
        // "as the <recognized-noun>" — must match the whole tail
        static const char* const kAsThe[] = {
            "as the payment method",
            "as the single payment method",
            "as the payment",
            "as the refund method",
            "as my payment method",
            nullptr,
        };
        for (const char* const* a = kAsThe; *a; ++a) {
            if (tail == *a) return true;
        }
        // Anchored amount-difference pattern:
        //   "for the $NNN.NN difference" / "for the $NNN.NN price difference"
        auto match_amount_diff = [](const std::string& s) -> bool {
            if (s.compare(0, 8, "for the ") != 0) return false;
            std::size_t p = 8;
            if (p >= s.size() || s[p] != '$') return false;
            ++p;
            while (p < s.size() &&
                   (std::isdigit(static_cast<unsigned char>(s[p])) ||
                    s[p] == '.' || s[p] == ','))
                ++p;
            if (p >= s.size() || s[p] != ' ') return false;
            std::string rest = s.substr(p + 1);
            return (rest == "difference" || rest == "price difference" ||
                    rest == "exchange" || rest == "exchange difference");
        };
        if (match_amount_diff(tail)) return true;
        // Compound: "as the <noun> for the $NNN.NN difference"
        // e.g. "as the single payment method for the $42.55 difference"
        for (const char* const* a = kAsThe; *a; ++a) {
            const std::size_t alen = std::strlen(*a);
            if (tail.compare(0, alen, *a) == 0) {
                // Check if the rest after the noun phrase is
                // " for the $NNN.NN difference"
                std::string after_noun = tail.substr(alen);
                // trim leading space
                if (!after_noun.empty() && after_noun[0] == ' ')
                    after_noun = after_noun.substr(1);
                if (match_amount_diff(after_noun)) return true;
            }
        }
        return false;
    };

    // Helper: try to match <verb> <id> [safe-tail] at start of `s`.
    auto try_verb_match = [&](const std::string& s) -> bool {
        std::string work = s;
        if (work.compare(0, 7, "charge ") == 0) {
            std::size_t p = 7;
            if (p < work.size() && work[p] == '$') {
                ++p;
                while (p < work.size() &&
                       (std::isdigit(static_cast<unsigned char>(work[p])) ||
                        work[p] == '.' || work[p] == ','))
                    ++p;
                while (p < work.size() && work[p] == ' ') ++p;
                if (p < work.size() && work.compare(p, 3, "to ") == 0) {
                    work = "charge to " + work.substr(p + 3);
                }
            }
        }
        for (const char* const* v = kVerbs; *v; ++v) {
            const std::size_t vlen = std::strlen(*v);
            if (work.size() < vlen) continue;
            if (work.compare(0, vlen, *v) != 0) continue;
            const std::string after_verb = work.substr(vlen);
            if (after_verb.size() < id_raw.size()) continue;
            if (after_verb.compare(0, id_raw.size(), id_raw) != 0)
                continue;
            const std::size_t id_end = id_raw.size();
            if (id_end < after_verb.size() &&
                is_id_char(static_cast<unsigned char>(
                    after_verb[id_end])))
                continue;
            std::string tail;
            if (id_end < after_verb.size())
                tail = after_verb.substr(id_end);
            std::size_t t0 = 0;
            while (t0 < tail.size() &&
                   (tail[t0] == ' ' || tail[t0] == '\t' ||
                    tail[t0] == ',' || tail[t0] == '.'))
                ++t0;
            std::string tail_clean;
            if (t0 < tail.size()) tail_clean = tail.substr(t0);
            while (!tail_clean.empty()) {
                const char c = tail_clean.back();
                if (c == ' ' || c == '\t' || c == '.' ||
                    c == ',' || c == '!')
                    tail_clean.pop_back();
                else break;
            }
            if (tail_is_safe(tail_clean)) return true;
        }
        return false;
    };

    // Strip repeatable safe prefixes from `s`, trying verb match at
    // each stage. Returns true if a verb+id match is found.
    static const char* const kPrefixWords[] = {
        "go ahead and", "go ahead",
        "i want to", "i'd like to", "i choose to",
        "i'll", "let's",
        "proceed with", "proceed to",
        "yes", "please", "sure", "ok", "okay",
        nullptr,
    };
    auto try_prefixed_verb = [&](std::string s) -> bool {
        if (try_verb_match(s)) return true;
        bool stripped = true;
        while (stripped) {
            stripped = false;
            for (const char* const* pw = kPrefixWords; *pw; ++pw) {
                const std::size_t wlen = std::strlen(*pw);
                if (s.size() >= wlen &&
                    s.compare(0, wlen, *pw) == 0) {
                    std::size_t k = wlen;
                    while (k < s.size() &&
                           (s[k] == ' ' || s[k] == ',' ||
                            s[k] == '\t' || s[k] == '-'))
                        ++k;
                    if (k > wlen || k == s.size()) {
                        s = s.substr(k);
                        stripped = true;
                        if (try_verb_match(s)) return true;
                        break;
                    }
                }
            }
        }
        // Compound: the only supported form is the exact pattern from
        // GPT task58 trial2:
        //   "exchanging both items on order #<ORDER_ID>" + " and " + verb
        // where ORDER_ID is one or more ASCII alphanumeric chars and
        // before_and matches that pattern EXACTLY (no extra words).
        static const char* const kCompoundPrefix =
            "exchanging both items on order #";
        const std::size_t cplen = std::strlen(kCompoundPrefix);
        const std::size_t and_pos = s.find(" and ");
        if (and_pos != std::string::npos && and_pos > cplen) {
            const std::string before = s.substr(0, and_pos);
            if (before.compare(0, cplen, kCompoundPrefix) == 0) {
                bool all_alnum = (before.size() > cplen);
                for (std::size_t j = cplen; j < before.size(); ++j) {
                    if (!std::isalnum(static_cast<unsigned char>(before[j]))) {
                        all_alnum = false;
                        break;
                    }
                }
                if (all_alnum) {
                    if (try_verb_match(s.substr(and_pos + 5))) return true;
                }
            }
        }
        return false;
    };

    // Full-message requirement: the message must be exactly one
    // recognized authorization sentence. Multi-sentence messages are
    // rejected — a later sentence could revoke, defer, or redirect.
    // Exception: sentences that are empty after trimming are ignored.
    if (sentences.size() > 1) {
        // Find which sentence (if any) contains the selection, and
        // verify every OTHER sentence is a harmless courtesy or empty.
        static const char* const kCourtesy[] = {
            "thank you", "thanks", "thx", "",
            nullptr,
        };
        int selection_idx = -1;
        for (std::size_t si = 0; si < sentences.size(); ++si) {
            std::string s = sentences[si];
            // trim
            std::size_t s0 = 0;
            while (s0 < s.size() && (s[s0]==' '||s[s0]=='\t')) ++s0;
            std::size_t s1 = s.size();
            while (s1 > s0 && (s[s1-1]==' '||s[s1-1]=='\t')) --s1;
            if (s0 >= s1) continue;
            s = s.substr(s0, s1 - s0);
            if (try_prefixed_verb(s)) {
                if (selection_idx >= 0) return 0; // two selections
                selection_idx = static_cast<int>(si);
            } else {
                // Must be a recognized courtesy
                bool ok = false;
                for (const char* const* c = kCourtesy; *c; ++c) {
                    if (s == *c) { ok = true; break; }
                }
                if (!ok) return 0; // unrecognized sentence
            }
        }
        return (selection_idx >= 0) ? 1 : 0;
    }

    // Single sentence
    if (sentences.size() == 1) {
        std::string s = sentences[0];
        std::size_t s0 = 0;
        while (s0 < s.size() && (s[s0]==' '||s[s0]=='\t')) ++s0;
        std::size_t s1 = s.size();
        while (s1 > s0 && (s[s1-1]==' '||s[s1-1]=='\t')) --s1;
        if (s0 < s1) {
            s = s.substr(s0, s1 - s0);
            if (try_prefixed_verb(s)) return 1;
        }
    }
    return 0;
}

// Returns the order ID (e.g. "#w5838674") named in a compound
// payment-selection message, or empty string if the selection is
// simple (no order context) or the message is not a valid selection.
// Used by the Datalog to scope a compound exact-ID exception to the
// named order only.
souffle::RamDomain user_selected_payment_order(souffle::SymbolTable* st,
                                               souffle::RecordTable*,
                                               souffle::RamDomain text_sym,
                                               souffle::RamDomain id_sym) {
    // Reuse user_selected_payment_id to check validity first.
    // If it returns 0, there is no valid selection at all.
    if (user_selected_payment_id(st, nullptr, text_sym, id_sym) == 0)
        return st->encode("");

    // Now extract the order from the compound pattern.
    std::string text = to_lower(st->decode(text_sym));
    // Strip Markdown **
    {
        std::string clean;
        clean.reserve(text.size());
        for (std::size_t i = 0; i < text.size(); ++i) {
            if (i + 1 < text.size() && text[i] == '*' && text[i+1] == '*') {
                ++i;
            } else {
                clean += text[i];
            }
        }
        text = clean;
    }
    // Normalize em-dash
    {
        std::string norm;
        norm.reserve(text.size());
        for (std::size_t i = 0; i < text.size(); ++i) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            if (c == 0xe2 && i + 2 < text.size() &&
                static_cast<unsigned char>(text[i+1]) == 0x80 &&
                (static_cast<unsigned char>(text[i+2]) == 0x93 ||
                 static_cast<unsigned char>(text[i+2]) == 0x94)) {
                norm += ", ";
                i += 2;
            } else if (c == '-' && i + 1 < text.size() && text[i+1] == '-') {
                norm += ", ";
                ++i;
            } else {
                norm += text[i];
            }
        }
        text = norm;
    }
    // Look for the compound prefix
    static const char* const kCP = "exchanging both items on order ";
    const std::size_t cplen = std::strlen(kCP);
    const std::size_t pos = text.find(kCP);
    if (pos == std::string::npos) return st->encode("");
    // Extract "#<alnum>" after the prefix
    std::size_t id_start = pos + cplen;
    if (id_start >= text.size() || text[id_start] != '#')
        return st->encode("");
    std::size_t id_end = id_start + 1;
    while (id_end < text.size() &&
           std::isalnum(static_cast<unsigned char>(text[id_end])))
        ++id_end;
    if (id_end == id_start + 1) return st->encode("");
    return st->encode(text.substr(id_start, id_end - id_start));
}

// Recognized whole-message unconditional approval for the preserved-
// consent path. A positive grammar of supported forms — anything not
// matching a recognized pattern denies by default. Recognized forms
// are multi-sentence: each sentence must be a recognized affirmative,
// a recognized payment selection, or silent punctuation / filler.
// Unknown sentences, conditions, questions, and arbitrary tails reject.
//
// Supported approval: the exact original positive shape and close
// structural variants (leading YES/confirm + optional payment-card
// selection sentence). Conservative; unsupported forms require fresh
// confirmation.
souffle::RamDomain user_confirmed_unconditional(
    souffle::SymbolTable* st, souffle::RecordTable* rt,
    souffle::RamDomain text_sym) {
    if (user_confirmed_check(st, rt, text_sym) == 0) return 0;
    const std::string raw = st->decode(text_sym);
    std::string text = to_lower(raw);

    // Normalize em-dash / double-dash to comma-space for splitting.
    {
        std::string norm;
        norm.reserve(text.size());
        for (std::size_t i = 0; i < text.size(); ++i) {
            unsigned char c = static_cast<unsigned char>(text[i]);
            if (c == 0xe2 && i + 2 < text.size() &&
                static_cast<unsigned char>(text[i+1]) == 0x80 &&
                (static_cast<unsigned char>(text[i+2]) == 0x93 ||
                 static_cast<unsigned char>(text[i+2]) == 0x94)) {
                norm += ", ";
                i += 2;
            } else if (c == '-' && i + 1 < text.size() && text[i+1] == '-') {
                norm += ", ";
                ++i;
            } else {
                norm += text[i];
            }
        }
        text = norm;
    }

    // Split into sentences by paragraph breaks and sentence-ending
    // punctuation. Skip "." inside numbers ($17.86).
    auto is_sentence_end = [&](std::size_t i) -> bool {
        char c = text[i];
        if (c == '!' || c == '?' || c == '\n') return true;
        if (c == '.') {
            if (i + 1 < text.size() &&
                std::isdigit(static_cast<unsigned char>(text[i + 1])))
                return false;
            if (i > 0 &&
                std::isdigit(static_cast<unsigned char>(text[i - 1])))
                return false;
            return true;
        }
        return false;
    };

    std::vector<std::string> sentences;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= text.size(); ++i) {
        bool at_end = (i == text.size());
        bool split = at_end || is_sentence_end(i);
        if (split) {
            std::string s = text.substr(start, i - start);
            while (!s.empty() && (s.front() == ' ' || s.front() == '\t'
                   || s.front() == '*' || s.front() == '\n'))
                s.erase(s.begin());
            while (!s.empty() && (s.back() == ' ' || s.back() == '\t'
                   || s.back() == '*'))
                s.pop_back();
            if (!s.empty()) sentences.push_back(s);
            start = i + 1;
        }
    }
    if (sentences.empty()) return 0;

    // Recognized affirmative sentence patterns (trimmed, lowercased).
    // Positive grammar: the sentence must be an affirmative word/phrase
    // optionally followed by a recognized safe tail. Unknown tails
    // (conditions, elaborations, arbitrary content) reject.
    auto is_affirmative = [](const std::string& s) -> bool {
        static const char* kAffirm[] = {
            "yes", "yep", "yeah", "please confirm",
            "please proceed", "go ahead", "confirmed",
            nullptr,
        };
        const char* matched = nullptr;
        std::size_t match_len = 0;
        for (const char** p = kAffirm; *p; ++p) {
            std::size_t n = std::strlen(*p);
            if (s.size() >= n && s.compare(0, n, *p) == 0) {
                if (s.size() == n || s[n] == ' ' || s[n] == ','
                    || s[n] == '*') {
                    matched = *p;
                    match_len = n;
                    break;
                }
            }
        }
        if (!matched) return false;
        if (s.size() == match_len) return true;
        // Extract the tail after the affirmative keyword.
        std::string tail = s.substr(match_len);
        // Strip leading punctuation/whitespace/markdown.
        while (!tail.empty() && (tail.front() == ' ' || tail.front() == ','
               || tail.front() == '*' || tail.front() == '\t'))
            tail.erase(tail.begin());
        if (tail.empty()) return true;
        // Recognized safe tails (complete match).
        static const char* kSafeTails[] = {
            "please confirm both changes",
            "please confirm",
            "please proceed",
            "proceed",
            "go ahead",
            "confirmed",
            "please",
            nullptr,
        };
        for (const char** p = kSafeTails; *p; ++p) {
            if (tail == *p) return true;
        }
        return false;
    };

    // Anchored payment-selection grammar: the ENTIRE sentence must
    // decompose into recognized segments with nothing left over.
    //   [prefix] verb filler "card" filler "ending in" bold? digits bold? [amount-tail]
    // prefix:      "and yes, " | ""
    // verb:        "use " | "charge " | "please use " | ...
    // filler:      "the " | "the original " | "my " | "your " | ...
    // amount-tail: "for the $N.NN difference" | "to cover the $N.NN" | "please" | ""
    // Anything else (conditions, questions, extra clauses) → reject.
    auto is_payment_selection = [](const std::string& s) -> bool {
        std::string r = s;
        // Strip prefix.
        for (const char* px : {"and yes, ", "and yes "}) {
            std::size_t n = std::strlen(px);
            if (r.size() >= n && r.compare(0, n, px) == 0)
                { r = r.substr(n); break; }
        }
        while (!r.empty() && r.front() == ' ') r.erase(r.begin());
        // Verb.
        static const char* kVerbs[] = {
            "please use ", "please charge ", "use ", "charge ",
            "charge it to ", "pay with ", nullptr,
        };
        bool verb_ok = false;
        for (const char** p = kVerbs; *p; ++p) {
            std::size_t n = std::strlen(*p);
            if (r.size() >= n && r.compare(0, n, *p) == 0)
                { r = r.substr(n); verb_ok = true; break; }
        }
        if (!verb_ok) return false;
        // Filler between verb and "card ending in": closed allowlist.
        std::size_t card_pos = r.find("card ending in ");
        std::size_t ce;
        if (card_pos != std::string::npos) {
            ce = card_pos + 15;
        } else {
            card_pos = r.find("card ends in ");
            if (card_pos == std::string::npos) return false;
            ce = card_pos + 13;
        }
        std::string filler = r.substr(0, card_pos);
        static const char* kFillerOk[] = {
            "", "the ", "my ", "your ",
            "the original ", "my original ", "your original ",
            "the credit ", "my credit ", "your credit ",
            nullptr,
        };
        bool filler_ok = false;
        for (const char** f = kFillerOk; *f; ++f) {
            if (filler == *f) { filler_ok = true; break; }
        }
        if (!filler_ok) return false;
        // Skip bold markers, read exactly 4+ digits.
        while (ce < r.size() && (r[ce] == '*' || r[ce] == ' ')) ++ce;
        std::string digits;
        while (ce < r.size() &&
               std::isdigit(static_cast<unsigned char>(r[ce])))
            digits += r[ce++];
        if (digits.size() != 4) return false;
        while (ce < r.size() && (r[ce] == '*' || r[ce] == ' ')) ++ce;
        // Tail must fully match a recognized amount phrase, or be
        // empty (including terminal punctuation the sentence splitter
        // kept because a digit preceded it — "1276.").
        std::string tail = (ce < r.size()) ? r.substr(ce) : "";
        while (!tail.empty() && (tail.back() == ' ' || tail.back() == '.'
               || tail.back() == '!')) tail.pop_back();
        if (tail.empty()) return true;
        // Structurally parse the tail.
        // Recognized: "for the $N.NN difference" / "to cover the $N.NN"
        // / "to cover the $N.NN price difference" / "please"
        // Amount uses validated parser: $D+[.DD], no malformed.
        auto try_amount_tail = [](const std::string& t,
                                   std::size_t p) -> std::size_t {
            int dummy_cents = 0;
            if (!consume_validated_amount(t, p, dummy_cents))
                return std::string::npos;
            while (p < t.size() && t[p] == '*') ++p;
            return p;
        };
        // "for the $X.XX difference"
        if (tail.substr(0, 8) == "for the ") {
            std::size_t ap = try_amount_tail(tail, 8);
            if (ap != std::string::npos) {
                std::string rest = tail.substr(ap);
                while (!rest.empty() && rest.front() == ' ')
                    rest.erase(rest.begin());
                if (rest.empty() || rest == "difference"
                    || rest == "price difference") return true;
            }
        }
        // "to cover the $X.XX [price difference]"
        if (tail.substr(0, 13) == "to cover the ") {
            std::size_t ap = try_amount_tail(tail, 13);
            if (ap != std::string::npos) {
                std::string rest = tail.substr(ap);
                while (!rest.empty() && rest.front() == ' ')
                    rest.erase(rest.begin());
                if (rest.empty() || rest == "difference"
                    || rest == "price difference") return true;
            }
        }
        if (tail == "please") return true;
        return false;
    };

    // Every sentence must be recognized. At most ONE payment-selection
    // sentence is allowed; multiple selections (even identical) are
    // ambiguous and rejected conservatively.
    bool has_affirm = false;
    int payment_count = 0;
    for (const auto& s : sentences) {
        if (is_affirmative(s)) { has_affirm = true; continue; }
        if (is_payment_selection(s)) { ++payment_count; continue; }
        // Unrecognized sentence → reject.
        return 0;
    }
    if (!has_affirm) return 0;
    if (payment_count > 1) return 0;

    // Global rejection checks on the full text.
    if (text.find('"') != std::string::npos) return 0;
    // Ends with question mark.
    std::size_t last = text.find_last_not_of(" \t\n\r");
    if (last != std::string::npos && text[last] == '?') return 0;

    return 1;
}

// 1 iff the proposal text contains an unconditional address-change
// section where the DESTINATION mentions all of the address fields.
// A "from [source] to [destination]" proposal only matches the fields
// in the destination, not the source address. Rejects proposals that
// only mention the order without proposing an address change, and
// conditional proposals ("if you'd like to change...").
souffle::RamDomain proposal_covers_address(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain proposal_sym,
    souffle::RamDomain a1_sym, souffle::RamDomain a2_sym,
    souffle::RamDomain city_sym, souffle::RamDomain state_sym,
    souffle::RamDomain zip_sym, souffle::RamDomain country_sym)
{
    std::string prop = to_lower(st->decode(proposal_sym));
    if (prop.find("address") == std::string::npos) return 0;
    static const char* kAddressMarkers[] = {
        "address change", "shipping address",
        "update the shipping", "update the address",
        "change the shipping", "change the address",
        nullptr,
    };
    bool has_addr_section = false;
    for (const char** m = kAddressMarkers; *m; ++m) {
        if (prop.find(*m) != std::string::npos) {
            has_addr_section = true; break;
        }
    }
    if (!has_addr_section) return 0;

    // Find the address SECTION heading — "address change" preceded
    // by a section marker like "**N." or at a paragraph start with
    // ":**" after it. Incidental mentions of "shipping address" in
    // prose paragraphs are NOT section headings.
    std::size_t addr_marker = std::string::npos;
    {
        std::size_t p = 0;
        while (true) {
            p = prop.find("address change", p);
            if (p == std::string::npos) break;
            // Check if this is in a heading context (preceded by
            // "**N. " or similar section marker, or followed by ":**")
            std::size_t after = p + 14;
            if (after + 2 <= prop.size() &&
                (prop.compare(after, 3, ":**") == 0 ||
                 prop.compare(after, 2, ":*") == 0 ||
                 prop[after] == ':')) {
                addr_marker = p; break;
            }
            p += 14;
        }
    }
    // Fallback: look for "update the shipping address" or
    // "change the shipping address" at a line start (the action verb
    // line itself serves as the address section marker).
    if (addr_marker == std::string::npos) {
        static const char* kLineVerbs[] = {
            "we will update the shipping address",
            "we'll update the shipping address",
            "we will change the shipping address",
            "update the shipping address",
            nullptr,
        };
        for (const char** lv = kLineVerbs; *lv; ++lv) {
            std::size_t p = prop.find(*lv);
            if (p == std::string::npos) continue;
            // Must be at line start (preceded by \n or start of text,
            // after optional whitespace).
            std::size_t ls = p;
            while (ls > 0 && prop[ls-1] == ' ') --ls;
            if (ls == 0 || prop[ls-1] == '\n') {
                addr_marker = p; break;
            }
        }
    }
    if (addr_marker == std::string::npos) return 0;

    // Find the affirmative action verb on the line after the
    // heading, or the heading IS the action line. The verb must
    // start the line (no conditional/illustrative prefix).
    std::size_t action_line = addr_marker;
    // If addr_marker is a heading ("address change:**"), skip to
    // the next line for the action sentence.
    {
        std::size_t colon = prop.find(':', addr_marker);
        if (colon != std::string::npos && colon < addr_marker + 25) {
            std::size_t nl = prop.find('\n', colon);
            if (nl != std::string::npos) {
                action_line = nl + 1;
                while (action_line < prop.size() &&
                       (prop[action_line] == ' ' || prop[action_line] == '*'))
                    ++action_line;
            }
        }
    }
    static const char* kAddrVerbs[] = {
        "we will update the shipping address",
        "we'll update the shipping address",
        "we will change the shipping address",
        "we'll change the shipping address",
        "we will update the address",
        "we'll update the address",
        "update the shipping address",
        nullptr,
    };
    std::size_t verb_end = std::string::npos;
    for (const char** v = kAddrVerbs; *v; ++v) {
        std::size_t vlen = std::strlen(*v);
        if (action_line + vlen <= prop.size() &&
            prop.compare(action_line, vlen, *v) == 0) {
            verb_end = action_line + vlen;
            break;
        }
    }
    if (verb_end == std::string::npos) return 0;

    std::string exp_a1  = to_lower(st->decode(a1_sym));
    std::string exp_a2  = to_lower(st->decode(a2_sym));
    std::string exp_ci  = to_lower(st->decode(city_sym));
    std::string exp_st  = to_lower(st->decode(state_sym));
    std::string exp_zip = st->decode(zip_sym);
    std::string exp_co  = to_lower(st->decode(country_sym));

    // ── Complete positive grammar for the action clause ──
    //
    // Three authorized action-line families. Everything between the
    // verb and the destination CSV must match one complete shape;
    // unknown words, conditions, or qualifiers reject structurally.
    //
    // Family A: " from your <src> address to your [default] <city> address:"
    //   followed by a canonical CSV address line after the colon.
    //   <src> is a simple location label (letters/spaces only).
    //   <city> must equal the expected destination city.
    //   Country may be omitted from the CSV iff "default" was present.
    //
    // Family B: " to:" or " to\n"
    //   followed by a canonical CSV address with country required.
    //
    // Family C: " to <inline CSV>."
    //   inline canonical CSV with country required.

    auto is_location_label = [](const std::string& s) -> bool {
        if (s.empty()) return false;
        for (char c : s) {
            if (!std::isalpha(static_cast<unsigned char>(c)) && c != ' ')
                return false;
        }
        return true;
    };

    auto find_dest_end = [&](std::size_t start) -> std::size_t {
        std::size_t end = start;
        while (end < prop.size()) {
            if (prop[end] == '\n') {
                std::size_t next = end + 1;
                while (next < prop.size() && prop[next] == ' ')
                    ++next;
                if (next >= prop.size() || prop[next] == '\n')
                    break;
                if (prop[next] == '#') break;
                if (prop[next] == '*' && next + 1 < prop.size()
                    && prop[next + 1] == '*') break;
                if (prop.compare(next, 4, "for ") == 0) break;
                if (prop.compare(next, 7, "please ") == 0) break;
            }
            ++end;
        }
        return end;
    };

    auto consume = [&](std::size_t p, const char* lit) -> std::size_t {
        std::size_t n = std::strlen(lit);
        if (p + n > prop.size()) return std::string::npos;
        if (prop.compare(p, n, lit) != 0) return std::string::npos;
        return p + n;
    };

    std::string search_area;
    bool is_default = false;

    // Family A: " from your <label> address to your [default] <city> address:"
    auto try_family_a = [&]() -> bool {
        std::size_t p = consume(verb_end, " from your ");
        if (p == std::string::npos) return false;
        std::size_t delim = prop.find(" address to your ", p);
        if (delim == std::string::npos || delim == p) return false;
        if (!is_location_label(prop.substr(p, delim - p))) return false;
        p = delim + 17;
        std::size_t dp = consume(p, "default ");
        if (dp != std::string::npos) { is_default = true; p = dp; }
        std::size_t colon = prop.find(" address:", p);
        if (colon == std::string::npos || colon == p) return false;
        std::string city_label = prop.substr(p, colon - p);
        while (!city_label.empty() && city_label.front() == ' ')
            city_label.erase(city_label.begin());
        while (!city_label.empty() && city_label.back() == ' ')
            city_label.pop_back();
        if (!is_location_label(city_label) || city_label != exp_ci)
            return false;
        p = colon + 9;
        while (p < prop.size() && (prop[p] == ' ' || prop[p] == '\n'
               || prop[p] == '\t' || prop[p] == '*'))
            ++p;
        search_area = prop.substr(p, find_dest_end(p) - p);
        return true;
    };

    // Family B: " to:" or " to\n"
    auto try_family_b = [&]() -> bool {
        if (consume(verb_end, " to:") == std::string::npos &&
            consume(verb_end, " to\n") == std::string::npos)
            return false;
        std::size_t p = verb_end + 3;
        while (p < prop.size() && (prop[p] == ':' || prop[p] == ' '
               || prop[p] == '\n' || prop[p] == '\t'))
            ++p;
        search_area = prop.substr(p, find_dest_end(p) - p);
        return true;
    };

    // Family C: " to " + inline CSV
    auto try_family_c = [&]() -> bool {
        std::size_t p = consume(verb_end, " to ");
        if (p == std::string::npos) return false;
        search_area = prop.substr(p, find_dest_end(p) - p);
        return true;
    };

    if (!try_family_a() && !try_family_b() && !try_family_c())
        return 0;
    if (search_area.empty()) return 0;

    // Strip markdown emphasis from the destination block.
    {
        std::string clean;
        clean.reserve(search_area.size());
        for (char c : search_area) {
            if (c != '*') clean += c;
        }
        search_area = clean;
    }

    // Parse the destination as a comma-separated address line:
    // "street, unit, city, state zip[, country]"
    std::vector<std::string> fields;
    {
        std::size_t sp = 0;
        while (sp < search_area.size()) {
            std::size_t comma = search_area.find(',', sp);
            if (comma == std::string::npos) comma = search_area.size();
            std::string f = search_area.substr(sp, comma - sp);
            while (!f.empty() && (f.front() == ' ' || f.front() == '\n'
                   || f.front() == '\t')) f.erase(f.begin());
            while (!f.empty() && (f.back() == ' ' || f.back() == '\n'
                   || f.back() == '\t' || f.back() == '.'))
                f.pop_back();
            if (!f.empty()) fields.push_back(f);
            sp = comma + 1;
        }
    }

    std::size_t fi = 0;
    if (fi >= fields.size() || fields[fi] != exp_a1) return 0;
    ++fi;
    if (!exp_a2.empty()) {
        if (fi >= fields.size() || fields[fi] != exp_a2) return 0;
        ++fi;
    }
    if (fi >= fields.size() || fields[fi] != exp_ci) return 0;
    ++fi;
    if (fi >= fields.size()) return 0;
    std::string state_zip = fields[fi];
    std::size_t sp_pos = state_zip.find(' ');
    if (sp_pos == std::string::npos) return 0;
    std::string parsed_state = state_zip.substr(0, sp_pos);
    std::string rest_zip = state_zip.substr(sp_pos);
    while (!rest_zip.empty() && rest_zip.front() == ' ')
        rest_zip.erase(rest_zip.begin());
    if (parsed_state != to_lower(exp_st)) return 0;
    if (rest_zip != exp_zip) return 0;
    ++fi;
    if (fi < fields.size()) {
        std::string parsed_co = fields[fi];
        if (!exp_co.empty() && parsed_co != exp_co) return 0;
        ++fi;
    } else {
        if (!exp_co.empty() && !is_default) return 0;
    }
    if (fi < fields.size()) return 0;
    return 1;
}

// Match val as a complete value in a clause section. Rejects matches
// where val is the tail of a longer multi-word value — e.g. "red"
// inside "dark red". The preceding word (if any) must be a structural
// keyword ("to", "from"), not a value qualifier like "dark" or "light".
static bool bounded_val_find_ext(const std::string& section,
                                  const std::string& val) {
    std::size_t pos = 0;
    while (true) {
        pos = section.find(val, pos);
        if (pos == std::string::npos) return false;
        std::size_t end = pos + val.size();
        bool right_ok = (end >= section.size() || !std::isalpha(
            static_cast<unsigned char>(section[end])));
        bool left_ok = true;
        if (pos > 0 && section[pos - 1] == ' ') {
            // Extract the preceding word.
            std::size_t ws = pos - 1;
            while (ws > 0 && section[ws - 1] == ' ') --ws;
            std::size_t we = ws;
            while (we > 0 && std::isalpha(
                       static_cast<unsigned char>(section[we - 1])))
                --we;
            std::string prev_word = section.substr(we, ws - we);
            // Structural keywords are OK; other words mean the
            // match is the tail of a compound value.
            if (!prev_word.empty() && prev_word != "to"
                && prev_word != "from" && prev_word != "keeping"
                && prev_word != "the" && prev_word != "a"
                && prev_word != "an" && prev_word != "and")
                left_ok = false;
        } else if (pos > 0 && std::isalpha(
                       static_cast<unsigned char>(section[pos - 1]))) {
            left_ok = false;
        }
        if (left_ok && right_ok) return true;
        ++pos;
    }
}

// 1 iff `val` appears (case-insensitive) in the "to" clause of
// the proposal's ITEM-CHANGE section. The item-change section is
// identified by a "from <old> to <new>" clause that sits AFTER an
// item-change heading ("item change", "item modification") or that
// contains product-variant language ("keeping", "options"). The
// address section's "from ... to ..." is explicitly skipped by
// requiring the surrounding context to NOT contain "address".
souffle::RamDomain proposal_item_option_in_to_clause(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain proposal_sym,
    souffle::RamDomain val_sym)
{
    std::string prop = to_lower(st->decode(proposal_sym));
    std::string val = to_lower(st->decode(val_sym));
    if (val.empty()) return 0;
    // Scan all " from " occurrences; pick the one in the item section.
    std::size_t pos = 0;
    while (true) {
        std::size_t from_pos = prop.find(" from ", pos);
        if (from_pos == std::string::npos) return 0;
        std::size_t to_pos = prop.find(" to ", from_pos + 6);
        if (to_pos == std::string::npos) { pos = from_pos + 6; continue; }
        // Reject if this from/to is inside the address section:
        // look backward to the nearest section heading or paragraph.
        std::size_t ctx_start = from_pos;
        while (ctx_start > 0 && prop[ctx_start-1] != '\n') --ctx_start;
        std::string context = prop.substr(ctx_start, from_pos - ctx_start);
        if (context.find("address") != std::string::npos) {
            pos = to_pos + 4;
            continue;
        }
        // Check the to-clause (up to sentence end or closing paren)
        std::size_t to_end = to_pos;
        while (to_end < prop.size() && prop[to_end] != '.'
               && prop[to_end] != '!' && prop[to_end] != '\n')
            ++to_end;
        std::string to_section = prop.substr(to_pos, to_end - to_pos);
        if (bounded_val_find_ext(to_section, val)) return 1;
        pos = to_pos + 4;
    }
}

// Same as proposal_item_option_in_to_clause but checks the "from"
// clause: the section between " from " and " to " in the item-change
// part. Used to verify the original item matches the proposal.
souffle::RamDomain proposal_item_option_in_from_clause(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain proposal_sym,
    souffle::RamDomain val_sym)
{
    std::string prop = to_lower(st->decode(proposal_sym));
    std::string val = to_lower(st->decode(val_sym));
    if (val.empty()) return 0;
    std::size_t pos = 0;
    while (true) {
        std::size_t from_pos = prop.find(" from ", pos);
        if (from_pos == std::string::npos) return 0;
        std::size_t to_pos = prop.find(" to ", from_pos + 6);
        if (to_pos == std::string::npos) { pos = from_pos + 6; continue; }
        std::size_t ctx_start = from_pos;
        while (ctx_start > 0 && prop[ctx_start-1] != '\n') --ctx_start;
        std::string context = prop.substr(ctx_start, from_pos - ctx_start);
        if (context.find("address") != std::string::npos) {
            pos = to_pos + 4;
            continue;
        }
        std::string from_section = prop.substr(from_pos, to_pos - from_pos);
        if (bounded_val_find_ext(from_section, val)) return 1;
        pos = to_pos + 4;
    }
}

// Helper: extract all option key-value pairs from a variant in the
// product JSON. Navigate: variants -> variant_id -> options -> {k:v,...}
namespace {

std::vector<std::pair<std::string, std::string>> extract_opts_from_obj(
    const char* opts_ptr)
{
    std::vector<std::pair<std::string, std::string>> result;
    const char* p = skip_ws(opts_ptr);
    if (*p != '{') return result;
    ++p;
    while (true) {
        p = skip_ws(p);
        if (*p == '}' || *p == 0) break;
        if (*p != '"') break;
        std::string key = read_key(p);
        p = skip_ws(skip_value(p));
        if (*p != ':') break;
        p = skip_ws(p + 1);
        if (*p == '"') {
            std::string val = read_key(p);
            p = skip_value(p);
            result.push_back({key, val});
        } else {
            p = skip_value(p);
        }
        p = skip_ws(p);
        if (*p == ',') ++p;
    }
    return result;
}

std::vector<std::pair<std::string, std::string>> extract_variant_options(
    const char* prod_json, const std::string& variant_id)
{
    const char* p = find_field(prod_json, "variants");
    if (!p) return {};
    p = find_field(p, variant_id);
    if (!p) return {};
    p = find_field(p, "options");
    if (!p) return {};
    return extract_opts_from_obj(p);
}

// Build {key -> set<value>} across ALL variants in the product.
std::map<std::string, std::set<std::string>> all_option_values(
    const char* prod_json)
{
    std::map<std::string, std::set<std::string>> domain;
    const char* vars = find_field(prod_json, "variants");
    if (!vars) return domain;
    const char* p = skip_ws(vars);
    if (*p != '{') return domain;
    ++p;
    while (true) {
        p = skip_ws(p);
        if (*p == '}' || *p == 0) break;
        if (*p != '"') break;
        p = skip_ws(skip_value(p));
        if (*p != ':') break;
        p = skip_ws(p + 1);
        const char* var_obj = p;
        const char* opts = find_field(var_obj, "options");
        if (opts) {
            auto kv = extract_opts_from_obj(opts);
            for (auto& [k, v] : kv)
                domain[k].insert(to_lower(v));
        }
        p = skip_value(var_obj);
        p = skip_ws(p);
        if (*p == ',') ++p;
    }
    return domain;
}

bool val_in_item_to_clause(const std::string& prop, const std::string& val) {
    std::size_t pos = 0;
    while (true) {
        std::size_t from_pos = prop.find(" from ", pos);
        if (from_pos == std::string::npos) return false;
        std::size_t to_pos = prop.find(" to ", from_pos + 6);
        if (to_pos == std::string::npos) { pos = from_pos + 6; continue; }
        std::size_t ctx_start = from_pos;
        while (ctx_start > 0 && prop[ctx_start-1] != '\n') --ctx_start;
        std::string context = prop.substr(ctx_start, from_pos - ctx_start);
        if (context.find("address") != std::string::npos) {
            pos = to_pos + 4; continue;
        }
        std::size_t to_end = to_pos;
        while (to_end < prop.size() && prop[to_end] != '.'
               && prop[to_end] != '!' && prop[to_end] != '\n')
            ++to_end;
        std::string to_section = prop.substr(to_pos, to_end - to_pos);
        if (bounded_val_find_ext(to_section, val)) return true;
        pos = to_pos + 4;
    }
}

bool val_in_item_from_clause(const std::string& prop, const std::string& val) {
    std::size_t pos = 0;
    while (true) {
        std::size_t from_pos = prop.find(" from ", pos);
        if (from_pos == std::string::npos) return false;
        std::size_t to_pos = prop.find(" to ", from_pos + 6);
        if (to_pos == std::string::npos) { pos = from_pos + 6; continue; }
        std::size_t ctx_start = from_pos;
        while (ctx_start > 0 && prop[ctx_start-1] != '\n') --ctx_start;
        std::string context = prop.substr(ctx_start, from_pos - ctx_start);
        if (context.find("address") != std::string::npos) {
            pos = to_pos + 4; continue;
        }
        std::string from_section = prop.substr(from_pos, to_pos - from_pos);
        if (bounded_val_find_ext(from_section, val)) return true;
        pos = to_pos + 4;
    }
}
} // namespace

// 1 iff the proposal correctly covers ALL option changes between old
// and new variants: every differing key's new value is in the to-clause
// and old value is in the from-clause, and every value mentioned in a
// "keeping" clause matches the actual new variant. Returns 0 if any
// option change is uncovered or any kept value is incorrect.
souffle::RamDomain variant_has_uncovered_option_change(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain prod_sym,
    souffle::RamDomain old_id_sym,
    souffle::RamDomain new_id_sym,
    souffle::RamDomain proposal_sym)
{
    const std::string& prod_json = st->decode(prod_sym);
    const std::string old_id = st->decode(old_id_sym);
    const std::string new_id = st->decode(new_id_sym);
    std::string proposal = to_lower(st->decode(proposal_sym));

    auto old_opts = extract_variant_options(prod_json.c_str(), old_id);
    auto new_opts = extract_variant_options(prod_json.c_str(), new_id);

    std::map<std::string, std::string> old_map, new_map;
    for (auto& p : old_opts) old_map[p.first] = to_lower(p.second);
    for (auto& p : new_opts) new_map[p.first] = to_lower(p.second);

    // Collect ALL keys from both variants.
    std::map<std::string, bool> all_keys;
    for (auto& p : old_map) all_keys[p.first] = true;
    for (auto& p : new_map) all_keys[p.first] = true;

    // For each key where values differ: new_val must be in to-clause,
    // old_val must be in from-clause.
    for (auto& [key, _] : all_keys) {
        std::string ov = old_map.count(key) ? old_map[key] : "";
        std::string nv = new_map.count(key) ? new_map[key] : "";
        if (ov == nv) continue;
        if (!nv.empty() && !val_in_item_to_clause(proposal, nv)) return 1;
        if (!ov.empty() && !val_in_item_from_clause(proposal, ov)) return 1;
    }

    // Check "keeping" clause: any catalog option value that appears
    // in "(keeping X, Y, ...)" must match the new variant's actual
    // value for that option key. E.g. "keeping 4-piece" when the
    // new variant is 2-piece → mismatch, even when old is also 2-piece.
    std::size_t keep_pos = proposal.find("keeping");
    if (keep_pos != std::string::npos) {
        std::size_t paren_start = proposal.find('(', keep_pos > 20 ? keep_pos - 20 : 0);
        if (paren_start != std::string::npos && paren_start < keep_pos + 10) {
            std::size_t paren_end = proposal.find(')', keep_pos);
            if (paren_end != std::string::npos) {
                std::string keeping = proposal.substr(keep_pos + 7,
                                                      paren_end - keep_pos - 7);
                auto domain = all_option_values(prod_json.c_str());
                for (auto& [key, vals] : domain) {
                    std::string nv = new_map.count(key) ? new_map[key] : "";
                    for (auto& dv : vals) {
                        if (dv == nv) continue;
                        if (bounded_val_find_ext(keeping, dv))
                            return 1;
                    }
                }
            }
        }
    }

    return 0;
}

// Unified preserved-consent item classifier. Returns 1 iff the
// proposal contains ONE affirmative item-action sentence with a
// complete from/to/keeping clause whose extracted values match
// the actual variant options by normalized equality.
//
// Supported sentence shape (the original positive):
//   "We will change your Luggage Set from Silver to Red
//    (keeping the Hardshell, 2-piece options)."
//
// Extracts:
//   from_value = "silver"
//   to_value   = "red"
//   keeping    = ["hardshell", "2-piece"]
//
// Compares each extracted value by exact normalized equality to
// the actual variant option values. "Red Orange" != "Red".
souffle::RamDomain proposal_item_action_verified(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain prod_sym,
    souffle::RamDomain old_id_sym,
    souffle::RamDomain new_id_sym,
    souffle::RamDomain proposal_sym)
{
    const std::string& prod_json = st->decode(prod_sym);
    const std::string old_id = st->decode(old_id_sym);
    const std::string new_id = st->decode(new_id_sym);
    std::string prop = to_lower(st->decode(proposal_sym));

    auto old_opts = extract_variant_options(prod_json.c_str(), old_id);
    auto new_opts = extract_variant_options(prod_json.c_str(), new_id);
    std::map<std::string, std::string> old_map, new_map;
    for (auto& p : old_opts) old_map[p.first] = to_lower(p.second);
    for (auto& p : new_opts) new_map[p.first] = to_lower(p.second);

    // Find each non-address " from ... to ..." clause.
    std::size_t pos = 0;
    while (true) {
        std::size_t from_pos = prop.find(" from ", pos);
        if (from_pos == std::string::npos) return 0;
        std::size_t to_pos = prop.find(" to ", from_pos + 6);
        if (to_pos == std::string::npos) { pos = from_pos + 6; continue; }

        // Skip address-section from/to.
        std::size_t line_start = from_pos;
        while (line_start > 0 && prop[line_start-1] != '\n') --line_start;
        std::string full_line = prop.substr(line_start,
            to_pos + 4 - line_start);
        if (full_line.find("address") != std::string::npos) {
            pos = to_pos + 4; continue;
        }

        // Check for affirmative verb before " from " in this sentence.
        std::size_t sb = from_pos;
        while (sb > 0 && prop[sb-1] != '.' && prop[sb-1] != '!'
               && prop[sb-1] != '\n') --sb;
        std::string sentence_before = prop.substr(sb, from_pos - sb);
        // Verbs with trailing product-name slot: the text between
        // the verb and " from " must be the catalog product name.
        struct VerbEntry { const char* text; bool has_product_slot; };
        static const VerbEntry kVerbs[] = {
            {"we will change your ", true},
            {"we'll change your ", true},
            {"we will update your ", true},
            {"we'll update your ", true},
            {"we will modify your ", true},
            {"we'll modify your ", true},
            {"we will switch your ", true},
            {"we'll switch your ", true},
            {"change your ", true},
            {nullptr, false},
        };
        // Extract the catalog product name from prod_json.
        std::string catalog_name;
        {
            const char* np = find_field(prod_json.c_str(), "name");
            if (np) catalog_name = to_lower(read_key(np));
        }
        bool has_verb = false;
        for (const VerbEntry* v = kVerbs; v->text; ++v) {
            std::size_t vpos = sentence_before.find(v->text);
            if (vpos == std::string::npos) continue;
            bool clean = true;
            for (std::size_t i = 0; i < vpos; ++i) {
                char c = sentence_before[i];
                if (c != ' ' && c != '\t' && c != '\n' && c != '*') {
                    clean = false; break;
                }
            }
            if (!clean) continue;
            if (v->has_product_slot) {
                std::size_t name_start = vpos + std::strlen(v->text);
                std::string found_name = sentence_before.substr(name_start);
                while (!found_name.empty() && found_name.back() == ' ')
                    found_name.pop_back();
                if (catalog_name.empty() || found_name != catalog_name)
                    continue;
            }
            has_verb = true;
            break;
        }
        if (!has_verb) { pos = to_pos + 4; continue; }

        // Extract COMPLETE from-value: text between " from " and " to ".
        std::string from_val = prop.substr(from_pos + 6,
                                            to_pos - from_pos - 6);
        // Trim whitespace.
        while (!from_val.empty() && from_val.back() == ' ')
            from_val.pop_back();
        while (!from_val.empty() && from_val.front() == ' ')
            from_val.erase(from_val.begin());

        // Extract COMPLETE to-value: text between " to " and the next
        // boundary — '(' for keeping clause, '.' or '\n' for sentence end.
        std::size_t to_start = to_pos + 4;
        std::size_t to_end = to_start;
        while (to_end < prop.size() && prop[to_end] != '('
               && prop[to_end] != '.' && prop[to_end] != '!'
               && prop[to_end] != '\n') ++to_end;
        std::string to_val = prop.substr(to_start, to_end - to_start);
        while (!to_val.empty() && to_val.back() == ' ')
            to_val.pop_back();
        while (!to_val.empty() && to_val.front() == ' ')
            to_val.erase(to_val.begin());

        if (from_val.empty() || to_val.empty()) {
            pos = to_pos + 4; continue;
        }

        // Extract keeping-clause values if present. If '(' follows
        // the to-value, it MUST be "(keeping [the] X, Y options)".
        // Unknown parentheticals reject. After ')', only terminal
        // punctuation is allowed — no arbitrary suffix.
        std::vector<std::string> keeping_vals;
        std::size_t sentence_end = to_end;
        if (to_end < prop.size() && prop[to_end] == '(') {
            std::size_t keep_start = to_end + 1;
            while (keep_start < prop.size() && prop[keep_start] == ' ')
                ++keep_start;
            if (keep_start + 7 > prop.size()
                || prop.compare(keep_start, 7, "keeping") != 0) {
                pos = to_pos + 4; continue;
            }
            std::size_t paren_end = prop.find(')', keep_start);
            if (paren_end == std::string::npos) {
                pos = to_pos + 4; continue;
            }
            std::string keep_body = prop.substr(keep_start + 7,
                paren_end - keep_start - 7);
            while (!keep_body.empty() && keep_body.front() == ' ')
                keep_body.erase(keep_body.begin());
            if (keep_body.substr(0, 4) == "the ")
                keep_body = keep_body.substr(4);
            // The body must end with " options" or " option".
            bool has_options_suffix = false;
            {
                std::size_t opt_pos = keep_body.find(" options");
                if (opt_pos != std::string::npos
                    && opt_pos + 8 == keep_body.size()) {
                    keep_body = keep_body.substr(0, opt_pos);
                    has_options_suffix = true;
                }
                if (!has_options_suffix) {
                    opt_pos = keep_body.find(" option");
                    if (opt_pos != std::string::npos
                        && opt_pos + 7 == keep_body.size()) {
                        keep_body = keep_body.substr(0, opt_pos);
                        has_options_suffix = true;
                    }
                }
            }
            if (!has_options_suffix) {
                pos = to_pos + 4; continue;
            }
            // Split on ", " to get individual kept values.
            std::size_t sp = 0;
            while (sp < keep_body.size()) {
                std::size_t comma = keep_body.find(", ", sp);
                if (comma == std::string::npos)
                    comma = keep_body.size();
                std::string kv = keep_body.substr(sp, comma - sp);
                while (!kv.empty() && kv.front() == ' ')
                    kv.erase(kv.begin());
                while (!kv.empty() && kv.back() == ' ')
                    kv.pop_back();
                if (!kv.empty()) keeping_vals.push_back(kv);
                sp = comma + 2;
            }
            sentence_end = paren_end + 1;
        }
        // After the action clause (to-value or keeping ')'),
        // only terminal punctuation is allowed. Reject unknown tails.
        {
            std::size_t tail = sentence_end;
            while (tail < prop.size() && prop[tail] == ' ') ++tail;
            if (tail < prop.size() && prop[tail] != '.'
                && prop[tail] != '!' && prop[tail] != '\n'
                && prop[tail] != '*') {
                pos = to_pos + 4; continue;
            }
        }

        // Now compare by exact equality to actual variant options.
        // Find the ONE differing key between old and new variants.
        std::map<std::string, bool> all_keys;
        for (auto& p : old_map) all_keys[p.first] = true;
        for (auto& p : new_map) all_keys[p.first] = true;

        std::string diff_key;
        int diff_count = 0;
        for (auto& [key, _] : all_keys) {
            std::string ov = old_map.count(key) ? old_map[key] : "";
            std::string nv = new_map.count(key) ? new_map[key] : "";
            if (ov != nv) { diff_key = key; ++diff_count; }
        }
        // Conservative: support only single-key difference.
        if (diff_count != 1) { pos = to_pos + 4; continue; }

        std::string expected_from = old_map.count(diff_key)
            ? old_map[diff_key] : "";
        std::string expected_to = new_map.count(diff_key)
            ? new_map[diff_key] : "";

        // Exact equality of the differing option value.
        if (from_val != expected_from) { pos = to_pos + 4; continue; }
        if (to_val != expected_to) { pos = to_pos + 4; continue; }

        // Verify keeping-clause: each kept value must match an
        // actual new-variant option value for a non-differing key.
        bool keeping_ok = true;
        for (auto& kv : keeping_vals) {
            bool found = false;
            for (auto& [key, nv] : new_map) {
                if (key == diff_key) continue;
                if (nv == kv) { found = true; break; }
            }
            if (!found) { keeping_ok = false; break; }
        }
        if (!keeping_ok) { pos = to_pos + 4; continue; }

        // All unchanged keys must be accounted for by a kept value
        // if keeping values were given.
        if (!keeping_vals.empty()) {
            for (auto& [key, _] : all_keys) {
                if (key == diff_key) continue;
                std::string nv = new_map.count(key) ? new_map[key] : "";
                if (nv.empty()) continue;
                bool covered = false;
                for (auto& kv : keeping_vals) {
                    if (kv == nv) { covered = true; break; }
                }
                if (!covered) { keeping_ok = false; break; }
            }
        }
        if (!keeping_ok) { pos = to_pos + 4; continue; }

        return 1;
    }
}

// 1 iff the proposal contains an affirmative item-change clause:
// a recognized verb followed by " from " and " to " in the SAME
// non-address sentence. The verb must start at a sentence boundary
// (no conditional/illustrative prefix). This binds the affirmative
// check to the exact same from/to clause that variant option
// matching validates.
souffle::RamDomain proposal_has_affirmative_item_action(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain proposal_sym)
{
    std::string prop = to_lower(st->decode(proposal_sym));
    // Find each non-address " from ... to ..." clause.
    std::size_t pos = 0;
    while (true) {
        std::size_t from_pos = prop.find(" from ", pos);
        if (from_pos == std::string::npos) return 0;
        std::size_t to_pos = prop.find(" to ", from_pos + 6);
        if (to_pos == std::string::npos) { pos = from_pos + 6; continue; }
        // Skip address-section from/to.
        std::size_t ctx_start = from_pos;
        while (ctx_start > 0 && prop[ctx_start-1] != '\n') --ctx_start;
        std::string full_line = prop.substr(ctx_start,
            to_pos + 4 - ctx_start);
        if (full_line.find("address") != std::string::npos) {
            pos = to_pos + 4; continue;
        }
        // Walk backward from " from " to find the sentence start.
        std::size_t sb = from_pos;
        while (sb > 0 && prop[sb-1] != '.' && prop[sb-1] != '!'
               && prop[sb-1] != '\n') --sb;
        std::string sentence = prop.substr(sb, from_pos - sb);
        // The sentence must contain a recognized affirmative verb.
        static const char* kVerbs[] = {
            "we will change ", "we'll change ",
            "we will update ", "we'll update ",
            "we will modify ", "we'll modify ",
            "we will switch ", "we'll switch ",
            "change your ",
            nullptr,
        };
        for (const char** v = kVerbs; *v; ++v) {
            std::size_t vpos = sentence.find(*v);
            if (vpos == std::string::npos) continue;
            // The verb must start the sentence (after whitespace/
            // markdown). No conditional/illustrative prefix allowed.
            bool prefix_clean = true;
            for (std::size_t i = 0; i < vpos; ++i) {
                char c = sentence[i];
                if (c != ' ' && c != '\t' && c != '\n' && c != '*') {
                    prefix_clean = false;
                    break;
                }
            }
            if (!prefix_clean) continue;
            return 1;
        }
        pos = to_pos + 4;
    }
}

// 1 iff the user's YES text explicitly references a card ending that
// differs from any card ending mentioned in the proposal text.
// Compares the user's "card ending in XXXX" against the proposal's
// "card ending in YYYY"; if both mention a card ending and they
// differ, the user changed their payment selection.
souffle::RamDomain user_yes_payment_mismatch(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym,
    souffle::RamDomain proposal_sym)
{
    auto extract_card_ending = [](const std::string& s) -> std::string {
        static const char* kEnding[] = {
            "ending in ", "ends in ", nullptr
        };
        std::string found;
        for (const char** pat = kEnding; *pat; ++pat) {
            std::size_t pos = 0;
            while ((pos = s.find(*pat, pos)) != std::string::npos) {
                std::size_t start = pos + std::strlen(*pat);
                while (start < s.size() && (s[start] == '*'
                       || s[start] == ' ')) ++start;
                std::string digits;
                while (start < s.size() &&
                       std::isdigit(static_cast<unsigned char>(s[start])))
                    digits += s[start++];
                if (digits.size() == 4) {
                    if (found.empty()) {
                        found = digits;
                    } else if (found != digits) {
                        return "";
                    }
                }
                pos = start;
            }
        }
        return found;
    };
    std::string text = to_lower(st->decode(text_sym));
    std::string proposal = to_lower(st->decode(proposal_sym));
    std::string user_ending = extract_card_ending(text);
    std::string prop_ending = extract_card_ending(proposal);
    if (user_ending.empty() || prop_ending.empty()) return 0;
    return user_ending != prop_ending ? 1 : 0;
}

// Returns the confirmed dollar amount in whole cents from the user's
// text, or -1 if no amount is found. Looks for "$X.XX" patterns.
souffle::RamDomain user_confirmed_charge_cents(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string& text = st->decode(text_sym);
    std::size_t pos = 0;
    while ((pos = text.find('$', pos)) != std::string::npos) {
        ++pos;
        while (pos < text.size() && text[pos] == '*') ++pos;
        std::string num;
        while (pos < text.size() && (std::isdigit(static_cast<unsigned char>(
                   text[pos])) || text[pos] == '.'))
            num += text[pos++];
        if (!num.empty()) {
            double val = std::atof(num.c_str());
            return static_cast<souffle::RamDomain>(std::round(val * 100));
        }
    }
    return -1;
}

// Recognized whole-message information-only interjection for the
// preserved-consent path. Positive grammar — only messages that
// match a recognized single-purpose info-request pattern pass.
// The supported form is a tracking / status / delivery question
// that is structurally a standalone request with no write-action,
// conditional, or substantive content.
//
// Supported shape: "[optional filler —] can you also send me the
// tracking number for the order I cancelled/returned?" and close
// structural variants. Single recognized sentence with an info
// keyword, a request verb ("can you", "could you", "send me"),
// and a trailing question mark. Every sentence in the message must
// be recognized; unknown sentences deny.
souffle::RamDomain user_info_only_interjection(souffle::SymbolTable* st,
                                               souffle::RecordTable*,
                                               souffle::RamDomain text_sym) {
    const std::string raw = st->decode(text_sym);
    std::string lower = to_lower(raw);
    if (lower.empty()) return 0;

    // Normalize em-dash / double-dash to comma-space for splitting.
    {
        std::string norm;
        norm.reserve(lower.size());
        for (std::size_t i = 0; i < lower.size(); ++i) {
            unsigned char c = static_cast<unsigned char>(lower[i]);
            if (c == 0xe2 && i + 2 < lower.size() &&
                static_cast<unsigned char>(lower[i+1]) == 0x80 &&
                (static_cast<unsigned char>(lower[i+2]) == 0x93 ||
                 static_cast<unsigned char>(lower[i+2]) == 0x94)) {
                norm += ", ";
                i += 2;
            } else if (c == '-' && i + 1 < lower.size() && lower[i+1] == '-') {
                norm += ", ";
                ++i;
            } else {
                norm += lower[i];
            }
        }
        lower = norm;
    }

    // Split into sentences. Skip "." inside numbers.
    auto is_sent_end = [&](std::size_t i) -> bool {
        char c = lower[i];
        if (c == '!' || c == '?' || c == '\n') return true;
        if (c == '.') {
            if (i + 1 < lower.size() &&
                std::isdigit(static_cast<unsigned char>(lower[i + 1])))
                return false;
            if (i > 0 &&
                std::isdigit(static_cast<unsigned char>(lower[i - 1])))
                return false;
            return true;
        }
        return false;
    };
    std::vector<std::string> sentences;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= lower.size(); ++i) {
        bool at_end = (i == lower.size());
        bool split = at_end || is_sent_end(i);
        if (split) {
            std::string s = lower.substr(start, i - start);
            while (!s.empty() && (s.front() == ' ' || s.front() == '\t'
                   || s.front() == '*' || s.front() == ','))
                s.erase(s.begin());
            while (!s.empty() && (s.back() == ' ' || s.back() == '\t'
                   || s.back() == '*'))
                s.pop_back();
            if (!s.empty()) sentences.push_back(s);
            start = i + 1;
        }
    }
    if (sentences.empty()) return 0;

    // Recognized info-request sentence: contains an info keyword AND
    // a request verb.
    static const char* info_kw[] = {
        "tracking", "track", "status", "delivery", "shipped",
        "shipping", "estimated", "arrive", "arrival", "eta",
        "where is", "when will", nullptr,
    };
    static const char* req_verbs[] = {
        "can you", "could you", "send me", "give me",
        "tell me", "let me know", "what is", "what's",
        "when will", "where is", nullptr,
    };
    // Complete-parse info-request grammar. The ENTIRE sentence must
    // match one of a finite set of recognized whole-sentence patterns.
    // Structure: [filler prefix] + request-verb + [also/optional] +
    //   send/give/tell me + [the] + info-object + [qualifier].
    // Qualifier: "for the order I cancelled/returned" and supported
    //   variants. Every character must be consumed; unknown words or
    //   additional clauses (conjunctions, instructions) deny.
    auto is_info_request = [&](const std::string& s) -> bool {
        std::string body = s;
        // Strip recognized filler prefix (em-dash-normalized).
        {
            std::size_t comma = body.find(", ");
            if (comma != std::string::npos && comma < 30) {
                std::string prefix = body.substr(0, comma);
                while (!prefix.empty() && prefix.back() == ' ')
                    prefix.pop_back();
                static const char* kFP[] = {
                    "just checking in", "by the way",
                    "one more thing", "also", "oh", "hey",
                    "hi", nullptr,
                };
                for (const char** f = kFP; *f; ++f) {
                    if (prefix == *f) {
                        body = body.substr(comma + 2);
                        while (!body.empty() && body.front() == ' ')
                            body.erase(body.begin());
                        break;
                    }
                }
            }
        }
        // Consume request-verb prefix.
        bool verb_found = false;
        for (const char** p = req_verbs; *p; ++p) {
            std::size_t n = std::strlen(*p);
            if (body.size() >= n && body.compare(0, n, *p) == 0) {
                body = body.substr(n);
                verb_found = true;
                break;
            }
        }
        if (!verb_found) return false;
        // Consume optional words: "also", "please".
        while (!body.empty() && body.front() == ' ')
            body.erase(body.begin());
        for (const char* opt : {"also ", "please "}) {
            std::size_t n = std::strlen(opt);
            if (body.size() >= n && body.compare(0, n, opt) == 0)
                body = body.substr(n);
        }
        // Consume optional transfer verb: "send me", "give me",
        // "tell me", "let me know".
        for (const char* tv : {"send me ", "give me ", "tell me ",
                               "let me know "}) {
            std::size_t n = std::strlen(tv);
            if (body.size() >= n && body.compare(0, n, tv) == 0) {
                body = body.substr(n); break;
            }
        }
        // Consume optional article "the ".
        if (body.size() >= 4 && body.compare(0, 4, "the ") == 0)
            body = body.substr(4);
        // Consume info-object: one of the recognized info keywords.
        bool info_found = false;
        for (const char** p = info_kw; *p; ++p) {
            std::size_t n = std::strlen(*p);
            if (body.size() >= n && body.compare(0, n, *p) == 0) {
                body = body.substr(n);
                info_found = true;
                break;
            }
        }
        if (!info_found) return false;
        // Consume optional noun completions: " number", " info",
        // " information", " details", " update".
        while (!body.empty() && body.front() == ' ')
            body.erase(body.begin());
        for (const char* nc : {"number", "info", "information",
                               "details", "update"}) {
            std::size_t n = std::strlen(nc);
            if (body.size() >= n && body.compare(0, n, nc) == 0) {
                body = body.substr(n); break;
            }
        }
        while (!body.empty() && body.front() == ' ')
            body.erase(body.begin());
        // If fully consumed, accept.
        if (body.empty()) return true;
        // Consume optional bounded qualifier: "for [the/my] order
        // [I/that I] [cancelled/returned/placed/shipped]" and close
        // variants. Must consume the ENTIRE remainder.
        static const char* kQualifiers[] = {
            "for the order i cancelled/returned",
            "for the order i cancelled",
            "for the order i returned",
            "for the order i placed",
            "for my order",
            "for the order",
            "for that order",
            "for order",
            nullptr,
        };
        for (const char** q = kQualifiers; *q; ++q) {
            std::size_t n = std::strlen(*q);
            if (body.size() >= n && body.compare(0, n, *q) == 0) {
                body = body.substr(n);
                while (!body.empty() && body.front() == ' ')
                    body.erase(body.begin());
                // Allow an order ID after "for order": "#WNNNN..."
                if (!body.empty() && body[0] == '#') {
                    std::size_t i = 1;
                    while (i < body.size() &&
                           std::isalnum((unsigned char)body[i]))
                        ++i;
                    body = body.substr(i);
                }
                break;
            }
        }
        while (!body.empty() && body.front() == ' ')
            body.erase(body.begin());
        return body.empty();
    };

    // Recognized filler sentence: "just checking in", "by the way",
    // "one more thing", "also". Short and harmless.
    auto is_filler = [](const std::string& s) -> bool {
        static const char* fillers[] = {
            "just checking in", "by the way", "one more thing",
            "also", "oh", "hey", "hi", nullptr,
        };
        for (const char** p = fillers; *p; ++p) {
            if (s == *p) return true;
        }
        return false;
    };

    // Every sentence must be recognized as either info-request or filler.
    bool has_info_req = false;
    for (const auto& s : sentences) {
        if (is_info_request(s)) { has_info_req = true; continue; }
        if (is_filler(s)) continue;
        return 0;
    }
    return has_info_req ? 1 : 0;
}

// 1 iff `token` appears in `text` as a whole token (bounded by
// non-alphanumeric characters or string boundaries on both sides).
// Prevents order-ID prefix collisions: "#W123" must not match
// "#W1234". Case-sensitive.
souffle::RamDomain str_contains_exact_token(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym,
    souffle::RamDomain token_sym)
{
    const std::string& text = st->decode(text_sym);
    const std::string& token = st->decode(token_sym);
    if (token.empty()) return 0;
    auto is_id_char = [](char c) -> bool {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    };
    std::size_t pos = 0;
    while ((pos = text.find(token, pos)) != std::string::npos) {
        bool left_ok = (pos == 0 || !is_id_char(text[pos - 1]));
        std::size_t end = pos + token.size();
        bool right_ok = (end >= text.size() || !is_id_char(text[end]));
        if (left_ok && right_ok) return 1;
        pos = end;
    }
    return 0;
}

// Count how many distinct order IDs (matching #W<alnum>+ pattern)
// appear in the text. Used to reject ambiguous multi-order proposals.
souffle::RamDomain count_order_ids(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    const std::string& text = st->decode(text_sym);
    std::set<std::string> ids;
    std::size_t pos = 0;
    while ((pos = text.find("#W", pos)) != std::string::npos) {
        std::size_t end = pos + 2;
        while (end < text.size() &&
               std::isalnum(static_cast<unsigned char>(text[end]))) ++end;
        if (end > pos + 2) ids.insert(text.substr(pos, end - pos));
        pos = end;
    }
    return static_cast<souffle::RamDomain>(ids.size());
}

// Extract the payable charge from the proposal's ITEM CHANGE section
// by fullmatch parsing a COMPLETE recognized pricing sentence.
// Returns the difference in cents, or -1 if none found.
//
// Supported sentence (the original positive):
//   "the <new> set costs $<new_price>, which is $<diff> more than
//    the original <old> set ($<old_price>)."
//
// The parser consumes every character of a candidate sentence. Labels
// are sequences of non-punctuation words. Prices are $X.XX decimals.
// Unknown sentence shapes return no proof (-1), which requires fresh
// confirmation in the Datalog.
souffle::RamDomain proposal_charge_cents(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    std::string text = to_lower(st->decode(text_sym));
    // Find the item-change section start.
    static const char* kItemMarkers[] = {
        "item change", "item modification", "item swap",
        "color change", "variant change", nullptr,
    };
    std::size_t section_start = std::string::npos;
    for (const char** m = kItemMarkers; *m; ++m) {
        std::size_t p = text.find(*m);
        if (p != std::string::npos && (section_start == std::string::npos
            || p < section_start)) section_start = p;
    }
    if (section_start == std::string::npos) {
        section_start = text.find("from ");
        if (section_start == std::string::npos) return -1;
    }

    // Section end: next "**N." heading or blank line.
    std::size_t section_end = text.size();
    for (std::size_t i = section_start + 1; i + 1 < text.size(); ++i) {
        if (i + 3 < text.size()
            && text[i] == '*' && text[i + 1] == '*'
            && std::isdigit(static_cast<unsigned char>(text[i + 2]))
            && text[i + 3] == '.') {
            section_end = i;
            break;
        }
        if (text[i] == '\n' && i + 1 < text.size() && text[i + 1] == '\n') {
            section_end = i;
            break;
        }
    }

    // Helper: try to consume a $X.XX amount starting at position p in s.
    // On success, writes the cents value and advances p past the amount.
    auto try_consume_amount = [](const std::string& s, std::size_t& p,
                                  int& cents) -> bool {
        if (p >= s.size() || s[p] != '$') return false;
        std::size_t q = p + 1;
        while (q < s.size() && s[q] == '*') ++q;
        std::string num;
        while (q < s.size() && (std::isdigit(static_cast<unsigned char>(
                   s[q])) || s[q] == '.'))
            num += s[q++];
        if (num.empty()) return false;
        double val = std::atof(num.c_str());
        cents = static_cast<int>(std::round(val * 100));
        p = q;
        return true;
    };

    // Helper: consume exact literal at p.
    auto consume = [](const std::string& s, std::size_t& p,
                       const char* lit) -> bool {
        std::size_t n = std::strlen(lit);
        if (p + n > s.size()) return false;
        if (s.compare(p, n, lit) != 0) return false;
        p += n;
        return true;
    };

    // Helper: find the FIRST occurrence of `delim` starting at p,
    // verify that the text between p and that occurrence is a non-empty
    // label (letters, digits, hyphens, spaces), and advance p past it.
    auto consume_label_until = [](const std::string& s, std::size_t& p,
                                   const char* delim) -> bool {
        std::size_t dn = std::strlen(delim);
        std::size_t found = s.find(delim, p);
        if (found == std::string::npos || found == p) return false;
        for (std::size_t i = p; i < found; ++i) {
            char c = s[i];
            if (!std::isalpha(static_cast<unsigned char>(c))
                && !std::isdigit(static_cast<unsigned char>(c))
                && c != '-' && c != ' ')
                return false;
        }
        p = found;
        return true;
    };

    // Split the bounded section into sentences and try each.
    int found_cents = -1;
    int match_count = 0;
    std::size_t sent_pos = section_start;

    while (sent_pos < section_end) {
        // Find sentence start (skip whitespace/markdown).
        while (sent_pos < section_end
               && (text[sent_pos] == ' ' || text[sent_pos] == '\t'
                   || text[sent_pos] == '\n' || text[sent_pos] == '*'))
            ++sent_pos;
        if (sent_pos >= section_end) break;
        // Find sentence end. Skip '.' inside numbers ($532.58).
        std::size_t sent_end = sent_pos;
        while (sent_end < section_end) {
            char c = text[sent_end];
            if (c == '!' || c == '\n') break;
            if (c == '.') {
                bool in_num =
                    (sent_end > 0 && std::isdigit(static_cast<unsigned char>(
                         text[sent_end - 1])))
                    || (sent_end + 1 < section_end
                        && std::isdigit(static_cast<unsigned char>(
                               text[sent_end + 1])));
                if (!in_num) break;
            }
            ++sent_end;
        }
        // Advance past the recognized sentence delimiter.
        if (sent_end < section_end
            && (text[sent_end] == '.' || text[sent_end] == '!'
                || text[sent_end] == '\n'))
            ++sent_end;
        std::string sent = text.substr(sent_pos, sent_end - sent_pos);
        sent_pos = sent_end;

        // Try fullmatch: "the <label> set costs $<np>, which is $<diff>
        //   more than the original <label> set ($<op>)."
        {
            std::size_t p = 0;
            int new_price_cents = 0, diff_cents = 0, old_price_cents = 0;
            if (consume(sent, p, "the ")
                && consume_label_until(sent, p, " set costs ")
                && consume(sent, p, " set costs ")
                && try_consume_amount(sent, p, new_price_cents)
                && consume(sent, p, ", which is ")
                && try_consume_amount(sent, p, diff_cents)
                && consume(sent, p, " more than the original ")
                && consume_label_until(sent, p, " set (")
                && consume(sent, p, " set (")
                && try_consume_amount(sent, p, old_price_cents)
                && consume(sent, p, ").")) {
                // Require complete consumption.
                while (p < sent.size()
                       && (sent[p] == ' ' || sent[p] == '*'))
                    ++p;
                if (p == sent.size()) {
                    ++match_count;
                    if (found_cents == -1) {
                        found_cents = diff_cents;
                    } else if (found_cents != diff_cents) {
                        return -1;
                    }
                }
            }
        }
    }

    return (match_count == 1) ? found_cents : -1;
}

// Extract EXACTLY four digits from the first "ending in NNNN" or
// "ends in NNNN" pattern in the text. Returns "" if none found or
// if the digit sequence is not exactly 4 characters (malformed
// selections do not supply proof and must not fall through to
// inheritance).
souffle::RamDomain extract_card_ending_from_text(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    std::string text = to_lower(st->decode(text_sym));
    static const char* kEnding[] = {
        "ending in ", "ends in ", nullptr
    };
    std::string found_ending;
    for (const char** pat = kEnding; *pat; ++pat) {
        std::size_t pos = 0;
        while ((pos = text.find(*pat, pos)) != std::string::npos) {
            std::size_t start = pos + std::strlen(*pat);
            while (start < text.size() && (text[start] == '*'
                   || text[start] == ' ')) ++start;
            std::string digits;
            while (start < text.size() &&
                   std::isdigit(static_cast<unsigned char>(text[start])))
                digits += text[start++];
            if (digits.size() == 4) {
                if (found_ending.empty()) {
                    found_ending = digits;
                } else if (found_ending != digits) {
                    return st->encode("");
                }
            }
            pos = start;
        }
    }
    return st->encode(found_ending);
}

// Extract the card ending from the proposal's payment-selection
// clauses via the shared parse_payment_selections parser. Returns
// the unique card ending or "" if none/conflicting/malformed.
souffle::RamDomain proposal_payment_card_ending(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    std::string text = to_lower(st->decode(text_sym));
    auto s = summarize_selections(parse_payment_selections(text));
    return st->encode(s.card);
}

// Extract the country from the proposal's address section. Looks for
// the last token in a comma-separated address following the zip code.
// Returns "" if no country is found in the address destination.
souffle::RamDomain proposal_address_country(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain proposal_sym)
{
    std::string prop = to_lower(st->decode(proposal_sym));
    // Find the address section.
    static const char* kMarkers[] = {
        "address change", "shipping address",
        "update the shipping", "update the address",
        "change the shipping", "change the address",
        nullptr,
    };
    std::size_t addr_start = std::string::npos;
    for (const char** m = kMarkers; *m; ++m) {
        std::size_t p = prop.find(*m);
        if (p != std::string::npos && (addr_start == std::string::npos
            || p < addr_start)) addr_start = p;
    }
    if (addr_start == std::string::npos) return st->encode("");
    // Scan for a five-digit zip code after the address marker.
    std::string after_marker = prop.substr(addr_start);
    // Look for "to" clause if present (use destination only).
    std::size_t to_pos = after_marker.find(" to ");
    if (to_pos != std::string::npos)
        after_marker = after_marker.substr(to_pos + 4);
    // Find a 5-digit sequence (zip code).
    std::size_t zip_pos = std::string::npos;
    for (std::size_t i = 0; i + 4 < after_marker.size(); ++i) {
        bool all_digit = true;
        for (std::size_t j = 0; j < 5 && i + j < after_marker.size(); ++j) {
            if (!std::isdigit(static_cast<unsigned char>(
                    after_marker[i + j]))) { all_digit = false; break; }
        }
        if (all_digit && i + 5 <= after_marker.size()) {
            bool left = (i == 0 || !std::isdigit(
                static_cast<unsigned char>(after_marker[i - 1])));
            bool right = (i + 5 >= after_marker.size() || !std::isdigit(
                static_cast<unsigned char>(after_marker[i + 5])));
            if (left && right) { zip_pos = i; break; }
        }
    }
    if (zip_pos == std::string::npos) return st->encode("");
    // After the zip, skip whitespace/punctuation and read the next word.
    std::size_t p = zip_pos + 5;
    while (p < after_marker.size() && (after_marker[p] == ' '
           || after_marker[p] == ',' || after_marker[p] == '*'))
        ++p;
    if (p >= after_marker.size()) return st->encode("");
    // Read the country name (up to next punctuation or end of line).
    std::size_t cstart = p;
    while (p < after_marker.size() && after_marker[p] != '*'
           && after_marker[p] != '\n' && after_marker[p] != '.'
           && after_marker[p] != ',' && after_marker[p] != ')')
        ++p;
    std::string country = after_marker.substr(cstart, p - cstart);
    while (!country.empty() && country.back() == ' ')
        country.pop_back();
    if (country.empty()) return st->encode("");
    return st->encode(country);
}

// Returns 1 if the text contains ambiguous payment selection
// evidence: conflicting card endings, conflicting stated amounts,
// or malformed digit sequences in "ending in" / "ends in" patterns.
// Checks BOTH card AND amount consistency across ALL recognized
// payment-selection shapes. Used on both explicit-selection and
// fallback paths of ApprovedCardEnding.
souffle::RamDomain user_card_selection_ambiguous(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    std::string text = to_lower(st->decode(text_sym));
    // Check structured payment-selection shapes for card+amount consistency.
    auto selections = parse_payment_selections(text);
    auto s = summarize_selections(selections);
    if (s.ambiguous) return 1;
    // Also check raw "ending in"/"ends in" patterns for malformed
    // digit sequences that don't match any fullmatch shape.
    static const char* kEnding[] = {
        "ending in ", "ends in ", nullptr
    };
    for (const char** pat = kEnding; *pat; ++pat) {
        std::size_t pos = 0;
        while ((pos = text.find(*pat, pos)) != std::string::npos) {
            std::size_t start = pos + std::strlen(*pat);
            while (start < text.size() && (text[start] == '*'
                   || text[start] == ' ')) ++start;
            std::string digits;
            while (start < text.size() &&
                   std::isdigit(static_cast<unsigned char>(text[start])))
                digits += text[start++];
            if (!digits.empty() && digits.size() != 4) return 1;
            pos = start;
        }
    }
    return 0;
}

// Extract the dollar amount (in whole cents) from the proposal's
// payment-selection question or statement, using the same fullmatch
// shapes and validated amount parser as proposal_payment_card_ending.
// Returns -1 if no recognized payment-selection clause exists.
souffle::RamDomain proposal_payment_question_cents(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    std::string text = to_lower(st->decode(text_sym));
    auto s = summarize_selections(parse_payment_selections(text));
    return s.cents;
}

souffle::RamDomain proposal_has_action_approval_request(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    std::string raw = to_lower(st->decode(text_sym));
    std::string low;
    low.reserve(raw.size());
    for (char c : raw) { if (c != '*') low += c; }

    const char* kHeader = "to proceed, please confirm";
    std::size_t hpos = low.find(kHeader);
    if (hpos == std::string::npos) return 0;
    std::size_t cur = hpos + std::strlen(kHeader);
    if (cur < low.size() && low[cur] == ':') ++cur;

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
    static const char* kFullQs[] = {
        "do you reply \"yes\" to confirm both the address change "
            "and the item modification?",
        "do you reply \"yes\" to confirm both the address "
            "and the item modification?",
        "do you reply \"yes\" to confirm both the address change "
            "and the item change?",
        nullptr
    };
    bool matched = false;
    for (const char** q = kFullQs; *q; ++q) {
        std::size_t qlen = std::strlen(*q);
        if (cur + qlen <= low.size()
            && low.compare(cur, qlen, *q) == 0)
            { cur += qlen; matched = true; break; }
    }
    if (!matched) return 0;

    skip();
    const char* kCard = "if yes, would you like to use";
    std::size_t cplen = std::strlen(kCard);
    if (cur + cplen <= low.size()
        && low.compare(cur, cplen, kCard) == 0) {
        std::size_t qm = cur;
        while (qm < low.size() && low[qm] != '?') ++qm;
        if (qm >= low.size()) return 0;
        std::string clause(low, cur, qm - cur);
        std::string card; int cents = -1;
        if (!try_card_amount_shape(clause, '?', card, cents))
            return 0;
        cur = qm + 1;
    }

    while (cur < low.size()) {
        char c = low[cur];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
            { ++cur; continue; }
        return 0;
    }
    return 1;
}

}  // extern "C"
