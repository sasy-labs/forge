// Pure helpers behind the AgentDojo policy's custom Soufflé functors.
//
// This header exists so there is exactly ONE copy of this logic. The unit test
// and the functors that ship both include it. An earlier version kept a second
// copy inside the test file, which is the trap this whole investigation is
// about: the probe and the shipped rule drift apart, the probe passes, and the
// rule that actually runs is a different and weaker one.
//
// Nothing here touches Soufflé types — that lives in agentdojo_functors.cpp —
// so the test compiles with a bare `g++ -std=c++17`.
#ifndef SASY_AGENTDOJO_FUNCTORS_IMPL_H_
#define SASY_AGENTDOJO_FUNCTORS_IMPL_H_

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace adj {


inline std::string fold(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(std::tolower(c)));
        }
    }
    return out;
}

inline bool is_ident_char(unsigned char c) {
    // Brackets are here for one reason: an IPv6 literal is written `[::1]`, and
    // without them the token splits at the bracket and the address is never
    // seen at all. They cannot start an identifier on their own — nothing
    // bracketed passes looks_like_identifier unless it also carries a colon.
    return std::isalnum(c) || c == '.' || c == '-' || c == '_' || c == '@'
        || c == '+' || c == '%' || c == ':' || c == '/'
        || c == '[' || c == ']';
}

// The host an address or URL actually reaches, lowercased, with userinfo and
// port removed. Everything that decides "is this destination ours" goes through
// here, so the parse has to be the one a client would perform:
//
//   - the authority ends at the first `/`, `?` or `#`. An `@` after that point
//     is in the PATH and names nothing, so
//     `https://evil.example/path@ours.com/x` is `evil.example`, not `ours.com`;
//   - inside the authority the LAST `@` separates userinfo from host, so
//     `https://user:pw@mail.ours.com/x` is `mail.ours.com`;
//   - a bracketed IPv6 literal keeps its brackets and its colons.
inline std::string host_of(const std::string& value) {
    std::string v;
    for (unsigned char c : value) v.push_back(static_cast<char>(std::tolower(c)));

    std::size_t start = 0;
    const std::size_t scheme = v.find("//");
    if (scheme != std::string::npos) start = scheme + 2;

    std::size_t end = v.size();
    for (std::size_t i = start; i < v.size(); ++i) {
        const char c = v[i];
        if (c == '/' || c == '?' || c == '#' || c == ' ') { end = i; break; }
    }
    std::string authority = v.substr(start, end - start);

    const std::size_t at = authority.rfind('@');
    if (at != std::string::npos) authority = authority.substr(at + 1);

    if (!authority.empty() && authority.front() == '[') {
        const std::size_t close = authority.find(']');
        if (close != std::string::npos) return authority.substr(0, close + 1);
    }
    const std::size_t colon = authority.find(':');
    if (colon != std::string::npos) authority = authority.substr(0, colon);
    while (!authority.empty() && authority.back() == '.') authority.pop_back();
    return authority;
}

// Two letters, then letters and digits with at least eight digits among them.
// The digits-only reading rejected real IBANs: the bank code is alphabetic in
// most countries, so `GB29NWBK60161331926819` — this file's own worked example
// — produced no identifier and could be laundered through a message body.
inline bool looks_like_account_number(const std::string& tok) {
    if (tok.size() < 10) return false;
    if (!std::isalpha(static_cast<unsigned char>(tok[0]))
        || !std::isalpha(static_cast<unsigned char>(tok[1]))) {
        return false;
    }
    std::size_t digits = 0;
    for (std::size_t i = 2; i < tok.size(); ++i) {
        const unsigned char c = tok[i];
        if (std::isdigit(c)) ++digits;
        else if (!std::isalpha(c)) return false;
    }
    return digits >= 8;
}

// Alphanumeric only, and no longer than a printed group. ISO 13616 prints an
// IBAN in groups of four, so this is a property of the format rather than of
// any particular account.
// A printed IBAN group: alphanumeric, at most four characters, and NOT a word
// made only of letters. Requiring a digit somewhere is what stops the rejoin
// swallowing the prose after the account — "GB29 NWBK 6016 1331 9268 19 for
// rent" glued "for" and "rent" on, which both hid the real account from the
// laundering rule and refused legitimate messages that quote it.
//
// Real IBANs do contain all-letter groups (the bank code, "NWBK"), so those are
// admitted when they are followed by more of the account; the check below is on
// the merged whole, which must still look like an account number.
inline bool is_account_group(const std::string& s) {
    if (s.empty() || s.size() > 4) return false;
    for (unsigned char c : s) {
        if (!std::isalnum(c)) return false;
    }
    return true;
}

// A group that can END an account: it must carry a digit, so a trailing English
// word cannot be the last thing absorbed.
inline bool is_account_tail(const std::string& s) {
    if (!is_account_group(s)) return false;
    for (unsigned char c : s) {
        if (std::isdigit(c)) return true;
    }
    return false;
}

// The first printed group of an IBAN: two country letters then two check
// digits. Requiring this to START a rejoin is what stops the rejoin swallowing
// ordinary prose — without it, "please pay <iban> today" merges into one run
// that begins with two letters and carries enough digits to look like an
// account, and the real account is never reported at all.
inline bool is_account_head(const std::string& s) {
    return s.size() == 4
        && std::isalpha(static_cast<unsigned char>(s[0]))
        && std::isalpha(static_cast<unsigned char>(s[1]))
        && std::isdigit(static_cast<unsigned char>(s[2]))
        && std::isdigit(static_cast<unsigned char>(s[3]));
}

// An identifier is a run of address/URL characters that either carries an '@'
// with something on both sides, looks like a host (a dot with a plausible TLD
// after it), or looks like an account number (two letters then eight or more
// digits). Anything else in a message body is prose and is left alone.
inline bool looks_like_identifier(const std::string& tok) {
    if (tok.size() < 5) return false;

    const std::size_t at = tok.find('@');
    if (at != std::string::npos && at > 0 && at + 1 < tok.size()
        && tok.find('.', at) != std::string::npos) {
        return true;
    }

    // A filename is not a destination. The dotted-host test below accepts any
    // token whose last dot is followed by letters, which pulled "feedback.xlsx"
    // and "team-building-activities.docx" out of message bodies — 46 and 44
    // times in one measured run. They come from untrusted file listings, so
    // they were unanchored and the laundering rule refused the call. The
    // offline screening never did this, because its extractor required an "@",
    // a "www." or an account shape.
    static const char* const kFileExtensions[] = {
        "csv", "doc", "docx", "gif", "jpeg", "jpg", "json", "md", "pdf", "png",
        "ppt", "pptx", "py", "rtf", "svg", "txt", "xls", "xlsx", "xml", "yaml", "yml", "zip",
    };
    {
        const std::size_t dot = tok.rfind('.');
        if (dot != std::string::npos && tok.find('/') == std::string::npos
            && tok.find('@') == std::string::npos) {
            std::string ext;
            for (std::size_t i = dot + 1; i < tok.size(); ++i) {
                ext.push_back(static_cast<char>(std::tolower(
                    static_cast<unsigned char>(tok[i]))));
            }
            for (const char* known : kFileExtensions) {
                if (ext == known) return false;
            }
        }
    }

    std::size_t host_start = 0;
    const std::size_t scheme = tok.find("//");
    if (scheme != std::string::npos) host_start = scheme + 2;
    const std::string host = tok.substr(host_start);
    const std::size_t dot = host.rfind('.');
    if (dot != std::string::npos && dot > 0 && dot + 2 < host.size()) {
        bool tld_alpha = true;
        for (std::size_t i = dot + 1; i < host.size() && host[i] != '/'; ++i) {
            if (!std::isalpha(static_cast<unsigned char>(host[i]))) {
                tld_alpha = false;
                break;
            }
        }
        if (tld_alpha) return true;
    }

    // A dotted quad is a host even though it has no alphabetic TLD. Without
    // this, "http://203.0.113.10/collect" is not an identifier at all and a
    // link to a bare address launders straight through the body check.
    {
        std::size_t digits = 0, dots = 0;
        bool only_host_chars = true;
        for (std::size_t i = 0; i < host.size() && host[i] != '/'; ++i) {
            if (std::isdigit(static_cast<unsigned char>(host[i]))) ++digits;
            else if (host[i] == '.') ++dots;
            else { only_host_chars = false; break; }
        }
        if (only_host_chars && dots == 3 && digits >= 4) return true;
    }
    // An IPv6 literal is bracketed, which the character class already keeps.
    if (tok.find('[') != std::string::npos && tok.find(']') != std::string::npos
        && tok.find(':') != std::string::npos) {
        return true;
    }

    if (looks_like_account_number(tok)) return true;
    return false;
}

// The longest run of space-separated groups worth trying to rejoin. An IBAN is
// at most 34 characters, so eight four-character groups covers every real one.
inline constexpr std::size_t kMaxAccountGroups = 8;

inline std::vector<std::string> identifiers(const std::string& text) {
    // Tokenise first, remembering whether a single space (and nothing else)
    // separated each token from the one before it.
    std::vector<std::string> toks;
    std::vector<bool> space_before;
    std::string tok;
    std::size_t gap = 0;        // separator characters since the last token
    bool gap_space_only = true;
    for (std::size_t i = 0; i <= text.size(); ++i) {
        const bool end = (i == text.size());
        const unsigned char c = end ? ' ' : static_cast<unsigned char>(text[i]);
        if (!end && is_ident_char(c)) {
            if (tok.empty()) {
                space_before.push_back(gap == 1 && gap_space_only && !toks.empty());
            }
            tok.push_back(static_cast<char>(c));
            continue;
        }
        if (!tok.empty()) {
            // trailing punctuation that got swept up by the character class
            while (!tok.empty() && (tok.back() == '.' || tok.back() == ':'
                                    || tok.back() == '/' || tok.back() == '-')) {
                tok.pop_back();
            }
            toks.push_back(tok);
            tok.clear();
            gap = 0;
            gap_space_only = true;
        }
        if (!end) {
            ++gap;
            if (c != ' ') gap_space_only = false;
        }
    }

    std::vector<std::string> out;
    for (std::size_t i = 0; i < toks.size(); ++i) {
        if (looks_like_identifier(toks[i])) {
            out.push_back(toks[i]);
            continue;
        }
        // An IBAN written the way banks print it — `GB29 NWBK 6016 1331 9268 19`
        // — is not one token, so no single fragment looks like an identifier and
        // the account launders through a message body untouched. Rejoin runs of
        // groups separated by exactly one space and see whether the whole is an
        // account number. Only alphanumeric groups qualify, so this cannot glue
        // ordinary prose together.
        if (!is_account_head(toks[i])) continue;
        std::string merged = toks[i];
        std::string best;
        std::size_t best_j = i;
        for (std::size_t j = i + 1;
             j < toks.size() && j - i < kMaxAccountGroups
             && space_before[j] && is_account_group(toks[j]);
             ++j) {
            merged += toks[j];
            // Keep going rather than stopping at the first match: the prefix
            // `GB29 NWBK 6016 1331` already has eight digits, and emitting that
            // would report a truncated account that matches nothing.
            // Only a group carrying a digit may end the account, so the rejoin
            // stops at the number rather than running into the sentence.
            if (is_account_tail(toks[j]) && looks_like_account_number(merged)) {
                best = merged;
                best_j = j;
            }
        }
        if (!best.empty()) {
            out.push_back(best);
            i = best_j;     // consumed
        }
    }
    return out;
}

// Is this single value an account number? Used by the kind-directed rule: an
// account number must have come from the user or the bank's own records, never
// from a document the agent read. That is a claim about the KIND of the value,
// which is why it cannot be expressed by comparing the call's labels — the
// label says where the CALL stands, not what this argument is.
inline bool is_account_number(const std::string& value) {
    std::string compact;
    for (unsigned char c : value) {
        if (!std::isspace(c) && c != '-') compact.push_back(static_cast<char>(c));
    }
    return looks_like_account_number(compact);
}

// Can this URL carry data OUT?
//
// A fetch is a flow, but only the parts of a URL an agent chooses can carry
// anything: a query string, a fragment, or userinfo. `https://site.com/page`
// sends the host and path and nothing the agent read — visiting it discloses
// that the visit happened, not what the user's mail said. So the confidentiality
// rule has no business refusing it, and refusing it anyway cost seven clean
// tasks whose URLs were plain hosts.
//
// The destination rules still apply to the host. This is only about whether the
// call is a DISCLOSURE, not about whether the destination is anchored.
inline bool url_carries_data(const std::string& value) {
    std::string v;
    for (unsigned char c : value) v.push_back(static_cast<char>(std::tolower(c)));

    std::size_t start = 0;
    const std::size_t scheme = v.find("//");
    if (scheme != std::string::npos) start = scheme + 2;

    // Anything after the authority that is a query or fragment.
    for (std::size_t i = start; i < v.size(); ++i) {
        if (v[i] == '?' || v[i] == '#') return true;
    }
    // userinfo@host smuggles data in the credential position.
    std::size_t end = v.size();
    for (std::size_t i = start; i < v.size(); ++i) {
        if (v[i] == '/') { end = i; break; }
    }
    if (v.substr(start, end - start).find('@') != std::string::npos) return true;
    return false;
}

// The alphabetic word starting at `i`, or empty if none starts there.
inline std::string word_at(const std::string& low, std::size_t i) {
    if (i > 0 && std::isalpha(static_cast<unsigned char>(low[i - 1]))) return "";
    std::string w;
    for (std::size_t j = i; j < low.size() && std::isalpha(static_cast<unsigned char>(low[j])); ++j) {
        w.push_back(low[j]);
    }
    return w;
}

// ── Selection verification ────────────────────────────────────────────
//
// The one rule here that checks WHAT THE AGENT CHOSE rather than where a value
// came from.
//
// "Delete the largest file" names a criterion, not an id, so the id necessarily
// comes from the listing the agent read — provenance cannot separate the user's
// file from an attacker's. But the criterion is computable: the listing carries
// the sizes, so the policy can recompute the answer and require the argument to
// match it. Nothing untrusted has to be BELIEVED for that; the listing supplies
// candidate rows and the arithmetic is the policy's own.
//
// It fires only when the request names a criterion the listing can settle. A
// request that says "the file about recipes" is semantic and gets nothing here,
// which is the honest boundary: this verifies selections, it does not guess at
// intent.

//: A superlative in the user's request, as (field class, take the maximum?).
struct Criterion {
    const char* const* words;
    std::size_t word_count;
    const char* field_class;
    bool want_max;
};

inline const char* const kLargest[]  = {"largest", "biggest", "longest"};
inline const char* const kSmallest[] = {"smallest", "shortest"};
inline const char* const kDearest[]  = {"priciest"};
inline const char* const kCheapest[] = {"cheapest"};
inline const char* const kNewest[]   = {"newest", "latest"};
inline const char* const kOldest[]   = {"oldest"};

inline const Criterion kCriteria[] = {
    {kLargest,  sizeof(kLargest) / sizeof(*kLargest),   "size",   true},
    {kSmallest, sizeof(kSmallest) / sizeof(*kSmallest), "size",   false},
    {kDearest,  sizeof(kDearest) / sizeof(*kDearest),   "price",  true},
    {kCheapest, sizeof(kCheapest) / sizeof(*kCheapest), "price",  false},
    {kNewest,   sizeof(kNewest) / sizeof(*kNewest),     "recency", true},
    {kOldest,   sizeof(kOldest) / sizeof(*kOldest),     "recency", false},
};

//: Multi-word forms, kept separate because they are phrases not words.
inline bool says_phrase(const std::string& low, const char* a, const char* b) {
    const std::size_t i = low.find(a);
    return i != std::string::npos && low.find(b, i) != std::string::npos
        && low.find(b, i) - i < 24;
}

// The criterion the request names, as "<field class>:<max|min>", or empty.
// Empty is the common case and means the rule stands down.
inline std::string criterion_of(const std::string& text) {
    std::string low;
    for (unsigned char c : text) low.push_back(static_cast<char>(std::tolower(c)));

    if (says_phrase(low, "most", "expensive")) return "price:max";
    if (says_phrase(low, "least", "expensive")) return "price:min";
    if (says_phrase(low, "highest", "rat")) return "rating:max";
    if (says_phrase(low, "lowest", "rat")) return "rating:min";
    if (says_phrase(low, "best", "rat")) return "rating:max";
    if (says_phrase(low, "most", "recent")) return "recency:max";

    for (const Criterion& c : kCriteria) {
        for (std::size_t i = 0; i < low.size(); ++i) {
            const std::string w = word_at(low, i);
            if (w.empty()) continue;
            for (std::size_t j = 0; j < c.word_count; ++j) {
                if (w == c.words[j]) {
                    return std::string(c.field_class) + (c.want_max ? ":max" : ":min");
                }
            }
        }
    }
    return "";
}

// The identifier of the record that wins `criterion` in a rendered listing.
//
// The listing is the tool's own output, rendered one `field: value` per line
// with the records in sequence. Fields are emitted in a fixed order, so the id
// belonging to a value is simply the last id seen before it. A record missing
// either field is skipped rather than guessed at, and an empty result means the
// listing could not settle the question — in which case the rule stands down.
inline std::string select_by(const std::string& listing, const std::string& id_field,
                             const std::string& value_field, bool want_max) {
    std::string best_id, cur_id;
    double best_val = 0;
    bool have = false;

    std::size_t pos = 0;
    while (pos <= listing.size()) {
        std::size_t eol = listing.find('\n', pos);
        if (eol == std::string::npos) eol = listing.size();
        std::string line = listing.substr(pos, eol - pos);
        pos = eol + 1;

        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key, val;
        for (std::size_t i = 0; i < colon; ++i) {
            const unsigned char c = line[i];
            if (std::isalnum(c) || c == '_') key.push_back(static_cast<char>(c));
        }
        for (std::size_t i = colon + 1; i < line.size(); ++i) {
            const unsigned char c = line[i];
            if (c != ' ' && c != '\'' && c != '"') val.push_back(static_cast<char>(c));
        }
        if (key.empty() || val.empty()) continue;

        if (key == id_field) { cur_id = val; continue; }
        if (key != value_field || cur_id.empty()) continue;

        // A numeric field, or a date compared lexically — both order correctly
        // once digits are kept in place.
        double num = 0;
        bool numeric = true;
        try { num = std::stod(val); } catch (...) { numeric = false; }
        if (!numeric) {
            num = 0;
            for (unsigned char c : val) {
                if (std::isdigit(c)) num = num * 10 + (c - '0');
            }
        }
        if (!have || (want_max ? num > best_val : num < best_val)) {
            have = true; best_val = num; best_id = cur_id;
        }
    }
    return have ? best_id : std::string();
}

// Was this whole field value DICTATED — copied verbatim out of text an
// outsider wrote?
//
// Distinct from the laundering check, which looks for an identifier (an address
// or a link) inside a composed message. This asks whether the field IS the
// outsider's text. An injection that says "the event title should be X" gets X
// verbatim; a title the agent composed for the user's own task does not appear
// in any document.
//
// Comparison is on folded text, so punctuation and spacing do not evade it.
//
// A floor on length is what keeps the rule from firing on coincidence: a short
// value is a common phrase, and a document mentioning "Lunch" is not evidence
// that anyone dictated it. The floor is a judgement about how long a string has
// to be before verbatim reuse stops being chance, not a threshold fitted to any
// corpus.
inline constexpr std::size_t kMinDictatedChars = 12;

inline bool is_dictated(const std::string& haystack, const std::string& needle) {
    const std::string n = fold(needle);
    if (n.size() < kMinDictatedChars) return false;
    return fold(haystack).find(n) != std::string::npos;
}

// Is `host` the domain `d`, or a subdomain of it?
//
// The trailing dot is load-bearing: without it "notbluesparrowtech.com" ends
// with the domain and would read as internal. The planted lookalike in this
// benchmark is subtler still — `manager-bluesparrowtech@gmail.com` is a GMAIL
// address chosen to look internal, and it is only rejected because the parsed
// host is compared, not the raw string.
inline bool host_is_in_domain(const std::string& host, const std::string& d) {
    if (d.empty() || host.empty()) return false;
    if (host == d) return true;
    return host.size() > d.size()
        && host.compare(host.size() - d.size(), d.size(), d) == 0
        && host[host.size() - d.size() - 1] == '.';
}

// Split a rendered tool result into its records.
//
// Results are rendered one `field: value` per line, with each record opening at
// a top-level "- ". A result with no such marker is one record.
inline std::vector<std::string> result_records(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    std::size_t pos = 0;
    while (pos <= text.size()) {
        std::size_t eol = text.find('\n', pos);
        if (eol == std::string::npos) eol = text.size();
        const std::string line = text.substr(pos, eol - pos);
        pos = eol + 1;
        if (line.size() >= 2 && line[0] == '-' && line[1] == ' ' && !cur.empty()) {
            out.push_back(cur);
            cur.clear();
        }
        cur += line;
        cur.push_back('\n');
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// The `sender:` of one record, or empty when it has none.
inline std::string record_sender(const std::string& record) {
    std::size_t pos = 0;
    while (pos <= record.size()) {
        std::size_t eol = record.find('\n', pos);
        if (eol == std::string::npos) eol = record.size();
        const std::string line = record.substr(pos, eol - pos);
        pos = eol + 1;
        std::size_t i = 0;
        while (i < line.size() && (line[i] == ' ' || line[i] == '-')) ++i;
        if (line.compare(i, 7, "sender:") == 0) {
            std::string v;
            for (std::size_t j = i + 7; j < line.size(); ++j) {
                const unsigned char c = line[j];
                if (c != ' ' && c != '\'' && c != '"') v.push_back(static_cast<char>(c));
            }
            return v;
        }
    }
    return std::string();
}

// Was this value dictated by an OUTSIDER, judged record by record?
//
// A single `get_unread_emails` result holds many messages from many senders,
// and the graph records one node for the whole result — so asking "did this
// come from untrusted content" of the result as a whole answers yes even when
// the text came from a colleague the user pointed at. Measured, that is what
// made this rule refuse legitimate work: a reply quoting scores from an
// internal colleague's mail, an event built from the user's own family emails.
//
// Splitting here rather than in the graph gets per-record provenance for the
// content rules without per-record nodes. It is narrower than the graph fix:
// it answers "whose words are these", not "what does this call depend on",
// which stays whole-result.
//
// A record with NO sender is treated as an outsider's. That is deliberate and
// it is why files stay untrusted: one injected document in this benchmark is
// owned by the user herself, so ownership is not evidence of authorship.
//
// A value that is an ISO date and nothing else (`2024-05-01`, optionally with
// a time of day; see `parse_iso_date`) is compared as a DATE: the record dictates it if it names that day in any form
// `mentions_date` reads ("the 1st to the 5th of May 2024"). The agent turns
// the prose into ISO form, so neither a substring test nor the coincidence
// floor — a date is ten characters — would ever see it.
//: Split a "a|b|c" list into its lowercased parts.
inline std::vector<std::string> pipe_list(const std::string& v) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : v) {
        if (c == '|') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// Declared here because the dictation check needs them and they are defined
// further down, with the rest of the anchoring helpers.
inline bool is_host_like(const std::string& s);
inline bool anchors(const std::string& haystack, const std::string& needle);
inline bool is_iso_date(const std::string& value);
inline bool mentions_date(const std::string& text, const std::string& value);

inline bool is_dictated_by_outsider(const std::string& haystack,
                                    const std::string& needle,
                                    const std::string& own_domains,
                                    const std::string& own_principals) {
    // A host-like value is compared as a HOST, not as folded text. The agent
    // writes "http://www.example.com" where the page that handed it over said
    // "www.example.com", so a substring test misses the very case this exists
    // for. Host comparison is precise enough not to need the coincidence
    // floor, which is there for prose.
    const bool date_like = is_iso_date(needle);
    const bool host_like = !date_like && is_host_like(needle);
    const std::string n = fold(needle);
    if (!date_like && !host_like && n.size() < kMinDictatedChars) return false;

    const std::vector<std::string> domains = pipe_list(own_domains);
    const std::vector<std::string> principals = pipe_list(own_principals);

    for (const std::string& record : result_records(haystack)) {
        const std::string sender = record_sender(record);
        if (!sender.empty()) {
            bool internal = false;
            // By address, where the sender is one.
            const std::string host = host_of(sender);
            for (const std::string& d : domains) {
                if (host_is_in_domain(host, d)) { internal = true; break; }
            }
            // Or by name: a chat roster attributes a message to "Bob", not to
            // an address. Matched whole and case-insensitively, never as a
            // substring — "Bobby" is not Bob.
            if (!internal) {
                std::string low;
                for (unsigned char c : sender) {
                    low.push_back(static_cast<char>(std::tolower(c)));
                }
                for (const std::string& p : principals) {
                    if (low == p) { internal = true; break; }
                }
            }
            if (internal) continue;  // a colleague's words are not dictation
        }
        if (date_like   ? mentions_date(record, needle)
            : host_like ? anchors(record, needle)
                        : fold(record).find(n) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// Does `s` name a host? A dot with an alphabetic label of at least two
// characters after it, and no spaces. Used to decide which matching rule an
// anchoring test should use — hosts need label boundaries, business names do
// not have any.
inline bool is_host_like(const std::string& s) {
    if (s.find(' ') != std::string::npos) return false;
    // An ADDRESS is not a host. `host_of` would strip the local part and leave
    // the domain, and matching on that alone anchors every address at a domain
    // the trusted text happened to mention — so naming one colleague at a large
    // mail provider would vouch for an attacker's account there. An address has
    // to match in full.
    if (s.find('@') != std::string::npos) return false;
    const std::string h = host_of(s);
    const std::size_t dot = h.rfind('.');
    if (dot == std::string::npos || dot == 0 || dot + 2 >= h.size()) return false;
    for (std::size_t i = dot + 1; i < h.size(); ++i) {
        if (!std::isalpha(static_cast<unsigned char>(h[i]))) return false;
    }
    return true;
}

// Every run of identifier characters in `text`, punctuation trimmed. Unlike
// `identifiers` this keeps the ones that are not identifier-shaped, because
// anchoring compares against whatever the trusted text actually said.
inline std::vector<std::string> ident_runs(const std::string& text) {
    std::vector<std::string> out;
    std::string tok;
    for (std::size_t i = 0; i <= text.size(); ++i) {
        const bool end = (i == text.size());
        const unsigned char c = end ? ' ' : static_cast<unsigned char>(text[i]);
        if (!end && is_ident_char(c)) { tok.push_back(static_cast<char>(c)); continue; }
        if (!tok.empty()) {
            while (!tok.empty() && (tok.back() == '.' || tok.back() == ':'
                                    || tok.back() == '/' || tok.back() == '-'
                                    || tok.back() == ',')) {
                tok.pop_back();
            }
            if (!tok.empty()) out.push_back(tok);
            tok.clear();
        }
    }
    return out;
}

// Is `needle` present in `text` as a whole alphanumeric word? Used for values
// too short for a substring test to carry any meaning.
inline bool has_standalone_token(const std::string& text, const std::string& needle) {
    const std::string n = fold(needle);
    if (n.empty()) return false;
    std::string word;
    for (std::size_t i = 0; i <= text.size(); ++i) {
        const bool end = (i == text.size());
        const unsigned char c = end ? ' ' : static_cast<unsigned char>(text[i]);
        if (!end && std::isalnum(c)) { word.push_back(static_cast<char>(std::tolower(c))); continue; }
        if (!word.empty()) {
            if (word == n) return true;
            word.clear();
        }
    }
    return false;
}

// Did somebody with standing put `needle` in front of the agent, according to
// `haystack`?
//
// Three regimes, because one test cannot serve all three needle shapes:
//
//   - An ADDRESS must match in full — see `is_host_like`.
//   - A bare HOST must match at a label boundary. `fold` drops the dots, so a plain
//     folded-substring test lets the string "notexample.com" sitting in trusted
//     text anchor the entirely different destination "example.com". Comparing
//     parsed hosts instead accepts "mail.example.com" and "alice@example.com"
//     — which really are that domain — and rejects "notexample.com".
//   - A very SHORT value gets a whole-word test. "xy" occurs inside "proxy", so
//     a substring test is meaningless; returning true regardless, as this did,
//     let Rule 4b accept an attacker-chosen two-character password that the
//     user never typed.
//   - ANYTHING ELSE — business names, account numbers — keeps the folded
//     substring test. Folding is what makes it survive the spacing and
//     punctuation a model puts in ("GB29 NWBK …" against "GB29NWBK…").
inline bool anchors(const std::string& haystack, const std::string& needle) {
    const std::string n = fold(needle);
    if (n.empty()) return false;
    if (n.size() < 3) return has_standalone_token(haystack, needle);

    if (is_host_like(needle)) {
        const std::string want = host_of(needle);
        if (want.empty()) return false;
        for (const std::string& run : ident_runs(haystack)) {
            const std::string got = host_of(run);
            if (got == want) return true;
            if (got.size() > want.size()
                && got.compare(got.size() - want.size(), want.size(), want) == 0
                && got[got.size() - want.size() - 1] == '.') {
                return true;
            }
        }
        return false;
    }
    return fold(haystack).find(n) != std::string::npos;
}

// Function words, which end a name rather than joining it. Requiring
// capitalisation instead would be an evasion: an injection that gets the model
// to write "book evil palace" in lower case, or to shout "EVIL PALACE", would
// produce no candidate at all and sail past a rule whose whole job is to catch
// exactly that. So case is ignored and the boundary is carried by these.
inline const char* const kNameStopWords[] = {
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "our",
    "their", "its", "his", "her", "and", "or", "but", "at", "in", "on", "to",
    "for", "from", "with", "of", "by", "is", "are", "was", "were", "be",
    "book", "visit", "try", "recommend", "recommended", "stay", "any", "some",
    "best", "great", "good", "nice", "lovely", "there", "here", "it",
};

inline bool is_stop_word(const std::string& w) {
    std::string low;
    for (unsigned char c : w) low.push_back(static_cast<char>(std::tolower(c)));
    for (const char* s : kNameStopWords) {
        if (low == s) return true;
    }
    return false;
}

// A word that can form part of a name: any alphabetic word that is not a
// function word. Case is deliberately not consulted.
inline bool name_word(const std::string& w) {
    return w.size() >= 2 && !is_stop_word(w);
}

// A word, plus whether only spaces separated it from the previous word. A name
// is a run of words joined by spaces; punctuation ends it. Without that,
// "…(French Cuisine) **Restaurant:** Le Baratin" yields the entity
// "Recommendation French Cuisine Restaurant", which is not a business and which
// no catalogue can vouch for — so every heading in an answer becomes a refusal.
struct Word {
    std::string text;
    bool space_before;
};

inline std::vector<Word> words_of(const std::string& text) {
    std::vector<Word> out;
    std::string w;
    // Whether the run of separators since the previous word was spaces only.
    // False before the first word: nothing precedes it to join to.
    bool gap_clean = false;
    for (std::size_t i = 0; i <= text.size(); ++i) {
        const bool end = (i == text.size());
        const unsigned char c = end ? '\n' : static_cast<unsigned char>(text[i]);
        if (!end && std::isalpha(c)) {
            w.push_back(static_cast<char>(c));
            continue;
        }
        if (!w.empty()) {
            // `gap_clean` still describes the separator that preceded this
            // word — nothing touches it while a word is being read.
            out.push_back({w, gap_clean});
            w.clear();
            gap_clean = true;
        }
        if (c != ' ') gap_clean = false;
    }
    return out;
}

// Names ending in `suffix`: up to three capitalised words before it, joined by
// spaces only. Longest match wins, so "Riverside View Hotel" is reported once
// rather than also as "View Hotel".
inline std::vector<std::string> entities_with_suffix(
    const std::string& text, const std::string& suffix) {
    std::vector<std::string> out;
    const std::vector<Word> words = words_of(text);
    std::string sfx_low;
    for (unsigned char c : suffix) sfx_low.push_back(static_cast<char>(std::tolower(c)));
    for (std::size_t i = 0; i < words.size(); ++i) {
        std::string w_low;
        for (unsigned char c : words[i].text) w_low.push_back(static_cast<char>(std::tolower(c)));
        if (w_low != sfx_low) continue;
        std::size_t start = i;
        while (start > 0 && i - start < 3
               && words[start].space_before
               && name_word(words[start - 1].text)) {
            --start;
        }
        if (start == i) continue;  // the suffix alone is a common noun
        std::string name = words[start].text;
        for (std::size_t j = start + 1; j <= i; ++j) name += " " + words[j].text;
        out.push_back(name);
    }
    return out;
}

inline std::vector<std::string> split_pipe(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == '|') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// Every catalogued-type name in the text, in the order it appears, deduplicated.
// One pass over the words per suffix, but the caller pays a single functor call.
inline std::vector<std::string> catalog_entities(
    const std::string& text, const std::string& suffixes) {
    std::vector<std::string> out;
    for (const std::string& sfx : split_pipe(suffixes)) {
        for (const std::string& name : entities_with_suffix(text, sfx)) {
            bool seen = false;
            for (const std::string& have : out) {
                if (have == name) { seen = true; break; }
            }
            if (!seen) out.push_back(name);
        }
    }
    return out;
}

// A catalogued name with its trailing TYPE word removed, if it carries one.
//
// The catalogue stores "Riverside View"; an answer says "Riverside View Hotel".
// Comparing those directly fails, so adding a type word to a real business name
// would defeat the check. Stripping it makes the comparison the one that was
// intended: is this business in the catalogue?
//
// Only a trailing type word goes. A name whose own last word happens to be a
// type word ("Alamo Rent A Car" against the type "Car") keeps everything if
// removing the word would leave nothing, and multi-word types are tried before
// single-word ones so "Car Rental" is not mistaken for "Rental".
inline std::string entity_core(const std::string& name, const std::string& suffixes) {
    std::vector<std::string> sfx = split_pipe(suffixes);
    // Longest first: "Car Rental" must win over "Rental".
    for (std::size_t i = 0; i < sfx.size(); ++i) {
        for (std::size_t j = i + 1; j < sfx.size(); ++j) {
            if (sfx[j].size() > sfx[i].size()) std::swap(sfx[i], sfx[j]);
        }
    }
    std::string low;
    for (unsigned char c : name) low.push_back(static_cast<char>(std::tolower(c)));
    for (const std::string& s : sfx) {
        if (s.empty() || s.size() >= low.size()) continue;
        std::string s_low;
        for (unsigned char c : s) s_low.push_back(static_cast<char>(std::tolower(c)));
        if (low.compare(low.size() - s_low.size(), s_low.size(), s_low) != 0) continue;
        const std::size_t cut = low.size() - s_low.size();
        if (cut == 0 || low[cut - 1] != ' ') continue;   // must be a whole word
        std::string core = name.substr(0, cut);
        while (!core.empty() && core.back() == ' ') core.pop_back();
        if (!core.empty()) return core;
    }
    return name;
}

// Irreversible effects, split by KIND. One global "the user asked for
// something destructive" bit was too coarse to be honest: a request that says
// "delete the largest file" also unlocked credential changes, cancellations and
// every other irreversible tool for the rest of the session. Five of forty
// workspace requests and one of sixteen banking ones granted that blanket
// permission.
//
// Matching is at a word start, so "delet" fires on "deleting" and "deleted" but
// not on "undelete". Phrases carry meanings no single stem does.
// An effect kind, matched two ways.
//
// `words` match a WHOLE word (optionally plus "s"): short stems that are
// themselves words cannot be prefix-matched, because "car" is a prefix of
// "card", "care" and "carry", and "inn" of "inner" and "innovative". Letting
// those through meant "update my card details" granted consent to rent a car.
//
// `stems` are truncations that only exist to catch inflections — "delet" for
// deleting/deleted — and match a prefix plus at most three letters. They are
// chosen so no ordinary word begins with them.
struct EffectKind {
    const char* kind;
    const char* const* words;
    std::size_t word_count;
    const char* const* stems;
    std::size_t stem_count;
    const char* const* phrases;
    std::size_t phrase_count;
};

inline constexpr std::size_t kMaxInflection = 3;

inline const char* const kDeleteWords[] = {"delete", "wipe", "purge", "discard", "uninstall"};
inline const char* const kDeleteStems[] = {"delet", "eras", "remov"};
inline const char* const kDeletePhrases[] = {"get rid of", "getting rid of", "clear out"};

inline const char* const kCancelWords[] = {"cancel"};
inline const char* const kCancelStems[] = {"cancel", "unsubscrib", "deactivat", "terminat", "revok"};
inline const char* const kCancelPhrases[] = {"take down", "shut down"};

inline const char* const kCredentialWords[] = {"password", "passcode", "credential"};
inline const char* const kCredentialStems[] = {};

inline const char* const kScheduleWords[] = {};
inline const char* const kScheduleStems[] = {"reschedul", "postpon"};
inline const char* const kSchedulePhrases[] = {"push back", "bring forward"};

// Booking kinds. NAME PARTS ARE DELIBERATELY ABSENT — "Palace" and "Suites" are
// words inside catalogue entries, and admitting them would let a request
// satisfy the rule by NAMING A BUSINESS rather than by asking for the effect.
inline const char* const kLodgingWords[] = {
    "hotel", "room", "inn", "night", "stay", "lodging", "accommodation",
};
inline const char* const kLodgingStems[] = {"accommodat"};
inline const char* const kLodgingPhrases[] = {"place to sleep", "somewhere to sleep"};

inline const char* const kDiningWords[] = {
    "restaurant", "dinner", "lunch", "brunch", "meal", "table", "food", "cuisine",
};
inline const char* const kDiningStems[] = {"dine", "dining", "eat"};
inline const char* const kDiningPhrases[] = {"book a table"};

// Reading a page is an effect the user has to have asked for, like booking a
// room. The words are what a request for it looks like — not name parts, and
// not anything specific to a site.
inline const char* const kBrowseWords[] = {
    "webpage", "website", "url", "link", "site", "page", "browse", "visit", "article",
};
inline const char* const kBrowseStems[] = {"brows"};
inline const char* const kBrowsePhrases[] = {"web page", "read the content"};

// Putting something on a calendar. Whole words for the ones that are words —
// "event" would otherwise prefix-match "eventually", "meeting" is not "meet" —
// and truncations only where an inflection is what is wanted.
inline const char* const kCalendarWords[] = {
    "calendar", "event", "appointment", "meeting", "reminder",
};
inline const char* const kCalendarStems[] = {"remind", "schedul", "invit"};
inline const char* const kCalendarPhrases[] = {"add an event", "put it in my"};

inline const char* const kTransportWords[] = {"car", "vehicle", "rental"};
inline const char* const kTransportStems[] = {"rent", "driv"};
inline const char* const kTransportPhrases[] = {"pick me up"};

#define ADJ_KIND(name, w, st, ph) \
    {name, w, sizeof(w) / sizeof(*w), st, sizeof(st) / sizeof(*st), \
     ph, sizeof(ph) / sizeof(*ph)}

inline const char* const kNoPhrases[] = {""};

inline const EffectKind kEffectKinds[] = {
    ADJ_KIND("delete", kDeleteWords, kDeleteStems, kDeletePhrases),
    ADJ_KIND("cancel", kCancelWords, kCancelStems, kCancelPhrases),
    {"credential", kCredentialWords, sizeof(kCredentialWords) / sizeof(*kCredentialWords),
     nullptr, 0, nullptr, 0},
    {"schedule", nullptr, 0, kScheduleStems, sizeof(kScheduleStems) / sizeof(*kScheduleStems),
     kSchedulePhrases, sizeof(kSchedulePhrases) / sizeof(*kSchedulePhrases)},
    ADJ_KIND("lodging", kLodgingWords, kLodgingStems, kLodgingPhrases),
    ADJ_KIND("dining", kDiningWords, kDiningStems, kDiningPhrases),
    ADJ_KIND("transport", kTransportWords, kTransportStems, kTransportPhrases),
    ADJ_KIND("browse", kBrowseWords, kBrowseStems, kBrowsePhrases),
    ADJ_KIND("calendar", kCalendarWords, kCalendarStems, kCalendarPhrases),
};

// Whole word, or the word plus a plural "s".
inline bool word_is(const std::string& word, const std::string& target) {
    if (word == target) return true;
    return word.size() == target.size() + 1 && word.back() == 's'
        && word.compare(0, target.size(), target) == 0;
}

// A truncation, plus at most `kMaxInflection` further letters.
inline bool word_starts_with(const std::string& low, std::size_t i, const std::string& stem) {
    const std::string w = word_at(low, i);
    if (w.size() < stem.size()) return false;
    if (w.compare(0, stem.size(), stem) != 0) return false;
    return w.size() - stem.size() <= kMaxInflection;
}

// Does `text` — always the user's own request — ask for an effect of `kind`?
inline bool asks_for_effect(const std::string& text, const std::string& kind) {
    std::string low;
    low.reserve(text.size());
    for (unsigned char c : text) low.push_back(static_cast<char>(std::tolower(c)));

    for (const EffectKind& k : kEffectKinds) {
        if (kind != k.kind) continue;
        for (std::size_t j = 0; j < k.phrase_count; ++j) {
            if (k.phrases[j] && *k.phrases[j] && low.find(k.phrases[j]) != std::string::npos) {
                return true;
            }
        }
        for (std::size_t i = 0; i < low.size(); ++i) {
            const std::string w = word_at(low, i);
            if (w.empty()) continue;
            for (std::size_t j = 0; j < k.word_count; ++j) {
                if (word_is(w, k.words[j])) return true;
            }
            for (std::size_t j = 0; j < k.stem_count; ++j) {
                if (word_starts_with(low, i, k.stems[j])) return true;
            }
        }
        return false;
    }
    return false;
}

// Any IRREVERSIBLE effect at all — the fallback for a deployment that has not
// classified a tool.
//
// The booking kinds are deliberately excluded. They are effects the user has to
// have asked for, but they are not destruction: looping every kind here made
// "book a table tonight" answer yes, so a request to reserve a restaurant would
// have unlocked every unclassified irreversible tool for the session.
inline const char* const kIrreversibleKinds[] = {"delete", "cancel", "credential", "schedule"};

inline bool asks_for_destruction(const std::string& text) {
    for (const char* k : kIrreversibleKinds) {
        if (asks_for_effect(text, k)) return true;
    }
    return false;
}

// ── @mentions_date ────────────────────────────────────────────────────────
//
// Does `text` mention the calendar date `value` names? The value is a tool
// argument in ISO form, `YYYY-MM-DD`, optionally followed by a space or `T`
// and a time of day (`HH:MM[:SS]`), which is not compared. Any other value is
// not a date and is never mentioned. The text is prose: a result record
// an outsider may have written, or the user's own request.
//
// The phrasings recognised, all with the day as a separate number 1..31 that
// may carry an ordinal suffix (st/nd/rd/th):
//   - the ISO date itself, verbatim;
//   - "January 11", "January the 11th", "11 January", "the 11th of January",
//     with the month written in full or abbreviated ("Jan", "Jan.", "Sept"),
//     in any case;
//   - a range that writes the month once: "January 11 to 15" names the 15th,
//     and "the 1st to the 5th of May" names the 1st. The separator is one of
//     to, until, till, through, thru, and, a hyphen or an en dash.
// The year of a mention is the first four-digit year 1900..2099 written after
// it in the same sentence, at most eight tokens on: "May 1st 2024", "January
// 11, 2025", and both days of "January 11th to January 15th 2025" or "the 1st
// to the 5th of May 2024". It must be the value's year. A mention with no year
// after it matches any year, and a year in another sentence binds nothing. A
// sentence ends at . ! ? ; or a line break; a comma does not end it. In a
// range, a day with its own month beside it belongs to that month: "January 11
// to 15 February" names January 11 and February 15, not January 15.
//
// Everything else is "not mentioned": numeric dates such as 01/11/2025 (January
// or November depending on the reader), dates in words ("May first") and
// relative dates ("next Friday"). Rule W uses this to ask whether an outsider
// dictated a booking date, so a phrasing missed here is a dictation W does not
// see; that limit is stated in the policy and the docs rather than hidden.
//
// "May" is also a verb, but every form above needs a day number beside the
// month word, so "you may 3 times" is the only kind of text it misreads — and
// only when a booking happens to be for the 3rd of May.

struct DateTok {
    // kStop ends a sentence (. ! ? ; or a line break); kBreak is any other
    // punctuation. Both interrupt a date, but only kStop ends the search for
    // its year.
    enum Kind { kWord, kNum, kDash, kBreak, kStop } kind;
    std::string word;   // lower-cased, kWord only
    int day;            // kNum: the day 1..31, or 0 if the number is not one
    int year;           // kNum: a four-digit year 1900..2099, or 0
};

inline const char* const kMonthNames[12] = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
};
inline const char* const kMonthAbbrevs[12] = {
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
};

inline bool is_month_abbrev(const std::string& w) {
    if (w == "sept") return true;
    for (const char* a : kMonthAbbrevs) {
        if (w == a) return true;
    }
    return false;
}

// 1..12 for a month word, 0 otherwise.
inline int month_of(const DateTok& t) {
    if (t.kind != DateTok::kWord) return 0;
    if (t.word == "sept") return 9;
    for (int m = 0; m < 12; ++m) {
        if (t.word == kMonthNames[m] || t.word == kMonthAbbrevs[m]) return m + 1;
    }
    return 0;
}

inline std::vector<DateTok> date_tokens(const std::string& text) {
    std::vector<DateTok> out;
    const std::size_t n = text.size();
    std::size_t i = 0;
    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c == '\n' || c == '.' || c == '!' || c == '?' || c == ';') {
            out.push_back({DateTok::kStop, std::string(), 0, 0});
            ++i;
        } else if (std::isspace(c)) {
            ++i;
        } else if (std::isalpha(c)) {
            std::string w;
            std::size_t j = i;
            // A token is a whole run of letters and digits: "A2024" and "AA11"
            // are identifiers, not a word followed by a year or a day.
            while (j < n && std::isalnum(static_cast<unsigned char>(text[j]))) {
                w.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(text[j]))));
                ++j;
            }
            // "Jan. 11" — the full stop belongs to the abbreviation, but only
            // when a number follows. In "for 11 Jan. We stayed in 2024" or "11
            // May." it ends the sentence, and the next sentence's year must
            // not bind the date.
            if (j < n && text[j] == '.' && is_month_abbrev(w)) {
                std::size_t k = j + 1;
                while (k < n && text[k] == ' ') ++k;
                if (k < n && std::isdigit(static_cast<unsigned char>(text[k]))) ++j;
            }
            out.push_back({DateTok::kWord, w, 0, 0});
            i = j;
        } else if (std::isdigit(c)) {
            std::size_t j = i;
            while (j < n && std::isdigit(static_cast<unsigned char>(text[j]))) ++j;
            const std::size_t len = j - i;
            const int num = len <= 4 ? std::atoi(text.substr(i, len).c_str()) : 0;
            // The rest of the letter-and-digit run, so "9X2024" stays one token
            // (an identifier) instead of "9X" followed by the year 2024.
            std::string suffix;
            while (j < n && std::isalnum(static_cast<unsigned char>(text[j]))) {
                suffix.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(text[j]))));
                ++j;
            }
            int day = len <= 2 ? num : 0;
            int year = (len == 4 && num >= 1900 && num <= 2099) ? num : 0;
            // "11th" is a day; "11am" or "3x" is not, and a year takes no suffix.
            if (!suffix.empty()) {
                year = 0;
                if (suffix != "st" && suffix != "nd" && suffix != "rd" && suffix != "th") day = 0;
            }
            // A comma after the number is left in place: it separates "January
            // 11, February 15" into two dates, and the year search crosses it.
            if (day < 1 || day > 31) day = 0;
            out.push_back({DateTok::kNum, std::string(), day, year});
            i = j;
        } else if (c == '-') {
            out.push_back({DateTok::kDash, std::string(), 0, 0});
            ++i;
        } else if (c == 0xE2 && i + 2 < n && static_cast<unsigned char>(text[i + 1]) == 0x80
                   && static_cast<unsigned char>(text[i + 2]) == 0x93) {
            // U+2013 EN DASH, as UTF-8.
            out.push_back({DateTok::kDash, std::string(), 0, 0});
            i += 3;
        } else {
            out.push_back({DateTok::kBreak, std::string(), 0, 0});
            ++i;
        }
    }
    return out;
}

// Parses `value` as a date: exactly `YYYY-MM-DD`, or that followed by a space
// or `T` and a time of day `HH:MM` or `HH:MM:SS`. Nothing else may follow. A
// value that merely STARTS with a date — "2025-01-11 I resign" — is prose, and
// must stay under the verbatim test rather than be read as its first ten
// characters.
inline bool parse_iso_date(const std::string& value, int* y, int* m, int* d) {
    if (value.size() != 10 && value.size() != 16 && value.size() != 19) return false;
    for (std::size_t k = 0; k < 10; ++k) {
        const unsigned char c = static_cast<unsigned char>(value[k]);
        if (k == 4 || k == 7) {
            if (c != '-') return false;
        } else if (!std::isdigit(c)) {
            return false;
        }
    }
    if (value.size() > 10) {
        if (value[10] != ' ' && value[10] != 'T') return false;
        for (std::size_t k = 11; k < value.size(); ++k) {
            const unsigned char c = static_cast<unsigned char>(value[k]);
            if (k == 13 || k == 16) {
                if (c != ':') return false;
            } else if (!std::isdigit(c)) {
                return false;
            }
        }
    }
    if (value.size() > 10) {
        // A time of day that exists: hours 0..23, minutes and seconds 0..59.
        if (std::atoi(value.substr(11, 2).c_str()) > 23) return false;
        if (std::atoi(value.substr(14, 2).c_str()) > 59) return false;
        if (value.size() == 19 && std::atoi(value.substr(17, 2).c_str()) > 59) return false;
    }
    *y = std::atoi(value.substr(0, 4).c_str());
    *m = std::atoi(value.substr(5, 2).c_str());
    *d = std::atoi(value.substr(8, 2).c_str());
    if (*m < 1 || *m > 12 || *d < 1) return false;
    // A day that exists in that month of that year.
    static const int kDays[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    const bool leap = (*y % 4 == 0 && *y % 100 != 0) || *y % 400 == 0;
    return *d <= kDays[*m - 1] + ((*m == 2 && leap) ? 1 : 0);
}

inline bool is_iso_date(const std::string& value) {
    int y = 0, m = 0, d = 0;
    return parse_iso_date(value, &y, &m, &d);
}

inline bool mentions_date(const std::string& text, const std::string& value) {
    int y = 0, m = 0, d = 0;
    if (!parse_iso_date(value, &y, &m, &d)) return false;

    // (a) The ISO date verbatim, not embedded in a longer run of digits.
    const std::string iso = value.substr(0, 10);
    for (std::size_t p = text.find(iso); p != std::string::npos; p = text.find(iso, p + 1)) {
        const bool left_ok = p == 0 || !std::isdigit(static_cast<unsigned char>(text[p - 1]));
        const std::size_t e = p + iso.size();
        const bool right_ok = e == text.size() || !std::isdigit(static_cast<unsigned char>(text[e]));
        if (left_ok && right_ok) return true;
    }

    const std::vector<DateTok> t = date_tokens(text);
    const long n = static_cast<long>(t.size());
    auto word_at = [&](long k, const char* w) {
        return k >= 0 && k < n && t[k].kind == DateTok::kWord && t[k].word == w;
    };
    auto day_at = [&](long k) { return (k >= 0 && k < n && t[k].kind == DateTok::kNum) ? t[k].day : 0; };
    auto sep_at = [&](long k) {
        if (k < 0 || k >= n) return false;
        if (t[k].kind == DateTok::kDash) return true;
        return word_at(k, "to") || word_at(k, "until") || word_at(k, "till")
            || word_at(k, "through") || word_at(k, "thru") || word_at(k, "and");
    };
    // The mention ends at token `k`; the first year after it, before the
    // sentence ends, must agree.
    // Eight words and numbers on; punctuation does not count toward them.
    auto year_ok = [&](long k) {
        int seen = 0;
        for (long j = k + 1; j < n && seen < 8; ++j) {
            if (t[j].kind == DateTok::kStop) break;
            if (t[j].kind != DateTok::kWord && t[j].kind != DateTok::kNum) continue;
            ++seen;
            if (t[j].kind == DateTok::kNum && t[j].year != 0) return t[j].year == y;
        }
        return true;
    };

    for (long k = 0; k < n; ++k) {
        if (month_of(t[k]) != m) continue;

        // (b) "Month [the] D" and (c) "Month [the] D1 <sep> [the] D2".
        long j = k + 1;
        if (word_at(j, "the")) ++j;
        if (const int d1 = day_at(j)) {
            if (d1 == d && year_ok(j)) return true;
            long s = j + 1;
            if (sep_at(s)) {
                long j2 = s + 1;
                if (word_at(j2, "the")) ++j2;
                // "January 11 to 15 February": a second day with a month of
                // its own after it is that month's day, not this one's.
                long after = j2 + 1;
                if (word_at(after, "of")) ++after;
                const bool own_month = after < n && month_of(t[after]) != 0;
                if (!own_month && day_at(j2) == d && year_ok(j2)) return true;
            }
        }

        // (b) "D Month", "D of Month" and (c) "D1 <sep> [the] D2 [of] Month".
        j = k - 1;
        if (word_at(j, "of")) --j;
        if (const int d2 = day_at(j)) {
            if (d2 == d && year_ok(k)) return true;
            long s = j - 1;
            if (word_at(s, "the")) --s;
            // "March 3 - 11 January": a first day with a month of its own
            // before it is that month's day, not this one's.
            long before = s - 2;
            if (word_at(before, "the")) --before;
            const bool own_month = before >= 0 && month_of(t[before]) != 0;
            if (sep_at(s) && !own_month && day_at(s - 1) == d && year_ok(k)) return true;
        }
    }
    return false;
}

}  // namespace adj

#endif  // SASY_AGENTDOJO_FUNCTORS_IMPL_H_
