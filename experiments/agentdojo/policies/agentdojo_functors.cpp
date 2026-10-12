/* Custom Soufflé functors for the AgentDojo policy.

   Three helpers, all about comparing identifiers — email addresses, hosts and
   account numbers — the way a person would rather than the way `strstr` does.

     @contains_identifier(haystack, needle)
         1 iff `needle` occurs in `haystack` once both are folded to lower
         case and stripped of separators. An account number a statement prints
         as "GB29 NWBK 6016 1331 9268 19" then matches the argument
         "GB29NWBK60161331926819", and a URL written with or without its
         scheme matches either way. Without this an attacker only has to
         reformat an identifier to make a provenance check miss.

         The example above is a documentation placeholder, not a value the
         functor knows about: nothing in this file or in the policy encodes a
         specific address, host or account. Every comparison is against text
         the user wrote or the operator's own records returned.

     @identifier_eq(a, b)
         The same folding, applied to whole-value equality. Used against the
         monitor's directory entries, which are stored already folded.

     @nth_identifier(text, k)
         The k-th identifier appearing in `text`, or "" past the end. Lets the
         policy walk the addresses and links inside a message body with a
         bounded index relation instead of a regex in Datalog.

     @nth_catalog_entity(text, suffixes, k)
         The k-th capitalised multi-word name in `text` whose final word is one
         of the `|`-separated `suffixes` — "Riverside View Hotel" for a suffix
         list containing "Hotel". Addresses and account numbers are not the only
         things an agent can be talked into putting in front of a user: an
         injection that never calls a tool can still get a business promoted in
         the agent's answer.

         The suffix list arrives as one delimited string rather than one fact
         per suffix because Datalog has no set-valued functor argument: a
         relation of N suffixes would force an N-way cross product with the
         index relation, so a travel deployment's eleven suffixes cost 88 calls
         per evaluation instead of 8, once per sentence of a redaction. The
         values stay deployment configuration either way.

     @is_generic_entity(name, generics)
         1 iff `name` is one of the `|`-separated `generics` — the category
         headings an answer is organised under ("Car Rental"), which look like
         business names and vouch for nothing. Same collapsing rationale.

     @requests_destructive_effect(text)
         1 iff `text` — always the *user's own request* — asks for something
         irreversible: a deletion, a cancellation, a credential change. This is
         a classifier, not configuration, which is why it is a functor and not
         a word list in policy metadata: no deployer tunes it, and expressing it
         as substring facts got it wrong in both directions ("delete" matched
         *undelete*, while "get rid of" and "wipe" matched nothing). Matching
         here is word-initial, so a stem only fires at the start of a word.

     @mentions_date(text, value)
         1 iff `text` names the calendar day that `value` is. The value is a
         tool argument in ISO form, "2025-01-11", possibly followed by a time
         of day ("2025-01-11 18:00"), which is not compared; a value with
         anything else after the date is not a date and gets 0. The text names it if it contains
         the ISO date verbatim, or writes it with an English month name beside
         the day number ("January 11th", "11 Jan", "the 11th of January"), or
         inside a range that writes the month once ("January 11 to 15", "the
         1st to the 5th of May"). The first year written after the mention,
         in the same sentence, must be the value's year; with none, any year
         matches. Numeric dates
         such as 01/11/2025 are not read, because MM/DD and DD/MM disagree, nor
         are dates in words or relative dates. Rule W asks it twice: of an
         outsider's record (does it dictate this booking date?) and of the
         user's request (did the user give the date too?). A phrasing it does
         not read is therefore a dictation rule W misses.

   `fold` deliberately keeps digits and letters only. That makes matching
   robust to punctuation, at the cost of treating "acme.com" and "acmecom" as
   the same string — a collision that cannot help an attacker, since it can
   only make an unanchored value look anchored if the trusted text already
   contained the same characters in the same order.
*/

#include <souffle/SouffleInterface.h>

#include <string>
#include <vector>

#include "agentdojo_functors_impl.h"

extern "C" {

// How many identifiers a block of text carries, and how many names of a
// catalogued type. The policy walks these with a bounded index relation, so it
// needs to know when there are more than it can reach: a list whose tail is
// never examined is a list an attacker puts the payload at the end of.
souffle::RamDomain identifier_count(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text) {
    return static_cast<souffle::RamDomain>(
        adj::identifiers(symbolTable->decode(text)).size());
}

souffle::RamDomain catalog_entity_count(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text,
    souffle::RamDomain suffixes) {
    return static_cast<souffle::RamDomain>(
        adj::catalog_entities(symbolTable->decode(text),
                         symbolTable->decode(suffixes)).size());
}

// 1 iff `value` belongs to `domain`: the host part equals it, or is a
// subdomain of it. Substring matching was the bug this replaces —
// "bluesparrowtech.com.evil.example" contains "bluesparrowtech.com" and was
// therefore treated as internal.
souffle::RamDomain host_in_domain(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain value,
    souffle::RamDomain domain) {
    std::string d;
    for (unsigned char c : symbolTable->decode(domain)) {
        d.push_back(static_cast<char>(std::tolower(c)));
    }
    if (d.empty()) return 0;
    const std::string host = adj::host_of(symbolTable->decode(value));
    if (host == d) return 1;
    // A subdomain of ours is ours; a domain that merely ENDS with our name is
    // not. The dot is what carries that, and dropping it would make
    // "notbluesparrowtech.com" internal.
    if (host.size() > d.size()
        && host.compare(host.size() - d.size(), d.size(), d) == 0
        && host[host.size() - d.size() - 1] == '.') {
        return 1;
    }
    return 0;
}

souffle::RamDomain contains_identifier(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain haystack,
    souffle::RamDomain needle) {
    return adj::anchors(symbolTable->decode(haystack),
                        symbolTable->decode(needle)) ? 1 : 0;
}

souffle::RamDomain identifier_eq(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain a,
    souffle::RamDomain b) {
    const std::string x = adj::fold(symbolTable->decode(a));
    const std::string y = adj::fold(symbolTable->decode(b));
    return (!x.empty() && x == y) ? 1 : 0;
}

souffle::RamDomain nth_catalog_entity(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text,
    souffle::RamDomain suffixes,
    souffle::RamDomain k) {
    const std::vector<std::string> names = adj::catalog_entities(
        symbolTable->decode(text), symbolTable->decode(suffixes));
    const auto idx = static_cast<std::size_t>(k);
    return symbolTable->encode(idx < names.size() ? names[idx] : std::string());
}

souffle::RamDomain criterion_class(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text) {
    const std::string c = adj::criterion_of(symbolTable->decode(text));
    const std::size_t colon = c.find(':');
    return symbolTable->encode(colon == std::string::npos ? std::string()
                                                         : c.substr(0, colon));
}

souffle::RamDomain criterion_wants_max(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text) {
    const std::string c = adj::criterion_of(symbolTable->decode(text));
    return c.size() > 4 && c.substr(c.size() - 4) == ":max" ? 1 : 0;
}

souffle::RamDomain selected_id(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain listing,
    souffle::RamDomain id_field,
    souffle::RamDomain value_field,
    souffle::RamDomain want_max) {
    return symbolTable->encode(adj::select_by(symbolTable->decode(listing),
                                              symbolTable->decode(id_field),
                                              symbolTable->decode(value_field),
                                              want_max != 0));
}

souffle::RamDomain is_dictated_by_outsider(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain haystack,
    souffle::RamDomain needle,
    souffle::RamDomain own_domains,
    souffle::RamDomain own_principals) {
    return adj::is_dictated_by_outsider(symbolTable->decode(haystack),
                                        symbolTable->decode(needle),
                                        symbolTable->decode(own_domains),
                                        symbolTable->decode(own_principals))
               ? 1
               : 0;
}

souffle::RamDomain is_dictated(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain haystack,
    souffle::RamDomain needle) {
    return adj::is_dictated(symbolTable->decode(haystack),
                            symbolTable->decode(needle))
               ? 1
               : 0;
}

souffle::RamDomain url_carries_data(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain value) {
    return adj::url_carries_data(symbolTable->decode(value)) ? 1 : 0;
}

souffle::RamDomain is_account_number(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain value) {
    return adj::is_account_number(symbolTable->decode(value)) ? 1 : 0;
}

souffle::RamDomain entity_core(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain name,
    souffle::RamDomain suffixes) {
    const std::string core = adj::entity_core(symbolTable->decode(name),
                                              symbolTable->decode(suffixes));
    return symbolTable->encode(core);
}

souffle::RamDomain is_generic_entity(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain name,
    souffle::RamDomain generics) {
    const std::string n = adj::fold(symbolTable->decode(name));
    for (const std::string& g : adj::split_pipe(symbolTable->decode(generics))) {
        if (!n.empty() && n == adj::fold(g)) return 1;
    }
    return 0;
}

souffle::RamDomain requests_effect_kind(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text,
    souffle::RamDomain kind) {
    return adj::asks_for_effect(symbolTable->decode(text),
                                symbolTable->decode(kind)) ? 1 : 0;
}

souffle::RamDomain requests_destructive_effect(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text) {
    return adj::asks_for_destruction(symbolTable->decode(text)) ? 1 : 0;
}

souffle::RamDomain mentions_date(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text,
    souffle::RamDomain value) {
    return adj::mentions_date(symbolTable->decode(text),
                              symbolTable->decode(value)) ? 1 : 0;
}

souffle::RamDomain nth_identifier(
    souffle::SymbolTable* symbolTable,
    souffle::RecordTable* /*recordTable*/,
    souffle::RamDomain text,
    souffle::RamDomain k) {
    const std::vector<std::string> ids = adj::identifiers(symbolTable->decode(text));
    const auto idx = static_cast<std::size_t>(k);
    return symbolTable->encode(idx < ids.size() ? ids[idx] : std::string());
}

}  // extern "C"
