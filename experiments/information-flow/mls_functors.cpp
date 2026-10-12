/* Custom functors for MLS policy */
#include <souffle/SouffleInterface.h>
#include <string>

extern "C" {

souffle::RamDomain file_security_level(
    souffle::SymbolTable* symbolTable, souffle::RecordTable*,
    souffle::RamDomain path_sym)
{
    const std::string& path = symbolTable->decode(path_sym);
    if (path.find("/secret/") != std::string::npos) return 3;
    if (path.find("/restricted/") != std::string::npos) return 2;
    if (path.find("/internal/") != std::string::npos) return 1;
    return 0;
}

souffle::RamDomain entity_clearance(
    souffle::SymbolTable* symbolTable, souffle::RecordTable*,
    souffle::RamDomain entity_sym)
{
    const std::string& entity = symbolTable->decode(entity_sym);
    if (entity == "jsmith" || entity == "dwilliams" || entity == "mdavis") return 3;
    if (entity == "sbrown" || entity == "achen") return 2;
    return 1;
}

souffle::RamDomain recipient_clearance(
    souffle::SymbolTable* symbolTable, souffle::RecordTable*,
    souffle::RamDomain email_sym)
{
    const std::string& email = symbolTable->decode(email_sym);
    if (email.empty()) return 0;
    if (email.find("@acme.com") != std::string::npos) {
        auto at = email.find('@');
        if (at != std::string::npos) {
            std::string user = email.substr(0, at);
            if (user == "jsmith" || user == "dwilliams" || user == "mdavis") return 3;
            if (user == "sbrown" || user == "achen") return 2;
            return 1;
        }
        return 1;
    }
    if (email.find("@xyz.com") != std::string::npos) return 1;
    return 0;
}

} // extern "C"
