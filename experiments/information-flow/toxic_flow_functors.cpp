/* Custom functors for toxic-flow policy */
#include <souffle/SouffleInterface.h>
#include <string>

extern "C" {

/// file_security_level(path) → number
/// /secret/ → 3, /restricted/ → 2, /internal/ → 1, else → 0
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

} // extern "C"
