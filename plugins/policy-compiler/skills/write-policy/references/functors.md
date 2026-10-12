# Custom C++ Functors

For content checks (user intent, keyword detection, pattern
matching), write custom C++ functors. They compile alongside the
policy and run in the evaluator process — no IPC or network
overhead.

## Custom functors vs `@llm_check_fn`

- **Default to custom functors** with keyword/phrase matching.
  Deterministic, fast, sufficient for most content checks, and
  testable offline. Use case-insensitive substring matching
  (`to_lower` + `find`).
- **Only use `@llm_check_fn` if the user explicitly requests
  LLM-based semantic checks**, or the policy requires paraphrase
  understanding that keyword matching demonstrably cannot cover.
  It requires the LLM oracle (`LLM_ENABLED=true`), adds network
  latency, returns 0 (condition-not-met) when the service is
  disabled, and answers from a stub table (or defaults to 0) in
  offline testing — see `SASY_LLM_STUB_FILE` in
  scripts/policy_test_harness.py.
- When you must write `@llm_check_fn` prompts: short, direct
  questions ("Does the user mention X?"), absence-safe framing so
  irrelevant messages get a clean `false`, and no exclusion
  clauses (they cause hedging).

## Declaring a functor

In the policy `.dl` file:

```datalog
.functor my_check(text: symbol): unsigned stateful
```

## File naming (REQUIRED)

Implement functors in a companion `.cpp` file the compiler can
auto-discover. Name the policy `<name>_policy.dl` and the functors
`<name>_functors.cpp` in the same directory:

- `airline_policy.dl` → `airline_functors.cpp`
- `retail_policy.dl` → `retail_functors.cpp`
- `functors.cpp` (no prefix) in the same directory also works

The offline test harness compiles the companion file too, so the
full policy is testable without a server.

## Example implementation

```cpp
#include <souffle/SouffleInterface.h>
#include <string>
#include <algorithm>

static std::string to_lower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return out;
}

extern "C" {

souffle::RamDomain my_check(
    souffle::SymbolTable* st, souffle::RecordTable*,
    souffle::RamDomain text_sym)
{
    std::string lower = to_lower(st->decode(text_sym));
    const char* phrases[] = {"keyword1", "keyword2",
                             "phrase three", nullptr};
    for (const char** p = phrases; *p; ++p)
        if (lower.find(*p) != std::string::npos) return 1;
    return 0;
}

} // extern "C"
```

Common functors from `functors_common.cpp` can be forward-declared
and reused:

```cpp
extern "C" souffle::RamDomain json_get_str(
    souffle::SymbolTable*, souffle::RecordTable*,
    souffle::RamDomain, souffle::RamDomain);
```

## Grounding keyword lists in real traffic

When captured traces are available (Mode L in the core workflow),
derive keyword lists from what users actually said in the traces —
not from what you imagine they might say. After the convergence
loop runs, mismatch reports show the exact message text the policy
mis-handled; extend the phrase table from those.
