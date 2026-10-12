# Optional Hardening: Paraphrase Check & Mutation Testing

Both steps are optional; the user can skip them ("skip paraphrase
check", "skip mutation testing").

## Paraphrase Check

Tests whether the translation is robust to different wordings of
the same policy:

1. Generate 2–3 rephrasings of the original English policy
   (different wording, same meaning).
2. For each rephrasing, consider whether it would produce different
   Soufflé rules — specifically, would it change the authorization
   decision for any existing test scenario?
3. If any rephrasing suggests a different policy interpretation,
   identify the **ambiguity** in the original English and ask the
   user to clarify.
4. Optionally: translate one rephrasing to Soufflé, save it as a
   temporary file, and run the shared scenario suite against it to
   verify behavioral equivalence.

Report which rephrasings were tested and whether any produced
different authorization decisions.

## Mutation Testing

Verifies the scenario suite is strong enough to catch policy bugs:

1. For each rule in the generated policy, describe one mutation:
   - Remove a condition (e.g. delete `HasRole("admin")` or a
     `msg.agent` filter)
   - Change a string literal (e.g. `"admin"` → `"user"`)
   - Swap `IsAuthorized` and `Unauthorized`
   - Remove `DenyUnauthorized` for a guarded resource
2. For each mutation, check whether any existing scenario would
   catch it (the mutated policy would produce a different result).
3. If a mutation would NOT be caught, **add a new scenario** that
   detects it.

The `msg.agent`-filter-removal mutation is mandatory whenever the
policy has content rules: the scenario that catches it is the
impersonation probe (same content, wrong agent attribution).

Report: "N mutations proposed, M caught by existing tests, K new
tests added."
