# Worked Example (Mode S)

**User:** "I want a policy where only premium users can use the
image generation API" (`FILE=experiments/premium_images_policy.dl`)

## Structured spec (step 5)

```
C1: Allow role=premium to make HTTP requests to api.openai.com/v1/images
C2: Inject transform "inject_openai_key" for authorized OpenAI requests
C3: Deny with error for non-premium users on api.openai.com/v1/images
```

## Generated Soufflé with citations (step 7)

```datalog
/* Premium Image Generation Policy */
// Round-trip audit: PASS (2026-06-10)
#include "../policies/common_policy.dl"

/* ── K-Constraints ───────────────────
   K1: no multi-party message sources in this system — no
       msg.agent filtering needed (HTTP-only policy, no
       SentMessage rules)
   ─────────────────────────────────── */

/* ── Structured Spec ─────────────────
   C1: Allow role=premium to make HTTP requests to api.openai.com/v1/images
   C2: Inject transform "inject_openai_key" for authorized OpenAI requests
   C3: Deny with error for non-premium users on api.openai.com/v1/images
   ─────────────────────────────────── */

/* C1: Allow role=premium to make HTTP requests to api.openai.com/v1/images */
IsAuthorized(idx) :-
    Actions(idx, a),
    QueriesHost(a, "api.openai.com"),
    a = $HTTPRequest(url, _, _),
    @str_contains(url, "/v1/images") = 1,
    HasRole("premium").

/* C2: Inject transform "inject_openai_key" for authorized OpenAI requests */
ApplyTransform(idx, "inject_openai_key") :-
    Actions(idx, a),
    QueriesHost(a, "api.openai.com"),
    Authorized(idx).

/* C3: Deny with error for non-premium users on api.openai.com/v1/images */
DenyUnauthorized(idx) :-
    Actions(idx, a),
    QueriesHost(a, "api.openai.com"),
    a = $HTTPRequest(url, _, _),
    @str_contains(url, "/v1/images") = 1.
```

(The `#include` path is relative to the policy file's location.)

## Validate (step 8)

```bash
make souffle-validate FILE=experiments/premium_images_policy.dl
```

## Test scenarios (step 9) — `experiments/premium_images_policy_tests.json`

```json
{
  "policy_description": "Only premium users can use OpenAI image generation",
  "scenarios": [
    {
      "name": "premium user can generate images",
      "clause": "C1",
      "entity": "alice",
      "roles": ["premium"],
      "action": {"type": "HTTPRequest",
                 "url": "https://api.openai.com/v1/images/generations",
                 "body": "{}"},
      "graph": {},
      "expected": {"authorized": true, "transforms": ["inject_openai_key"]}
    },
    {
      "name": "free user is denied image generation",
      "clause": "C3",
      "entity": "bob",
      "roles": ["free"],
      "action": {"type": "HTTPRequest",
                 "url": "https://api.openai.com/v1/images/generations",
                 "body": "{}"},
      "graph": {},
      "expected": {"authorized": false, "deny_unauthorized": true}
    }
  ]
}
```

## Run tests + finalize (steps 10, 12)

```bash
make policy-test POLICY=experiments/premium_images_policy.dl \
    SCENARIOS=experiments/premium_images_policy_tests.json
make policy-finalize POLICY=experiments/premium_images_policy.dl \
    SCENARIOS=experiments/premium_images_policy_tests.json
```
