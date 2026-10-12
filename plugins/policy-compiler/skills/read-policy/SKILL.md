---
name: read-policy
description: Translate a Soufflé Datalog policy back into structured English with evidence citations
---

# Read Soufflé Policy

This skill translates a Soufflé Datalog policy into structured English,
with each English statement grounded by citing the specific Soufflé rules
that support it. This is the reverse of the `/write-policy` skill.

## MANDATORY FIRST STEP: Read Schema

Before interpreting any policy, you MUST read the common policy schema:

```bash
cat policies/common_policy.dl
```

This defines ALL relations, types, and functors. You need this to
understand what each predicate means.

## Workflow

1. **Read the schema** — read `policies/common_policy.dl`
2. **Read the target policy** — read the Soufflé `.dl` file specified by the user
3. **Identify all rules** — list every rule in the policy (excluding `#include` and common_policy definitions)
4. **Translate to structured English** — produce a numbered clause list with evidence (see format below)
5. **Coverage check** — verify every rule is cited by at least one clause, and flag any orphan rules
6. **Cross-check citations** *(if available)* — if the policy has `/* C_n: ... */` comments from `/write-policy`, compare your reconstruction against them
7. **Present to user** — show the structured English with evidence

## Input

The user specifies the policy file:
```
/read-policy policies/my_policy.dl
```

Or:
```
/read-policy FILE=experiments/airline_policy.dl
```

## Output Format

The output MUST be structured, NOT narrative prose. Each clause is a
single authorization statement grounded by evidence from the Soufflé
source:

```
C1: Admins can use the delete_user tool
    Evidence: IsAuthorized(idx) at line 8
    — matches CallTool actions where IsTool(a, "delete_user")
      AND HasRole("admin")

C2: Non-admins are denied with an error for delete_user
    Evidence: DenyUnauthorized(idx) at line 13
    — triggers for all CallTool("delete_user") actions

C3: Authorized OpenAI requests get API key injection
    Evidence: ApplyTransform(idx, "inject_openai_key") at line 18
    — applies when QueriesHost(a, "api.openai.com") AND Authorized(idx)
```

### Rules for structuring output

- **One clause per authorization decision** — each clause describes one
  allow, deny, transform, or passthrough rule
- **Evidence is mandatory** — every clause must cite the specific Soufflé
  rule(s) with line numbers
- **Evidence explains the conditions** — list the predicates that must hold,
  translated to plain English
- **Group related rules** — if multiple Soufflé rules implement a single
  logical concept (e.g., a recursive taint tracker), combine them into
  one clause with multiple evidence lines
- **Flag ambiguity** — if a rule's intent is unclear, say so explicitly
  rather than guessing

## Soufflé Syntax Reference (for reading)

### Output relations → English meaning

| Soufflé | English meaning |
|---------|-----------------|
| `IsAuthorized(idx)` | This action is allowed |
| `Unauthorized(idx)` | This action is denied (overrides allow) |
| `DenialReason(idx, reason, suggestion)` | Annotated deny with structured feedback (sugar.py rewrites `// @deny(...)` `Unauthorized` rules into this; `common_policy.dl` derives `Unauthorized(idx) :- DenialReason(idx, _, _).`) |
| `ApplyTransform(idx, "id")` | Inject credential transform "id" |
| `DenyUnauthorized(idx)` | Return error if action is not authorized |
| `AllowPassthrough(idx)` | Let action through unchanged if not authorized |

### Helper relations → English meaning

| Soufflé | English meaning |
|---------|-----------------|
| `HasRole("X")` | The authenticated entity has role X |
| `QueriesHost(a, "host")` | The HTTP request targets this host |
| `IsTool(a, "name")` | The action is a call to tool "name" |
| `IsToolCall(a)` | The action is any tool call |
| `CurrentDepends(src)` | The current request depends on event src |
| `HasAuthenticatedEntity()` | There is an authenticated entity |
| `Authorized(idx)` | Final decision: allowed AND not denied |

### Type constructors → English meaning

| Soufflé | English meaning |
|---------|-----------------|
| `a = $CallTool(name, args)` | The action is a tool call with these args |
| `a = $HTTPRequest(url, body, _)` | The action is an HTTP request to this URL |
| `a = $SendAttempt(msg)` | The action is sending a message |

### Functor calls → English meaning

| Soufflé | English meaning |
|---------|-----------------|
| `@str_contains(s, "x") = 1` | String s contains "x" |
| `@str_starts_with(s, "x") = 1` | String s starts with "x" |
| `@str_to_lower(s)` | Lowercase version of s |
| `@json_get_str(args, "field")` | Extract string field from JSON args |
| `@json_get_int(args, "field")` | Extract integer field from JSON args |
| `@json_has_field(args, "field") = 1` | JSON args has this field |
| `@url_host(url)` | Extract hostname from URL |
| `@llm_check_fn("prompt", ctx) = 1` | LLM semantic check returns true |

### Dot notation (sugar.py)

When you see `msg.contents`, `msg.agent`, `msg.entity`, etc., these
are field accesses on Message records — desugared to positional tuple
unpacking by sugar.py.

### Negation

`!Relation(x)` means "Relation does NOT hold for x." For example, `!Unauthorized(idx)` means "action idx is not denied."

### Common policy patterns (for recognition)

**RBAC pattern:**
```datalog
IsAuthorized(idx) :- Actions(idx, a), <selector>, HasRole("role").
```
→ "Users with role X can perform action Y"

**Blacklist pattern:**
```datalog
Unauthorized(idx) :- Actions(idx, a), <selector>.
```
→ "Action Y is always denied" (overrides any allow)

**Annotated deny pattern (DenialReason):**
```datalog
// @deny(reason="...", suggestion="...")
Unauthorized(idx) :- Actions(idx, a), <selector>.
```
→ sugar.py rewrites this into `DenialReason(idx, reason, suggestion)`,
and `common_policy.dl` derives `Unauthorized(idx) :- DenialReason(idx, _, _).`
The agent receives structured feedback with the reason and suggestion.

**Conditional deny pattern:**
```datalog
Unauthorized(idx) :- Actions(idx, a), <selector>, <condition>.
```
→ "Action Y is denied when condition Z holds"

**Deny-with-error pattern:**
```datalog
DenyUnauthorized(idx) :- Actions(idx, a), <selector>.
```
→ "Return an error for unauthorized attempts at action Y"

**Credential injection pattern:**
```datalog
ApplyTransform(idx, "transform_id") :- ..., Authorized(idx).
```
→ "When action Y is authorized, inject credential Z"

**Graph-dependent pattern:**
```datalog
IsAuthorized(idx) :- ..., CurrentDepends(src), SomeCondition(src).
```
→ "Allow action Y only if something happened earlier in the conversation"

**Taint tracking pattern:**
```datalog
.decl Tainted(id: symbol)
Tainted(id) :- <source condition>.
Tainted(id) :- Tainted(src), Edge(src, id).
Unauthorized(idx) :- ..., CurrentDepends(src), Tainted(src).
```
→ "Block action Y if data flowed from an untrusted source"

**Default authorize-all pattern:**
```datalog
IsAuthorized(idx) :- Actions(idx, a).
```
→ "All actions are allowed by default (unless explicitly denied)"

## Coverage Check

After translating all rules, verify completeness:

1. **Every Soufflé rule** should be cited as evidence by at least one
   English clause
2. **Orphan rules** (rules not cited by any clause) may indicate:
   - A helper relation you missed (look at how it's used downstream)
   - Dead code that is never triggered
3. **User-defined relations** (`.decl` not in common_policy.dl):
   explain what they compute based on their rule bodies

## Cross-Check with Citations (if available)

If the policy was generated by `/write-policy`, it may contain
`/* C_n: ... */` citation comments. When present:

1. Compare your independently derived English clauses against
   the original citation comments
2. Note any discrepancies — these may indicate:
   - Translation errors in the original `/write-policy` output
   - Rules that evolved after initial generation
   - Mismatches between intent and implementation
3. Report discrepancies explicitly to the user

## Example

**Input policy:**
```datalog
/* Premium Image Generation Policy */
#include "../policies/common_policy.dl"

/* C1: Allow role=premium HTTP requests to api.openai.com/v1/images */
IsAuthorized(idx) :-
    Actions(idx, a),
    QueriesHost(a, "api.openai.com"),
    a = $HTTPRequest(url, _, _),
    @str_contains(url, "/v1/images") = 1,
    HasRole("premium").

/* C2: Inject "inject_openai_key" for authorized OpenAI requests */
ApplyTransform(idx, "inject_openai_key") :-
    Actions(idx, a),
    QueriesHost(a, "api.openai.com"),
    Authorized(idx).

/* C3: Deny with error for non-premium on api.openai.com/v1/images */
DenyUnauthorized(idx) :-
    Actions(idx, a),
    QueriesHost(a, "api.openai.com"),
    a = $HTTPRequest(url, _, _),
    @str_contains(url, "/v1/images") = 1.
```

**Output:**
```
C1: Premium users can access the OpenAI image generation API
    Evidence: IsAuthorized(idx) at line 6
    — matches HTTP requests to api.openai.com where URL contains
      "/v1/images" AND entity has role "premium"

C2: Authorized OpenAI requests get API key injection
    Evidence: ApplyTransform(idx, "inject_openai_key") at line 13
    — applies to any action targeting api.openai.com that is
      authorized (IsAuthorized AND NOT Unauthorized)

C3: Non-premium users get an error for image generation requests
    Evidence: DenyUnauthorized(idx) at line 19
    — triggers for all HTTP requests to api.openai.com where URL
      contains "/v1/images", regardless of role

Cross-check: All 3 clauses match the original C1/C2/C3 citations.
No orphan rules found.
```
