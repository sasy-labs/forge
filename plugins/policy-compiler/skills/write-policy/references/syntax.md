# Soufflé Syntax Reference

Read this when generating or debugging Soufflé code. The
authoritative schema is `policies/common_policy.dl` — never invent
predicates.

## Syntax notes

- Use `.decl` instead of `input relation` / `output relation`
- Use `!Relation(x)` instead of `not Relation(x)`
- ADT pattern matching: `a = $CallTool(name, args)` instead of
  `match (a) { CallTool{} -> ... }`
- Use `@functor_name(args)` for C++ functor calls
- Use `msg.field` dot notation (desugared by sugar.py) for record
  field access
- Soufflé uses `,` between clauses (not `and`); rules end with `.`
- Do NOT nest functor calls — `@str_contains(@str_to_lower(...))`
  is rejected; bind the inner result to a variable first

## Type constructors

```datalog
a = $CallTool(name, args)        // match a tool call action
a = $HTTPRequest(url, body, _)   // match an HTTP request action
a = $SendAttempt(msg)            // match a send message action
```

## String operations (C++ functors)

```datalog
@str_contains(s, "substring") = 1    // returns unsigned (1=true)
@str_starts_with(s, "prefix") = 1
@str_to_lower(s) = lower_s           // returns symbol
```

## JSON operations (C++ functors)

```datalog
val = @json_get_str(args, "field_name"), val != ""
n = @json_get_int(args, "count")
@json_has_field(args, "field") = 1
```

## URL operations

```datalog
@url_host(url) = "api.example.com"
```

## Dot notation sugar (processed by sugar.py)

```datalog
msg.agent           // access field of a Message record
msg.contents        // message text content
msg.entity          // caller-supplied actor (NOT auth-attested)
msg.principal       // server-stamped caller identity
```

## Key Relations

### Input relations (provided by the system)

| Relation | Description |
|----------|-------------|
| `Actions(idx, a)` | Actions to authorize (idx=unsigned, a=Action) |
| `Principal(id)` | Server-stamped auth-derived caller identity |
| `Entity(id)` | Caller-supplied actor (free-form, unattested) |
| `PrincipalRole(principal, role)` | Principal's assigned roles |
| `Current(id)` | Current message node(s) |
| `Edge(src, dst)` | Direct dependency edges (dst depends on src) |
| `EdgeData(src, dst, [proximal, message_index])` | Sparse edge metadata (-1 = unset) |
| `SentMessage(id, msg)` | Message record for every graph node |
| `ToolResult(id, fn_name, args)` | Marks a node as a tool result |
| `TenantId(id)` | Tenant identifier |

The `Message` record is
`[contents, tools_json, agent, agent_role, entity, principal]`.

### Output relations (your policy derives these)

| Relation | Description |
|----------|-------------|
| `IsAuthorized(idx)` | Whitelist — allow this action |
| `Unauthorized(idx)` | Blacklist — deny (overrides allow) |
| `DenialReason(idx, kind, reason, suggestion)` | Annotated deny. `kind` is `"block"` (hard deny) or `"ask"` (**soft deny**: not authorized to proceed autonomously — the engine downgrades it to a user prompt, precedence deny > ask > allow). Only block sets `Unauthorized`; ask is still a refusal to self-authorize, not an allow. Deployment-conditional — see SKILL.md step 5a. |
| `ApplyTransform(idx, id)` | Apply transform on authorization |
| `DenyUnauthorized(idx)` | Return error if not authorized |
| `AllowPassthrough(idx)` | Allow through unchanged if not authorized |

### Computed relations (from common_policy.dl)

| Relation | Description |
|----------|-------------|
| `Authorized(idx)` | Final decision: `HasPrincipal(), IsAuthorized(idx), !Unauthorized(idx)` |
| `CurrentDepends(src)` | Backward reachability from Current nodes |
| `HasRole(role)` | Does the authenticated principal have this role? |
| `HasPrincipal()` | Is there an authenticated principal? |
| `QueriesHost(a, host)` | Does HTTP action target this host? |
| `IsTool(a, name)` | Is this a specific tool call? |
| `IsToolCall(a)` | Is this any tool call? |

### C++ functor summary

```datalog
@str_contains(s, sub)         // → unsigned (1=match, 0=no)
@str_starts_with(s, prefix)   // → unsigned
@str_to_lower(s)              // → symbol
@json_get_str(json, field)    // → symbol (empty if not found)
@json_get_int(json, field)    // → number (0 if not found)
@json_has_field(json, field)  // → unsigned
@str_len(s)                   // → number
@url_host(url)                // → symbol (hostname)
@registrable_domain(url)      // → symbol (eTLD+1)
@count_urls(s)                // → number
@hours_between(t1, t2)        // → number

// Reaching INTO a JSON array. A top-level accessor cannot see array
// elements, so `@json_get_str(args, "flight_number")` on a payload
// shaped {"flights": [{...}]} silently returns "" and the rule that
// depends on it never fires.
@json_array_len(json, field)                     // → unsigned
@json_array_get_str(json, field, i)              // → symbol
@json_array_get_field_str(json, field, i, key)   // → symbol
@json_object_has_key(json, key)                  // → unsigned
@json_object_field_int(json, key)                // → number
@json_get_str_path(json, "a.b.c")                // → symbol
@json_get_int_path(json, "a.b.c")                // → number

// Shell-command analysis (Claude Code guard policies).
@normalize_cmd(cmd)           // → symbol
@cmd_runs_sub(cmd, sub)       // → unsigned
@cmd_invokes(cmd, name)       // → unsigned
@cmd_egress_class(cmd, tok)   // → unsigned (2=runs, 1=unprovable, 0=absent)
@dangerous_rm(cmd)            // → unsigned

// Hidden/deceptive Unicode.
@has_unicode_tags(s)          // → unsigned
@has_deceptive_unicode(s)     // → unsigned
@decode_unicode_tags(s)       // → symbol

@llm_check_fn(prompt, ctx)    // → unsigned (LLM semantic check —
                              //   deployment-conditional, see
                              //   SKILL.md step 5a; custom functors
                              //   preferred)
```

`common_policy.dl` is the authority on this list — read it (workflow
step 1) rather than trusting this summary to be current.

## Validation errors and what they mean

- **sugar.py errors** — bad dot notation (unknown field/type)
- **Soufflé parse errors** — syntax (missing periods, wrong
  operators)
- **Soufflé type errors** — mismatched types, wrong arity
- **Unknown relation** — undefined relation (check
  common_policy.dl)
- **Lint: `@deny(reason=...)`** — wrong annotation format; use
  separate `// @deny_message:` and `// @suggestion:` lines
  (see references/patterns.md)
- **Lint: nested functor calls** — bind intermediate results to
  variables
