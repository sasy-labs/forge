# Policy Pattern Library

Match each spec clause to one or more of these patterns. Every
generated rule must cite its clause: `/* C_n: ... */`.

## Pattern 1: Role-Based Access Control (RBAC)

**English:** "Users with role X can access service Y"

```datalog
IsAuthorized(idx) :-
    Actions(idx, a),
    QueriesHost(a, "api.example.com"),
    HasRole("role-name").
```

## Pattern 2: Blacklist (Deny Rule)

**English:** "Block all access to X"

```datalog
Unauthorized(idx) :-
    Actions(idx, a),
    QueriesHost(a, "blocked.api.com").
```

`Unauthorized` overrides any `IsAuthorized` rules.

**Annotated deny with reason** — annotation comments on separate
lines above an `Unauthorized` rule; sugar.py rewrites the head into
`DenialReason(idx, kind, reason, suggestion)` and common_policy derives
`Unauthorized` automatically:

```datalog
// @deny_message: Blocked API
// @suggestion: Use the approved endpoint instead
Unauthorized(idx) :-
    Actions(idx, a),
    QueriesHost(a, "blocked.api.com").
```

Annotation keys: `@deny_message:` (or `@reason:`, `@message:`),
`@suggestion:` (or `@fix:`, `@hint:`), `@tool_pattern:` — each on
its own `//` comment line.

**IMPORTANT:** Do NOT use `@deny(reason="...", suggestion="...")` —
that format is not supported by sugar.py (the lint validator
rejects it).

## Pattern 3: Tool-Based Authorization

**English:** "Only role Y can use tool X"

```datalog
IsAuthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "tool_name"),
    HasRole("role-name").

DenyUnauthorized(idx) :- Actions(idx, a), IsTool(a, "tool_name").
```

## Pattern 4: Argument-Based Conditions

**English:** "Allow X only if argument Y has value Z"

```datalog
IsAuthorized(idx) :-
    Actions(idx, a),
    a = $CallTool(_, args),
    IsTool(a, "send_email"),
    recipient = @json_get_str(args, "to"),
    @str_contains(recipient, "@company.com") = 1.
```

## Pattern 5: Graph-Based Conditional Access

**English:** "Allow X only if Y happened earlier in this
conversation"

```datalog
// Define what constitutes approval
.decl ApprovesAction(id: symbol)
ApprovesAction(id) :-
    SentMessage(id, msg),
    @str_contains(msg.contents, "approved") = 1.

// Allow action only if current node depends on approval
IsAuthorized(idx) :-
    Actions(idx, a),
    QueriesHost(a, "api.example.com"),
    Current(id),
    CurrentDepends(approval_id),
    ApprovesAction(approval_id).
```

Use `CurrentDepends(src)` instead of `Depends(dest, src)` — it's
backward reachability from Current nodes only, much cheaper.

## Pattern 6: Taint Tracking (Toxic Flow Detection)

**English:** "Block action if data came from untrusted source"

```datalog
.decl TaintedByUntrusted(id: symbol)
TaintedByUntrusted(id) :-
    ToolResult(id, "read_file", args),
    @str_contains(args, "/untrusted/") = 1.
TaintedByUntrusted(id) :-
    TaintedByUntrusted(src),
    Edge(src, id).

Unauthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "send_email"),
    Current(id),
    CurrentDepends(src),
    TaintedByUntrusted(src).
```

## Pattern 7: Credential Injection Transform

**English:** "When request X is authorized, inject credentials for
service Y"

```datalog
ApplyTransform(idx, "inject_api_key") :-
    Actions(idx, a),
    QueriesHost(a, "api.example.com"),
    Authorized(idx).
```

Transform IDs must match entries in `config/transforms.json`.

## Pattern 8: Message Content Checks

**English:** "Block action if user mentioned keyword X"

```datalog
Unauthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "cancel_reservation"),
    Current(id),
    CurrentDepends(msg_id),
    SentMessage(msg_id, msg),
    msg.agent = "MainAgent",
    msg.agent_role = $UserRole(),
    @str_contains(msg.contents, "accidentally") = 1.
```

**Always pair content checks with the agent/role filters from your
K-constraints** (see the core workflow, step 4): in multi-party
systems a `$UserRole()` message can come from a simulator or
another agent — filter on `msg.agent` too, or the rule matches
messages it must not.

## Clause ordering for expensive functors

Soufflé evaluates functors when their arguments become bound. Place
cheap relational filters before expensive functors to narrow the
tuple set:

```datalog
// GOOD: IsTool narrows to send_email before the check fires
Unauthorized(idx) :-
    Actions(idx, a),
    IsTool(a, "send_email"),
    a = $CallTool(_, args),
    @my_check(args) = 1.
```
