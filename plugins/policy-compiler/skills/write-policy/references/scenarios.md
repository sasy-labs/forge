# Test Scenario Format

Scenarios are JSON files run offline by
`scripts/policy_test_harness.py` (Soufflé interpreted mode, no
server). If the policy is at `experiments/my_policy.dl`, save scenarios
to `experiments/my_policy_tests.json`.

## File format

```json
{
  "policy_description": "Only admins can delete users",
  "scenarios": [
    {
      "name": "admin can delete users",
      "clause": "C1",
      "entity": "alice",
      "roles": ["admin"],
      "action": {
        "type": "CallTool",
        "fn_name": "delete_user",
        "args": "{\"user_id\": \"bob\"}"
      },
      "graph": {
        "current_nodes": [],
        "edges": [],
        "sent_messages": [],
        "tool_results": []
      },
      "expected": {"authorized": true}
    }
  ]
}
```

## Identity fields

- `entity` — caller-supplied actor (maps to the `Entity` fact)
- `roles` — bound to the principal via `PrincipalRole`
- `principal` — optional; defaults to `entity`. The final
  `Authorized` decision requires `HasPrincipal()`, so leave
  `entity` (or `principal`) non-empty in any scenario expected to
  be authorized.

## Action types

- **CallTool:** `{"type": "CallTool", "fn_name": "...", "args": "{...}"}`
- **HTTPRequest:** `{"type": "HTTPRequest", "url": "...", "body": "...", "headers_json": "[]"}`
- **SendAttempt:** `{"type": "SendAttempt", "contents": "...", "agent": "...", "agent_role": "assistant", "entity": "..."}`

## Expected fields

| Field | Type | Meaning |
|-------|------|---------|
| `authorized` | bool | Action index 0 is in `Authorized` |
| `unauthorized` | bool | Any tuple in `Unauthorized` |
| `is_authorized` | bool | Any tuple in `IsAuthorized` |
| `deny_unauthorized` | bool | Action index 0 in `DenyUnauthorized` |
| `allow_passthrough` | bool | Action index 0 in `AllowPassthrough` |
| `transforms` | list[str] | Transform IDs applied to action 0 |

`expected: null` means "no assertions yet" — the harness loads but
asserts nothing (used by captured scenarios awaiting oracle labels).

## Graph-dependent scenarios

For policies using `CurrentDepends`, `SentMessage`, `ToolResult`,
or `Edge`, populate the `graph` field:

```json
"graph": {
  "current_nodes": ["msg-3"],
  "edges": [["msg-1", "msg-2"], ["msg-2", "msg-3"]],
  "edges_data": [
    {"source": "msg-1", "destination": "msg-3",
     "message_index": 0, "proximal": true}
  ],
  "sent_messages": [
    {"id": "msg-1", "contents": "approved", "agent": "Manager",
     "agent_role": "user", "entity": "alice@co.com"}
  ],
  "tool_results": [
    {"id": "msg-2", "fn_name": "get_reservation_details",
     "args": "{\"reservation_id\": \"X\"}"}
  ]
}
```

Notes that mirror how the live system records graphs:

- **Every node is a `SentMessage`** — a tool-result node appears in
  BOTH `tool_results` (marking which tool produced it) and
  `sent_messages` (its `contents` carry the result payload that
  content rules read).
- `agent_role` values: `system`, `user`, `agent`, `assistant`.
- `sent_messages[].tools_json` (optional) carries the JSON list of
  tool calls the message proposed.
- `edges_data` is optional sparse metadata (`message_index`,
  `proximal`); omit entries for plain edges.

## Scenario sources and labels

A scenario file can mix three label provenances (used by
`experiments/translation/policy_convergence.py`):

1. **Author-written** — you write `expected` directly (and may add
   `"label": {"source": "author", ...}`); trusted.
2. **Ground-truth** — derived from scripted attack/benign runs;
   `"label": {"source": "ground-truth", ...}`; trusted.
3. **Oracle-labeled** — produced by `experiments/translation/oracle_label.py`;
   trusted unless `oracle_weak: true` (advisory — see
   references/trace-loop.md for the dispute protocol).

## Running

```bash
make policy-test POLICY=<path> SCENARIOS=<path>
# or
python3 scripts/policy_test_harness.py --policy <p> --scenarios <s>
```

Exit 0 = all pass. Run it yourself and show the results — never
just tell the user how to run it.

## Authoring guidance

- ≥1 positive and ≥1 negative scenario per clause C_n.
- For every content/graph rule, include a near-miss negative (e.g.
  the confirmation text present but attributed to the wrong
  `msg.agent`).
- Captured scenarios (Mode L) cover real traffic shapes; author
  top-ups for clauses capture didn't exercise — especially
  role-gated clauses, which captured traces cannot test (the
  capture can't see server-side roles; such scenarios carry
  `provenance.roles_unknown: true`).
