# System profile (derived from traces)

Decision points: 64 · nodes: 306 · edges: 1550

## Agents and roles

| agent | role | messages | example node |
|---|---|---:|---|
| `LLMAgent` | `LLM` | 95 | `098d020d-b15e-4499-b012-be9806f6cacb` |
| `LLMAgent` | `AGENT` | 53 | `cb4d93d2-982f-4972-ad21-e232684671aa` |
| `LLMAgent` | `USER` | 46 | `d772a48e-11b3-4745-9b0c-1cd4c58719de` |
| `UserSimulator` | `LLM` | 46 | `387012b8-a2ca-47cf-9791-c7e738361768` |
| `UserSimulator` | `USER` | 46 | `dcd2b7c8-68b6-4da8-a09d-12c06c05fa7d` |
| `LLMAgent` | `SYSTEM` | 10 | `39465bc2-7ac6-4df5-9ec5-6ec830617c21` |
| `UserSimulator` | `SYSTEM` | 10 | `10168585-3004-499c-a7cf-55da4f3ddcb1` |

## Role collisions

More than one agent name speaks with these roles. A rule keyed on the role alone will match all of them, so it needs an `msg.agent` filter too.

- role `LLM` — emitted by `LLMAgent`, `UserSimulator`
- role `SYSTEM` — emitted by `LLMAgent`, `UserSimulator`
- role `USER` — emitted by `LLMAgent`, `UserSimulator`

## Replicas

An agent name can cover several independent instances. `direct` joins same-named nodes that a dependency edge links directly; `reach` joins them when one is an ancestor of the other through any path. Where the two disagree, the system interleaves an agent's turns with foreign nodes and the choice of rule is load-bearing.

| agent | nodes | replicas (direct) | replicas (reach) | agree |
|---|---:|---:|---:|---|
| `LLMAgent` | 204 | 10 | 10 | yes |
| `UserSimulator` | 102 | 10 | 10 | yes |

## Tools

| tool | calls | results | argument fields | result fields |
|---|---:|---:|---|---|
| `get_reservation_details` | 52 | 26 | reservation_id:str | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:null, total_baggages:int, user_id:str |
| `get_user_details` | 20 | 10 | user_id:str | address:object, dob:str, email:str, membership:str, name:object, payment_methods:object, reservations:list, saved_passengers:list, user_id:str |
| `search_direct_flight` | 12 | 4 | date:str, destination:str, origin:str | — |
| `book_reservation` | 10 | 3 | cabin:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_methods:list, total_baggages:int, user_id:str | — |
| `search_onestop_flight` | 10 | 5 | date:str, destination:str, origin:str | — |
| `cancel_reservation` | 8 | 1 | reservation_id:str | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:str, total_baggages:int, user_id:str |
| `calculate` | 8 | 3 | expression:str | — |
| `get_flight_status` | 2 | 1 | date:str, flight_number:str | — |
| `send_certificate` | 2 | 0 | amount:int, user_id:str | — |
| `transfer_to_human_agents` | 2 | 0 | summary:str | — |
| `update_reservation_flights` | 2 | 0 | cabin:str, flights:list, payment_id:str, reservation_id:str | — |

## Slice shapes per gated tool

| tool | decision points | nodes (min/mean/max) | agents in slice (max) |
|---|---:|---|---:|
| `get_reservation_details` | 26 | 11/17.2/31 | 2 |
| `get_user_details` | 10 | 11/13.6/21 | 2 |
| `search_direct_flight` | 6 | 15/23.3/29 | 2 |
| `book_reservation` | 5 | 17/31.4/45 | 2 |
| `search_onestop_flight` | 5 | 15/26.6/41 | 2 |
| `cancel_reservation` | 4 | 11/25.2/33 | 2 |
| `calculate` | 4 | 25/34.0/43 | 2 |
| `get_flight_status` | 1 | 11/11.0/11 | 2 |
| `send_certificate` | 1 | 39/39.0/39 | 2 |
| `transfer_to_human_agents` | 1 | 37/37.0/37 | 2 |
| `update_reservation_flights` | 1 | 22/22.0/22 | 2 |

## Warnings

- role `LLM` is emitted by 2 agent names (LLMAgent, UserSimulator) — rules matching this role need an msg.agent filter
- role `SYSTEM` is emitted by 2 agent names (LLMAgent, UserSimulator) — rules matching this role need an msg.agent filter
- role `USER` is emitted by 2 agent names (LLMAgent, UserSimulator) — rules matching this role need an msg.agent filter
- agent `LLMAgent` has 10 independent replicas — a rule that treats the name as one instance will cross-contaminate them
- agent `UserSimulator` has 10 independent replicas — a rule that treats the name as one instance will cross-contaminate them
- tool `search_direct_flight` returns no parseable JSON object — field accessors will not work on its result
- tool `book_reservation` returns no parseable JSON object — field accessors will not work on its result
- tool `search_onestop_flight` returns no parseable JSON object — field accessors will not work on its result
- tool `calculate` returns no parseable JSON object — field accessors will not work on its result
- tool `get_flight_status` returns no parseable JSON object — field accessors will not work on its result
- tool `send_certificate` was called 2x but no result node carries its provenance — rules joining on ToolResult for it cannot fire
- tool `transfer_to_human_agents` was called 2x but no result node carries its provenance — rules joining on ToolResult for it cannot fire
- tool `update_reservation_flights` was called 2x but no result node carries its provenance — rules joining on ToolResult for it cannot fire
