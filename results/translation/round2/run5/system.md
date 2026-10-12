# System profile (derived from traces)

Decision points: 88 · nodes: 340 · edges: 2290

## Agents and roles

| agent | role | messages | example node |
|---|---|---:|---|
| `LLMAgent` | `LLM` | 109 | `75f0536e-5607-41f1-8d07-c43d43293329` |
| `LLMAgent` | `AGENT` | 76 | `5840ef33-804e-4b11-a75d-17300f28465d` |
| `LLMAgent` | `USER` | 45 | `7793eb2b-a90e-44d9-9e2e-a0aa24654f29` |
| `UserSimulator` | `LLM` | 45 | `633a4b19-2806-42f7-a74c-51adaac102cc` |
| `UserSimulator` | `USER` | 45 | `ef5882f4-45fe-499b-88ee-0d760036cc12` |
| `LLMAgent` | `SYSTEM` | 10 | `33d903b1-4194-48b0-896a-2f4437480311` |
| `UserSimulator` | `SYSTEM` | 10 | `b48cc217-0b30-47ba-bb47-34850a6474da` |

## Role collisions

More than one agent name speaks with these roles. A rule keyed on the role alone will match all of them, so it needs an `msg.agent` filter too.

- role `LLM` — emitted by `LLMAgent`, `UserSimulator`
- role `SYSTEM` — emitted by `LLMAgent`, `UserSimulator`
- role `USER` — emitted by `LLMAgent`, `UserSimulator`

## Replicas

An agent name can cover several independent instances. `direct` joins same-named nodes that a dependency edge links directly; `reach` joins them when one is an ancestor of the other through any path. Where the two disagree, the system interleaves an agent's turns with foreign nodes and the choice of rule is load-bearing.

| agent | nodes | replicas (direct) | replicas (reach) | agree |
|---|---:|---:|---:|---|
| `LLMAgent` | 240 | 10 | 10 | yes |
| `UserSimulator` | 100 | 10 | 10 | yes |

## Tools

| tool | calls | results | argument fields | result fields |
|---|---:|---:|---|---|
| `get_reservation_details` | 56 | 28 | reservation_id:str | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:null, total_baggages:int, user_id:str |
| `get_user_details` | 20 | 10 | user_id:str | address:object, dob:str, email:str, membership:str, name:object, payment_methods:object, reservations:list, saved_passengers:list, user_id:str |
| `update_reservation_flights` | 20 | 5 | cabin:str, flights:list, payment_id:str, reservation_id:str | — |
| `calculate` | 20 | 10 | expression:str | — |
| `search_direct_flight` | 18 | 9 | date:str, destination:str, origin:str | — |
| `book_reservation` | 14 | 6 | cabin:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_methods:list, total_baggages:int, user_id:str | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:null, total_baggages:int, user_id:str |
| `update_reservation_passengers` | 8 | 3 | passengers:list, reservation_id:str | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:null, total_baggages:int, user_id:str |
| `search_onestop_flight` | 6 | 3 | date:str, destination:str, origin:str | — |
| `cancel_reservation` | 6 | 1 | reservation_id:str | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:str, total_baggages:int, user_id:str |
| `transfer_to_human_agents` | 4 | 0 | summary:str | — |
| `update_reservation_baggages` | 4 | 1 | nonfree_baggages:int, payment_id:str, reservation_id:str, total_baggages:int | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:null, total_baggages:int, user_id:str |

## Slice shapes per gated tool

| tool | decision points | nodes (min/mean/max) | agents in slice (max) |
|---|---:|---|---:|
| `get_reservation_details` | 28 | 11/16.5/31 | 2 |
| `get_user_details` | 10 | 7/11.4/17 | 2 |
| `update_reservation_flights` | 10 | 21/36.2/48 | 2 |
| `calculate` | 10 | 20/50.6/70 | 2 |
| `search_direct_flight` | 9 | 13/19.4/37 | 2 |
| `book_reservation` | 7 | 46/64.3/84 | 2 |
| `update_reservation_passengers` | 4 | 13/25.8/33 | 2 |
| `search_onestop_flight` | 3 | 15/16.3/17 | 2 |
| `cancel_reservation` | 3 | 31/35.3/44 | 2 |
| `transfer_to_human_agents` | 2 | 23/28.0/33 | 2 |
| `update_reservation_baggages` | 2 | 21/27.5/34 | 2 |

## Warnings

- role `LLM` is emitted by 2 agent names (LLMAgent, UserSimulator) — rules matching this role need an msg.agent filter
- role `SYSTEM` is emitted by 2 agent names (LLMAgent, UserSimulator) — rules matching this role need an msg.agent filter
- role `USER` is emitted by 2 agent names (LLMAgent, UserSimulator) — rules matching this role need an msg.agent filter
- agent `LLMAgent` has 10 independent replicas — a rule that treats the name as one instance will cross-contaminate them
- agent `UserSimulator` has 10 independent replicas — a rule that treats the name as one instance will cross-contaminate them
- tool `update_reservation_flights` returns no parseable JSON object — field accessors will not work on its result
- tool `calculate` returns no parseable JSON object — field accessors will not work on its result
- tool `search_direct_flight` returns no parseable JSON object — field accessors will not work on its result
- tool `search_onestop_flight` returns no parseable JSON object — field accessors will not work on its result
- tool `transfer_to_human_agents` was called 4x but no result node carries its provenance — rules joining on ToolResult for it cannot fire
