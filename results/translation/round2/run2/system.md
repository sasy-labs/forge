# System profile (derived from traces)

Decision points: 74 · nodes: 311 · edges: 1609

## Agents and roles

| agent | role | messages | example node |
|---|---|---:|---|
| `LLMAgent` | `LLM` | 101 | `3db23ecd-2af7-4847-a80e-842a5a42d333` |
| `LLMAgent` | `AGENT` | 58 | `f5561251-ae11-4357-9dae-71cdbf108d27` |
| `LLMAgent` | `USER` | 44 | `e0318884-3c7a-4107-a4b3-39690194cca5` |
| `UserSimulator` | `LLM` | 44 | `ee82ca04-bf5e-4d77-bf74-ce4119f63b93` |
| `UserSimulator` | `USER` | 44 | `e9c9a779-2c5e-4295-b843-b71ce5fa4f31` |
| `LLMAgent` | `SYSTEM` | 10 | `8293ba1a-c5b9-473e-93b1-478c23f5d74d` |
| `UserSimulator` | `SYSTEM` | 10 | `71d1fd66-6799-4a39-ae55-642af4e4c018` |

## Role collisions

More than one agent name speaks with these roles. A rule keyed on the role alone will match all of them, so it needs an `msg.agent` filter too.

- role `LLM` — emitted by `LLMAgent`, `UserSimulator`
- role `SYSTEM` — emitted by `LLMAgent`, `UserSimulator`
- role `USER` — emitted by `LLMAgent`, `UserSimulator`

## Replicas

An agent name can cover several independent instances. `direct` joins same-named nodes that a dependency edge links directly; `reach` joins them when one is an ancestor of the other through any path. Where the two disagree, the system interleaves an agent's turns with foreign nodes and the choice of rule is load-bearing.

| agent | nodes | replicas (direct) | replicas (reach) | agree |
|---|---:|---:|---:|---|
| `LLMAgent` | 213 | 10 | 10 | yes |
| `UserSimulator` | 98 | 10 | 10 | yes |

## Tools

| tool | calls | results | argument fields | result fields |
|---|---:|---:|---|---|
| `get_reservation_details` | 34 | 17 | reservation_id:str | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:null, total_baggages:int, user_id:str |
| `get_user_details` | 28 | 14 | user_id:str | address:object, dob:str, email:str, membership:str, name:object, payment_methods:object, reservations:list, saved_passengers:list, user_id:str |
| `search_direct_flight` | 18 | 9 | date:str, destination:str, origin:str | — |
| `cancel_reservation` | 18 | 1 | reservation_id:str | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:str, total_baggages:int, user_id:str |
| `get_flight_status` | 10 | 5 | date:str, flight_number:str | — |
| `update_reservation_passengers` | 6 | 3 | passengers:list, reservation_id:str | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:null, total_baggages:int, user_id:str |
| `update_reservation_baggages` | 6 | 2 | nonfree_baggages:int, payment_id:str, reservation_id:str, total_baggages:int | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:null, total_baggages:int, user_id:str |
| `calculate` | 6 | 2 | expression:str | — |
| `search_onestop_flight` | 6 | 3 | date:str, destination:str, origin:str | — |
| `book_reservation` | 6 | 1 | cabin:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_methods:list, total_baggages:int, user_id:str | — |
| `send_certificate` | 4 | 0 | amount:int, user_id:str | — |
| `update_reservation_flights` | 4 | 1 | cabin:str, flights:list, payment_id:str, reservation_id:str | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:null, total_baggages:int, user_id:str |
| `transfer_to_human_agents` | 2 | 0 | summary:str | — |

## Slice shapes per gated tool

| tool | decision points | nodes (min/mean/max) | agents in slice (max) |
|---|---:|---|---:|
| `get_reservation_details` | 17 | 7/18.8/33 | 2 |
| `get_user_details` | 14 | 7/14.7/33 | 2 |
| `search_direct_flight` | 9 | 15/22.6/29 | 2 |
| `cancel_reservation` | 9 | 13/15.0/17 | 2 |
| `get_flight_status` | 5 | 7/26.2/31 | 2 |
| `update_reservation_passengers` | 3 | 11/20.3/25 | 2 |
| `update_reservation_baggages` | 3 | 25/35.0/41 | 2 |
| `calculate` | 3 | 19/26.3/41 | 2 |
| `search_onestop_flight` | 3 | 27/27.7/28 | 2 |
| `book_reservation` | 3 | 31/35.7/41 | 2 |
| `send_certificate` | 2 | 31/37.5/44 | 2 |
| `update_reservation_flights` | 2 | 25/28.0/31 | 2 |
| `transfer_to_human_agents` | 1 | 23/23.0/23 | 2 |

## Warnings

- role `LLM` is emitted by 2 agent names (LLMAgent, UserSimulator) — rules matching this role need an msg.agent filter
- role `SYSTEM` is emitted by 2 agent names (LLMAgent, UserSimulator) — rules matching this role need an msg.agent filter
- role `USER` is emitted by 2 agent names (LLMAgent, UserSimulator) — rules matching this role need an msg.agent filter
- agent `LLMAgent` has 10 independent replicas — a rule that treats the name as one instance will cross-contaminate them
- agent `UserSimulator` has 10 independent replicas — a rule that treats the name as one instance will cross-contaminate them
- tool `search_direct_flight` returns no parseable JSON object — field accessors will not work on its result
- tool `get_flight_status` returns no parseable JSON object — field accessors will not work on its result
- tool `calculate` returns no parseable JSON object — field accessors will not work on its result
- tool `search_onestop_flight` returns no parseable JSON object — field accessors will not work on its result
- tool `book_reservation` returns no parseable JSON object — field accessors will not work on its result
- tool `send_certificate` was called 4x but no result node carries its provenance — rules joining on ToolResult for it cannot fire
- tool `transfer_to_human_agents` was called 2x but no result node carries its provenance — rules joining on ToolResult for it cannot fire
