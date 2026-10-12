# System profile (derived from traces)

Decision points: 95 · nodes: 397 · edges: 2550

## Agents and roles

| agent | role | messages | example node |
|---|---|---:|---|
| `LLMAgent` | `LLM` | 126 | `bf5046ae-1bfe-41de-a743-6c7f1640acbc` |
| `LLMAgent` | `AGENT` | 80 | `bcc75f27-acf1-4d37-8f00-a23091bf5929` |
| `LLMAgent` | `USER` | 57 | `00d775ca-8a7a-4b10-8715-e120f712b070` |
| `UserSimulator` | `LLM` | 57 | `5debfc57-40d5-4066-bc55-b96a95614456` |
| `UserSimulator` | `USER` | 57 | `245c98f3-2cc6-4e1b-8fc9-651ad569e4ab` |
| `LLMAgent` | `SYSTEM` | 10 | `ca3ab416-c4ea-4bdc-8b31-7aa3071b6f1f` |
| `UserSimulator` | `SYSTEM` | 10 | `3f8b1e87-213a-4d81-9e4b-c6735137bbe2` |

## Role collisions

More than one agent name speaks with these roles. A rule keyed on the role alone will match all of them, so it needs an `msg.agent` filter too.

- role `LLM` — emitted by `LLMAgent`, `UserSimulator`
- role `SYSTEM` — emitted by `LLMAgent`, `UserSimulator`
- role `USER` — emitted by `LLMAgent`, `UserSimulator`

## Replicas

An agent name can cover several independent instances. `direct` joins same-named nodes that a dependency edge links directly; `reach` joins them when one is an ancestor of the other through any path. Where the two disagree, the system interleaves an agent's turns with foreign nodes and the choice of rule is load-bearing.

| agent | nodes | replicas (direct) | replicas (reach) | agree |
|---|---:|---:|---:|---|
| `LLMAgent` | 273 | 10 | 10 | yes |
| `UserSimulator` | 124 | 10 | 10 | yes |

## Tools

| tool | calls | results | argument fields | result fields |
|---|---:|---:|---|---|
| `get_reservation_details` | 52 | 20 | reservation_id:str | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:null, total_baggages:int, user_id:str |
| `get_user_details` | 32 | 16 | user_id:str | address:object, dob:str, email:str, membership:str, name:object, payment_methods:object, reservations:list, saved_passengers:list, user_id:str |
| `search_direct_flight` | 24 | 12 | date:str, destination:str, origin:str | — |
| `calculate` | 22 | 10 | expression:str | — |
| `update_reservation_flights` | 20 | 7 | cabin:str, flights:list, payment_id:str, reservation_id:str | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:null, total_baggages:int, user_id:str |
| `cancel_reservation` | 12 | 6 | reservation_id:str | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:str, total_baggages:int, user_id:str |
| `search_onestop_flight` | 8 | 4 | date:str, destination:str, origin:str | — |
| `update_reservation_baggages` | 8 | 2 | nonfree_baggages:int, payment_id:str, reservation_id:str, total_baggages:int | — |
| `transfer_to_human_agents` | 4 | 0 | summary:str | — |
| `book_reservation` | 4 | 1 | cabin:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_methods:list, total_baggages:int, user_id:str | — |
| `get_flight_status` | 2 | 1 | date:str, flight_number:str | — |
| `update_reservation_passengers` | 2 | 1 | passengers:list, reservation_id:str | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:null, total_baggages:int, user_id:str |

## Slice shapes per gated tool

| tool | decision points | nodes (min/mean/max) | agents in slice (max) |
|---|---:|---|---:|
| `get_reservation_details` | 26 | 13/18.8/39 | 2 |
| `get_user_details` | 16 | 7/14.2/35 | 2 |
| `search_direct_flight` | 12 | 17/23.8/37 | 2 |
| `calculate` | 11 | 19/29.2/45 | 2 |
| `update_reservation_flights` | 10 | 25/39.3/64 | 2 |
| `cancel_reservation` | 6 | 7/12.3/15 | 2 |
| `search_onestop_flight` | 4 | 28/28.0/28 | 2 |
| `update_reservation_baggages` | 4 | 25/33.2/43 | 2 |
| `transfer_to_human_agents` | 2 | 41/53.5/66 | 2 |
| `book_reservation` | 2 | 31/34.0/37 | 2 |
| `get_flight_status` | 1 | 27/27.0/27 | 2 |
| `update_reservation_passengers` | 1 | 25/25.0/25 | 2 |

## Warnings

- role `LLM` is emitted by 2 agent names (LLMAgent, UserSimulator) — rules matching this role need an msg.agent filter
- role `SYSTEM` is emitted by 2 agent names (LLMAgent, UserSimulator) — rules matching this role need an msg.agent filter
- role `USER` is emitted by 2 agent names (LLMAgent, UserSimulator) — rules matching this role need an msg.agent filter
- agent `LLMAgent` has 10 independent replicas — a rule that treats the name as one instance will cross-contaminate them
- agent `UserSimulator` has 10 independent replicas — a rule that treats the name as one instance will cross-contaminate them
- tool `search_direct_flight` returns no parseable JSON object — field accessors will not work on its result
- tool `calculate` returns no parseable JSON object — field accessors will not work on its result
- tool `search_onestop_flight` returns no parseable JSON object — field accessors will not work on its result
- tool `update_reservation_baggages` returns no parseable JSON object — field accessors will not work on its result
- tool `transfer_to_human_agents` was called 4x but no result node carries its provenance — rules joining on ToolResult for it cannot fire
- tool `book_reservation` returns no parseable JSON object — field accessors will not work on its result
- tool `get_flight_status` returns no parseable JSON object — field accessors will not work on its result
