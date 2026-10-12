# System profile (derived from traces)

Decision points: 76 · nodes: 289 · edges: 1593

## Agents and roles

| agent | role | messages | example node |
|---|---|---:|---|
| `LLMAgent` | `LLM` | 95 | `0dab25ef-6f29-485b-850c-92bdba679d76` |
| `LLMAgent` | `AGENT` | 60 | `dba9e9c7-0157-4be5-b443-9b20dcde7158` |
| `LLMAgent` | `USER` | 38 | `587a71de-7240-4dcf-99d4-a5216d929231` |
| `UserSimulator` | `LLM` | 38 | `fa87ff05-324d-4690-8f84-d2f09b860147` |
| `UserSimulator` | `USER` | 38 | `321efe49-dccc-47d4-ae4c-97fbcb72f822` |
| `LLMAgent` | `SYSTEM` | 10 | `8e16653d-1095-4766-9a35-ad79f36438e1` |
| `UserSimulator` | `SYSTEM` | 10 | `e2cbaf1f-1f85-41fa-848f-9945319d005c` |

## Role collisions

More than one agent name speaks with these roles. A rule keyed on the role alone will match all of them, so it needs an `msg.agent` filter too.

- role `LLM` — emitted by `LLMAgent`, `UserSimulator`
- role `SYSTEM` — emitted by `LLMAgent`, `UserSimulator`
- role `USER` — emitted by `LLMAgent`, `UserSimulator`

## Replicas

An agent name can cover several independent instances. `direct` joins same-named nodes that a dependency edge links directly; `reach` joins them when one is an ancestor of the other through any path. Where the two disagree, the system interleaves an agent's turns with foreign nodes and the choice of rule is load-bearing.

| agent | nodes | replicas (direct) | replicas (reach) | agree |
|---|---:|---:|---:|---|
| `LLMAgent` | 203 | 10 | 10 | yes |
| `UserSimulator` | 86 | 10 | 10 | yes |

## Tools

| tool | calls | results | argument fields | result fields |
|---|---:|---:|---|---|
| `get_flight_status` | 32 | 16 | date:str, flight_number:str | — |
| `get_reservation_details` | 26 | 13 | reservation_id:str | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:null, total_baggages:int, user_id:str |
| `cancel_reservation` | 26 | 4 | reservation_id:str | cabin:str, created_at:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_history:list, reservation_id:str, status:str, total_baggages:int, user_id:str |
| `get_user_details` | 24 | 12 | user_id:str | address:object, dob:str, email:str, membership:str, name:object, payment_methods:object, reservations:list, saved_passengers:list, user_id:str |
| `search_direct_flight` | 22 | 11 | date:str, destination:str, origin:str | — |
| `update_reservation_flights` | 6 | 0 | cabin:str, flights:list, payment_id:str, reservation_id:str | — |
| `search_onestop_flight` | 4 | 2 | date:str, destination:str, origin:str | — |
| `book_reservation` | 4 | 0 | cabin:str, destination:str, flight_type:str, flights:list, insurance:str, nonfree_baggages:int, origin:str, passengers:list, payment_methods:list, total_baggages:int, user_id:str | — |
| `update_reservation_passengers` | 2 | 1 | passengers:list, reservation_id:str | — |
| `calculate` | 2 | 1 | expression:str | — |
| `transfer_to_human_agents` | 2 | 0 | summary:str | — |
| `send_certificate` | 2 | 0 | amount:int, user_id:str | — |

## Slice shapes per gated tool

| tool | decision points | nodes (min/mean/max) | agents in slice (max) |
|---|---:|---|---:|
| `get_flight_status` | 16 | 19/29.7/43 | 2 |
| `get_reservation_details` | 13 | 7/15.6/31 | 2 |
| `cancel_reservation` | 13 | 13/16.1/25 | 2 |
| `get_user_details` | 12 | 7/17.0/31 | 2 |
| `search_direct_flight` | 11 | 13/19.2/35 | 2 |
| `update_reservation_flights` | 3 | 21/32.3/43 | 2 |
| `search_onestop_flight` | 2 | 15/20.0/25 | 2 |
| `book_reservation` | 2 | 23/26.0/29 | 2 |
| `update_reservation_passengers` | 1 | 13/13.0/13 | 2 |
| `calculate` | 1 | 41/41.0/41 | 2 |
| `transfer_to_human_agents` | 1 | 53/53.0/53 | 2 |
| `send_certificate` | 1 | 42/42.0/42 | 2 |

## Warnings

- role `LLM` is emitted by 2 agent names (LLMAgent, UserSimulator) — rules matching this role need an msg.agent filter
- role `SYSTEM` is emitted by 2 agent names (LLMAgent, UserSimulator) — rules matching this role need an msg.agent filter
- role `USER` is emitted by 2 agent names (LLMAgent, UserSimulator) — rules matching this role need an msg.agent filter
- agent `LLMAgent` has 10 independent replicas — a rule that treats the name as one instance will cross-contaminate them
- agent `UserSimulator` has 10 independent replicas — a rule that treats the name as one instance will cross-contaminate them
- tool `get_flight_status` returns no parseable JSON object — field accessors will not work on its result
- tool `search_direct_flight` returns no parseable JSON object — field accessors will not work on its result
- tool `update_reservation_flights` was called 6x but no result node carries its provenance — rules joining on ToolResult for it cannot fire
- tool `search_onestop_flight` returns no parseable JSON object — field accessors will not work on its result
- tool `book_reservation` was called 4x but no result node carries its provenance — rules joining on ToolResult for it cannot fire
- tool `update_reservation_passengers` returns no parseable JSON object — field accessors will not work on its result
- tool `calculate` returns no parseable JSON object — field accessors will not work on its result
- tool `transfer_to_human_agents` was called 2x but no result node carries its provenance — rules joining on ToolResult for it cannot fire
- tool `send_certificate` was called 2x but no result node carries its provenance — rules joining on ToolResult for it cannot fire
