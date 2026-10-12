## airline

### Client-side authorization latency

Per-call wall time as the agent observes it: SDK serialization + wire + server processing + response deserialization. Includes any GIL / channel-pool contention on the client.

| mode | conc | n | p50 | mean | p95 | p99 | max |
|------|-----:|--:|----:|-----:|----:|----:|----:|
| enforce-on | 32 | 998 | 719us | 5.69ms | 29.41ms | 58.51ms | 810.07ms |

### End-to-end throughput

| mode | conc | tasks | wall (s) | tasks/s | task mean (s) | task stdev (s) |
|------|-----:|------:|---------:|--------:|--------------:|---------------:|
| enforce-on | 32 | 128 | 381.2 | 0.336 | 84.8 | 21.0 |

### Server-side authorization latency

Per-call ``total_us`` from the engine's JSONL log captured while the cell ran. This is the real server cost under the actual concurrency the agent drove — sits between the client-side row (client + wire + server) and the replay row (server-only at saturation).

| mode | conc | n | p50 | mean | p95 | p99 | max |
|------|-----:|--:|----:|-----:|----:|----:|----:|
| enforce-on | 32 | 998 | 187us | 418us | 1.59ms | 2.63ms | 3.46ms |
