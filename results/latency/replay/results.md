# Authorization replay from saved benchmark records

These measurements replay tool-authorization requests against a copy of the persisted graph at benchmark stop. Sessions are warmed before timing. They exclude LLM calls, tool execution, SDK overhead, and historical graph updates; they do not measure end-to-end task throughput.

Throughput is calls divided by the monotonic timed window, including dispatch and output writes and excluding startup, warmup, and cleanup. Repetitions use the same saved graph.

Original live timings and task durations are in `live-results.csv` and `live-results.md`, at the source run's concurrency.

| source cell | replay concurrency | trials | calls | p50 | p99 | calls/s | status |
|---|---:|---:|---:|---:|---:|---:|---:|
| tau2/airline_enforce-on_gpt | 1 | 5 | 499000 | 203us | 1.26ms | 2910 | 0 |
| tau2/airline_enforce-on_gpt | 2 | 5 | 499000 | 206us | 1.35ms | 5582 | 0 |
| tau2/airline_enforce-on_gpt | 4 | 5 | 499000 | 228us | 1.44ms | 10426 | 0 |
| tau2/airline_enforce-on_gpt | 8 | 5 | 499000 | 300us | 1.50ms | 17527 | 0 |
| tau2/airline_enforce-on_gpt | 16 | 5 | 499000 | 509us | 1.79ms | 22692 | 0 |
| tau2/airline_enforce-on_gpt | 32 | 5 | 499000 | 803us | 2.75ms | 23935 | 0 |
