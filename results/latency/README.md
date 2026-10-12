# Latency results

The experiments measure authorization latency and throughput at concurrency 1, 2, 4, 8, 16 and 32.

- **`live/`**: an airline evaluation using GPT-5.2 as agent and Opus 4.5 as user across eight tasks. `results.csv` summarizes the complete sweep. `bootstrap_summary.json` reports client-side and server-side timings with whole-session resampling within task strata (2,000 replicates, seed 42).
- **`replay/`**: a queued replay of the live concurrency-32 capture: 998 calls from 128 sessions, repeated 100 times per trial, with five trials per concurrency. All 2,994,000 decisions and per-session tool order matched the source.

Replay measures authorization against the saved graph, excluding LLM calls, tool execution and graph updates. Throughput includes dispatch and output writes; startup, warmup and cleanup are excluded. Client round-trip latency is recorded separately. Server timing logs include warmup records, which must be excluded when computing replay percentiles. `replay/live-results.*` summarizes the live concurrency-32 source.

Manifests record trial counts, source paths and file hashes. The saved results include timing logs and trial receipts; graph stores, credentials and conversations are omitted. To inspect a compressed replay log from this directory:

```bash
gzip -dc replay/c32/loadgen-trial-1.jsonl.gz
```
