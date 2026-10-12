## AgentDojo Results: gpt-4o-2024-08-06 / tool_knowledge

Domains: banking, slack, travel, workspace
Matched runs: 97 clean + 949 attacked per sweep
Sweeps: 5
Keep the full attacked population. Correct native positives only when every identifiable attacker-goal call in Slack injection task 5 was explicitly refused by policy. Retain output-only outcomes and conservative errors.

### Without Repeat User Prompt

| Configuration | Clean Utility | Util Under Attack | ASR (AgentDojo) | ASR (Effect) |
|---------------|---------------|-------------------|-----------------|-------------|
| No runtime defense | 79.0 ± 2.02 | 56.9 ± 1.43 | 25.2 ± 1.12 | 25.2 ± 1.12 |
| Tool-call filter | 72.4 ± 2.39 | 63.5 ± 0.83 | 4.2 ± 0.18 | 4.2 ± 0.18 |
| FORGE (FIDES IFC Policy) | 32.2 ± 2.39 | 31.0 ± 0.55 | 1.5 ± 0.40 | 1.0 ± 0.41 |
| FORGE (provenance policy) | 57.3 ± 3.49 | 49.4 ± 0.59 | 1.7 ± 0.31 | 0.8 ± 0.15 |

### With Repeat User Prompt

| Configuration | Clean Utility | Util Under Attack | ASR (AgentDojo) | ASR (Effect) |
|---------------|---------------|-------------------|-----------------|-------------|
| No runtime defense | 83.7 ± 0.41 | 72.5 ± 0.81 | 14.1 ± 0.46 | 14.1 ± 0.46 |
| Tool-call filter | 73.4 ± 3.35 | 69.5 ± 0.74 | 3.1 ± 0.48 | 3.1 ± 0.48 |
| FORGE (FIDES IFC Policy) | 33.2 ± 1.89 | 34.4 ± 0.51 | 0.6 ± 0.12 | 0.3 ± 0.20 |
| FORGE (provenance policy) | 62.5 ± 1.40 | 63.8 ± 0.91 | 0.9 ± 0.30 | 0.2 ± 0.12 |

### Supplementary Statistics

| Arm | Sweeps | Errors | Runs w/Refusal | Refusal Calls | Runtime Coverage | Runtime Mean | Runtime Median | Runtime Min | Runtime Max |
|-----|--------|--------|----------------|---------------|-----------------|----------------|--------------|-------------|-------------|
| N-none | 5 | 1 | 0 | 0 | 5230/5230 | 4.83s | 4.36s | 0.89s | 114.43s |
| N-tf | 5 | 0 | 0 | 0 | 5230/5230 | 5.05s | 4.11s | 1.28s | 101.68s |
| P-fides | 5 | 0 | 3005 | 4651 | 5230/5230 | 5.67s | 4.90s | 0.69s | 93.27s |
| C-w2-norep | 5 | 2 | 2134 | 3103 | 5230/5230 | 5.03s | 4.51s | 0.92s | 159.44s |
| N-none-rep | 5 | 0 | 0 | 0 | 5230/5230 | 5.93s | 4.43s | 0.74s | 606.55s |
| N-tf-rep | 5 | 0 | 0 | 0 | 5230/5230 | 6.40s | 4.25s | 1.30s | 189.64s |
| P-fides-rep | 5 | 0 | 3032 | 15188 | 5230/5230 | 10.18s | 5.06s | 0.79s | 630.56s |
| C-w2 | 5 | 0 | 1726 | 4252 | 5230/5230 | 7.70s | 4.69s | 0.82s | 146.27s |

**Note:** Cost data is not recorded in run results; cost is unavailable.
