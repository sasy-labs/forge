### Results

| Configuration | Policy | ASR | Utility |
|---|---|---:|---:|
| Instrumented | MLS (TOP_SECRET) | 0/5 | 5/5 |
|  | MLS (SECRET) | 0/5 | 5/5 |
|  | Toxic Flow | 0/5 | 5/5 |
| Non-Instrumented | Anti-Exfil. Prompt | 5/5 | 5/5 |

### Runtime and cost

| Configuration | Policy | Mean Time (s) | Mean Cost ($) |
|---|---|---:|---:|
| Instrumented | MLS (TOP_SECRET) | 7.9 | 0.0035 |
|  | MLS (SECRET) | 7.6 | 0.0038 |
|  | Toxic Flow | 11.1 | 0.0047 |
| Non-Instrumented | Anti-Exfil. Prompt | 5.3 | 0.0030 |

### Supplementary: Runtime statistics per group

| Configuration | Policy | N (scenarios) | Mean (s) | Min (s) | Max (s) | Values |
|---|---|---:|---:|---:|---:|---|
| Instrumented | MLS (TOP_SECRET) | 10 | 7.9 | 4.9 | 13.2 | 7.0, 6.2, 13.2, 5.0, 7.3, 7.3, 11.5, 4.9, 9.0, 7.2 |
| Instrumented | MLS (SECRET) | 10 | 7.6 | 5.1 | 12.1 | 6.9, 6.6, 5.1, 7.3, 8.4, 6.5, 12.1, 6.9, 8.9, 6.7 |
| Instrumented | Toxic Flow | 10 | 11.1 | 7.0 | 16.8 | 15.0, 14.1, 16.8, 9.7, 12.4, 7.0, 9.1, 8.0, 11.1, 7.8 |
| Non-Instrumented | Anti-Exfil. Prompt | 10 | 5.3 | 4.2 | 6.9 | 6.2, 5.9, 5.4, 6.9, 4.6, 5.0, 5.2, 4.4, 4.9, 4.2 |
