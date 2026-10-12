Task cohorts: any action denied in any of four enforcement-on trials, or none. The same tasks are compared off/on; all four trials retained. pass^4 means all four succeed. Deltas are percentage points.

| Domain | Agent | Cohort | Tasks | Trials | Success off → on | Δ pp | pass^4 off → on | Δ pp |
|---|---|---|---:|---:|---:|---:|---:|---:|
| airline | opus | any_blocked | 12 | 48 | 45.8% → 72.9% | +27.1 | 8.3% → 50.0% | +41.7 |
| airline | opus | none_blocked | 38 | 152 | 88.8% → 90.1% | +1.3 | 81.6% → 78.9% | -2.6 |
| airline | gpt | any_blocked | 19 | 76 | 36.8% → 71.1% | +34.2 | 5.3% → 57.9% | +52.6 |
| airline | gpt | none_blocked | 31 | 124 | 79.0% → 81.5% | +2.4 | 51.6% → 64.5% | +12.9 |
| airline | gemini | any_blocked | 2 | 8 | 12.5% → 37.5% | +25.0 | 0.0% → 0.0% | +0.0 |
| airline | gemini | none_blocked | 48 | 192 | 85.4% → 82.8% | -2.6 | 79.2% → 70.8% | -8.3 |
| retail | opus | any_blocked | 17 | 68 | 72.1% → 77.9% | +5.9 | 47.1% → 64.7% | +17.6 |
| retail | opus | none_blocked | 97 | 388 | 91.2% → 91.0% | -0.3 | 82.5% → 81.4% | -1.0 |
| retail | gpt | any_blocked | 40 | 160 | 70.0% → 80.0% | +10.0 | 37.5% → 60.0% | +22.5 |
| retail | gpt | none_blocked | 74 | 296 | 89.9% → 89.2% | -0.7 | 78.4% → 71.6% | -6.8 |
| retail | gemini | any_blocked | 15 | 60 | 86.7% → 93.3% | +6.7 | 80.0% → 86.7% | +6.7 |
| retail | gemini | none_blocked | 99 | 396 | 87.6% → 86.4% | -1.3 | 73.7% → 75.8% | +2.0 |
| airline | all | any_blocked | 33 | 132 | 38.6% → 69.7% | +31.1 | 6.1% → 51.5% | +45.5 |
| airline | all | none_blocked | 117 | 468 | 84.8% → 84.8% | +0.0 | 72.6% → 71.8% | -0.9 |
| airline | all | all | 150 | 600 | 74.7% → 81.5% | +6.8 | 58.0% → 67.3% | +9.3 |
| retail | all | any_blocked | 72 | 288 | 74.0% → 82.3% | +8.3 | 48.6% → 66.7% | +18.1 |
| retail | all | none_blocked | 270 | 1080 | 89.5% → 88.8% | -0.7 | 78.1% → 76.7% | -1.5 |
| retail | all | all | 342 | 1368 | 86.3% → 87.4% | +1.2 | 71.9% → 74.6% | +2.6 |
| all | all | any_blocked | 105 | 420 | 62.9% → 78.3% | +15.5 | 35.2% → 61.9% | +26.7 |
| all | all | none_blocked | 387 | 1548 | 88.1% → 87.6% | -0.5 | 76.5% → 75.2% | -1.3 |
| all | all | all | 492 | 1968 | 82.7% → 85.6% | +2.9 | 67.7% → 72.4% | +4.7 |

Individual trial cohorts (paired by task/trial/seed; pass^4 is not calculated on filtered trials):

| Domain | Agent | Cohort | Trials | Success off → on | Δ pp | Gains / losses |
|---|---|---|---:|---:|---:|---:|
| airline | opus | any_blocked | 25 | 40.0% → 80.0% | +40.0 | 12 / 2 |
| airline | opus | none_blocked | 175 | 84.0% → 86.9% | +2.9 | 14 / 9 |
| airline | gpt | any_blocked | 38 | 36.8% → 76.3% | +39.5 | 16 / 1 |
| airline | gpt | none_blocked | 162 | 69.1% → 77.8% | +8.6 | 26 / 12 |
| airline | gemini | any_blocked | 5 | 20.0% → 60.0% | +40.0 | 2 / 0 |
| airline | gemini | none_blocked | 195 | 84.1% → 81.5% | -2.6 | 7 / 12 |
| retail | opus | any_blocked | 34 | 70.6% → 73.5% | +2.9 | 6 / 5 |
| retail | opus | none_blocked | 422 | 89.8% → 90.3% | +0.5 | 22 / 20 |
| retail | gpt | any_blocked | 74 | 64.9% → 81.1% | +16.2 | 19 / 7 |
| retail | gpt | none_blocked | 382 | 86.4% → 86.9% | +0.5 | 27 / 25 |
| retail | gemini | any_blocked | 23 | 87.0% → 100.0% | +13.0 | 3 / 0 |
| retail | gemini | none_blocked | 433 | 87.5% → 86.6% | -0.9 | 18 / 22 |
| airline | all | any_blocked | 68 | 36.8% → 76.5% | +39.7 | 30 / 3 |
| airline | all | none_blocked | 532 | 79.5% → 82.1% | +2.6 | 47 / 33 |
| airline | all | all | 600 | 74.7% → 81.5% | +6.8 | 77 / 36 |
| retail | all | any_blocked | 131 | 70.2% → 82.4% | +12.2 | 28 / 12 |
| retail | all | none_blocked | 1237 | 88.0% → 88.0% | +0.0 | 67 / 67 |
| retail | all | all | 1368 | 86.3% → 87.4% | +1.2 | 95 / 79 |
| all | all | any_blocked | 199 | 58.8% → 80.4% | +21.6 | 58 / 15 |
| all | all | none_blocked | 1769 | 85.4% → 86.2% | +0.8 | 114 / 100 |
| all | all | all | 1968 | 82.7% → 85.6% | +2.9 | 172 / 115 |

Groups are descriptive post-treatment subsets, not causal estimates. All counts and aggregate pass^4 match tau2-results-audit.json. All denial flags match [BLOCKED] tool-response markers. No unresolved infrastructure failures.
