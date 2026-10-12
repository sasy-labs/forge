# Variant set, round 1 (tools_json inert in the offline harness)

Five independent runs of the trace-grounded /write-policy skill,
graded on tau2 airline tasks 8/43/47 at 5 trials each.

| run | reward | task 8 | task 43 | task 47 | tools_json refs |
|---|---|---|---|---|---|
| 1 | 1.000 | 5/5 | 5/5 | 5/5 | 0 |
| 2 | 0.867 | 5/5 | 3/5 | 5/5 | 3 |
| 3 | 0.933 | 5/5 | 4/5 | 5/5 | 3 |
| 4 | 0.533 | 5/5 | 0/5 | 3/5 | 2 |
| 5 | 0.933 | 4/5 | 5/5 | 5/5 | 1 |

mean 0.853 +/- 0.185 (n=5). Hand-written reference: 1.000.

All five converged offline with zero mismatches, so offline
convergence did not predict live reward. Run 4 false-blocked
cancel_reservation in 5/5 task-43 simulations for "the user has not
confirmed", on a task the reference passes 5/5.

Round 1 evaluated `Message.tools_json` as `[]`, so rules using that field
were tested against an empty tool list. Round 2 corrected the event
representation.
