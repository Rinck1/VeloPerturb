# W13 final results: requested steps 1/2/4 complete

27 formal runs completed; step 3 omitted. Source kinetics remain analytical, not original GFG/JVP.

Direct-distance endpoint36k: energy 0.595683; +4.137% vs static and +4.606% vs shuffled kinetics (5h).

## Genuine small ODE, frozen weights, 3h inference step

| Hours | Energy | vs own static | vs shuffled | vs matched endpoint time model |
|---|---:|---:|---:|---:|
| 5 | 0.589504 | +3.723% | +4.620% | +1.226% |
| 10 | 1.027032 | +6.358% | +7.418% | +2.821% |
| 15 | 1.007833 | +2.415% | +1.573% | +0.027% |

The fixed 6h study remains unchanged. 3h inference uses identical weights, with no training or target-dependent checkpoint selection. The largest 3h vs 1.5h source-only difference is 0.427% of predicted movement.

ODE improves absolute 5/10h fit, including the static controls; it is not evidence of a larger velocity-specific interaction. At 15h it is essentially tied with the time endpoint control and slightly worse than the original b1 endpoint. Use the small ODE as the short-horizon candidate and retain the endpoint reference, not a post-hoc per-window winner assembly.

Mean prediction is frozen throughout. Windows are same-series development, not independent replication or unseen genetic perturbation validation.
