# W13 frozen forecast horizons

All fixed arms and windows; weighted three-seed predictive mixtures.

| Model | Horizon (h) | Windows | Energy | vs static | vs shuffled |
|---|---:|---:|---:|---:|---:|
| ode_40k_step3h | 5 | 3 | 0.589504 | +3.723% | +4.620% |
| ode_40k_step3h | 10 | 2 | 1.027032 | +6.358% | +7.418% |
| ode_40k_step3h | 15 | 1 | 1.007833 | +2.415% | +1.573% |

Shared overlapping windows are not independent biological replicates. All models retain the same analytical mean branch; static denotes the local router control, not whole-model U-free prediction.
