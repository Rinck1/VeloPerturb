# W13 frozen forecast horizons

All fixed arms and windows; weighted three-seed predictive mixtures.

| Model | Horizon (h) | Windows | Energy | vs static | vs shuffled |
|---|---:|---:|---:|---:|---:|
| b1_endpoint36k | 5 | 3 | 0.595683 | +4.137% | +4.606% |
| b1_endpoint36k | 10 | 2 | 1.052618 | +6.836% | +7.382% |
| b1_endpoint36k | 15 | 1 | 1.004217 | +2.247% | +1.564% |
| endpoint_time_40k | 5 | 3 | 0.596823 | +4.333% | +4.935% |
| endpoint_time_40k | 10 | 2 | 1.056843 | +6.922% | +7.837% |
| endpoint_time_40k | 15 | 1 | 1.008107 | +1.648% | +2.146% |
| ode_40k | 5 | 3 | 0.589607 | +3.707% | +4.613% |
| ode_40k | 10 | 2 | 1.027366 | +6.339% | +7.410% |
| ode_40k | 15 | 1 | 1.007838 | +2.407% | +1.576% |

Shared overlapping windows are not independent biological replicates. All models retain the same analytical mean branch; static denotes the local router control, not whole-model U-free prediction.
