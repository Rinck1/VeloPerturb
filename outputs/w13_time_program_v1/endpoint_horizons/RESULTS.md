# W13 frozen forecast horizons

All fixed arms and windows; weighted three-seed predictive mixtures.

| Model | Horizon (h) | Windows | Energy | vs static | vs shuffled |
|---|---:|---:|---:|---:|---:|
| b0_endpoint36k | 5 | 3 | 0.598119 | +3.613% | +3.967% |
| b0_endpoint36k | 10 | 2 | 1.059150 | +6.138% | +6.596% |
| b0_endpoint36k | 15 | 1 | 1.002958 | +2.233% | +2.028% |
| b1_endpoint36k | 5 | 3 | 0.595683 | +4.137% | +4.606% |
| b1_endpoint36k | 10 | 2 | 1.052618 | +6.836% | +7.382% |
| b1_endpoint36k | 15 | 1 | 1.004217 | +2.247% | +1.564% |

Shared overlapping windows are not independent biological replicates. All models retain the same analytical mean branch; static denotes the local router control, not whole-model U-free prediction.
