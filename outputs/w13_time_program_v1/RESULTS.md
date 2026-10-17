# W13: continuation, horizons, and genuine ODE

| Model | Hours | Energy | vs static | vs shuffled |
|---|---:|---:|---:|---:|
| b0_endpoint36k | 5 | 0.598119 | +3.613% | +3.967% |
| b0_endpoint36k | 10 | 1.059150 | +6.138% | +6.596% |
| b0_endpoint36k | 15 | 1.002958 | +2.233% | +2.028% |
| b1_endpoint36k | 5 | 0.595683 | +4.137% | +4.606% |
| b1_endpoint36k | 10 | 1.052618 | +6.836% | +7.382% |
| b1_endpoint36k | 15 | 1.004217 | +2.247% | +1.564% |
| endpoint_time_40k | 5 | 0.596823 | +4.333% | +4.935% |
| endpoint_time_40k | 10 | 1.056843 | +6.922% | +7.837% |
| endpoint_time_40k | 15 | 1.008107 | +1.648% | +2.146% |
| ode_40k | 5 | 0.589607 | +3.707% | +4.613% |
| ode_40k | 10 | 1.027366 | +6.339% | +7.410% |
| ode_40k | 15 | 1.007838 | +2.407% | +1.576% |

## Integration effect

- 5h: +1.209% vs matched endpoint time model.
- 10h: +2.789% vs matched endpoint time model.
- 15h: +0.027% vs matched endpoint time model.

Same-series development, not independent replication or genetic perturbation generalization. Static/shuffled labels describe local routing; the analytical mean is shared. Step 3 was not run. No original GFG/JVP was introduced.
