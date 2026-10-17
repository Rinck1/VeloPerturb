# Source-only response-mass calibration

All students use S-only local routing and frozen teacher experts. Teacher source-cohort
mode masses are inferred from source S/U, never from future targets. The shared frozen
mean predictor still uses analytic dynamics. Lower weighted predictive-mixture energy is better.

| Teacher family | Plain S student | Source-mass-calibrated student | Teacher cohort-average routing |
|---|---:|---:|---:|
| static | 0.621382413 | 0.621372068 | 0.617171596 |
| real dynamics | 0.620290909 | 0.606786916 | 0.604149992 |
| shuffled dynamics | 0.618372172 | 0.625728850 | 0.620654071 |

The real-dynamics teacher had energy 0.604895770. Source-mass calibration recovers
87.716% of the performance lost by its S-only student. The uncalibrated student
does not beat the shuffled-teacher student; do not call uncalibrated distillation successful.
Calibration fits no new parameters, but the teacher kinetic router is still needed
to infer source masses. Do not claim U-free deployment.

All source interventions were frozen in prediction_manifest.json before target
evaluation. Exact values, interval scores, and source-only mass errors are in
metrics.csv, interval_metrics.csv, and source_calibration.csv; source/data/code
provenance is in summary.json and code_snapshot/.
