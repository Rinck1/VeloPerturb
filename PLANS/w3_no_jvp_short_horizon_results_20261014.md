# RENGE day4→day5 no-JVP velocity pilot — 2026-10-14

## Scope

This is a development experiment on the existing condition-held-out RENGE fold. The task is adjacent-time population-mean prediction: day4 source S/U and perturbation condition → day5 group mean in the frozen training-fold PCA-50 space. The three seeds use held-out TFs LIN28A, NANOG, POU5F1, and ZIC3. All source-only predictions and model hashes were frozen before the validation target pack was opened.

The velocity estimator is the target-domain-trained, shallow GFGv2 no-JVP head (per-gene 3-input linear velocity head, β=1, training-source γ and U-innovation frozen). The encoder/attention/decoder path is bypassed during head training. This is a velocity-utility pilot, not the full VeloRoute router/experts model and not a cell-fate experiment.

## Frozen held-out result

| Arm | Mean group-mean MSE | Relative to S-only |
|---|---:|---:|
| S-only ridge | 0.0399565 | reference |
| Target-retrained no-JVP velocity | 0.0393029 | 1.64% lower |
| Same estimator, locally shuffled U | 0.0422265 | 5.68% higher |
| Training-fold kinetic reference | 0.0390351 | 2.31% lower |

The learned no-JVP velocity arm beats S-only on 3/4 held-out TFs (LIN28A, POU5F1, ZIC3) and loses on NANOG. The kinetic reference beats S-only on all four and is slightly better on average than the learned head. Seeds are algorithmic repeats; the independent units are four TF conditions, so no cell-level significance claim is warranted.

## Training-fold leave-one-TF-out diagnostic

Using only the 14 training TFs, with the same local-U shuffle used in the held-out pilot:

| Arm | Mean LOTO MSE | Relative to S-only |
|---|---:|---:|
| S-only ridge | 0.0385219 | reference |
| Target-retrained no-JVP velocity | 0.0350205 | 9.09% lower |
| Same estimator, locally shuffled U | 0.0332157 | 13.77% lower |
| Training-fold kinetic reference | 0.0378366 | 1.78% lower |

This train/held-out gap is a warning, not evidence to select the stronger training-fold number. The shuffle preserves broad condition-level U composition, so it breaks local cell-level U↔S correspondence more than it removes population-level U information. Its strong LOTO result means the present experiment does **not** isolate a uniquely useful learned velocity mechanism.

## Interpretation

There is a small encouraging signal: a freshly fitted no-JVP velocity head adds predictive information to a static S-only ridge on most of the four held-out TFs for this short interval. But the signal is modest, the kinetic reference is at least as good, and the train-fold advantage does not carry over at the same magnitude. Stable velocity utility is **not established**; this pilot does not justify full router/experts training yet.

## Immediate next experiment

1. Keep the day4→day5 split and compare population-mean velocity with velocity computed from pooled/group-mean U/S, plus S-only and a condition-level U permutation that actually removes TF-specific population U information.
2. Report condition-wise errors and sensitivity to source-cell subsampling; treat TFs, not cells or seeds, as the independent units.
3. Only if the held-out increment survives those controls, proceed to the full router experiment. Process day2/day3 S/U separately before making any claim about longer temporal horizons.

Artifacts:

- Frozen predictions/models: `/data/yuchang/veloroute_w3_20261014/short_horizon_no_jvp/seed{0,1,2}_v1/`
- Held-out metrics/provenance: `outputs/w3_7_shorttime_results_20261014/evaluation_v2/`
- Train-only LOTO metrics: `outputs/w3_7_shorttime_results_20261014/train_cv_matched_v1/`
