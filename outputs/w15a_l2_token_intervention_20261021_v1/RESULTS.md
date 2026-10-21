# Frozen CellFlow kinetic-token intervention

No training. Original three-seed checkpoints, source IDs, target caches, clocks and other conditions stay fixed.
Original36 windows are reused; masked72 ODE forecasts are all frozen before intervention scoring.
Training-mean mask sets the STANDARDIZED group token to0; source05 mask uses raw0 normalized by frozen statistics.
These masks are distinct unless the training mean is exactly0. Origin05 group context was already a self-reference constant.

| Mask | Clock | Horizon | Original Energy | Masked Energy | Relative error increase |
|---|---|---:|---:|---:|---:|
| cf_kin_mask_training_mean | decay | 5h | 0.592171 | 0.747569 | +26.24% |
| cf_kin_mask_training_mean | decay | 10h | 0.918257 | 1.493618 | +62.66% |
| cf_kin_mask_training_mean | decay | 15h | 0.876706 | 1.961115 | +123.69% |
| cf_kin_mask_training_mean | linear | 5h | 0.600255 | 0.733688 | +22.23% |
| cf_kin_mask_training_mean | linear | 10h | 0.976736 | 1.531750 | +56.82% |
| cf_kin_mask_training_mean | linear | 15h | 0.940142 | 2.051228 | +118.18% |
| cf_kin_mask_source05_reference | decay | 5h | 0.592171 | 0.590554 | -0.27% |
| cf_kin_mask_source05_reference | decay | 10h | 0.918257 | 0.873924 | -4.83% |
| cf_kin_mask_source05_reference | decay | 15h | 0.876706 | 0.876706 | +0.00% |
| cf_kin_mask_source05_reference | linear | 5h | 0.600255 | 0.597920 | -0.39% |
| cf_kin_mask_source05_reference | linear | 10h | 0.976736 | 0.943500 | -3.40% |
| cf_kin_mask_source05_reference | linear | 15h | 0.940142 | 0.940142 | +0.00% |

Positive error increase means actual token is useful relative to that frozen-weight mask; negative means masking improves this metric.
Endpoint changes test whether the model uses the token at inference. They do not alone show that the token is beneficial.
Origin05 reference-mask endpoints should be unchanged; improvements/deterioration from the training-mean mask there only probe a constant context.
Windows overlap; seeds are optimization repeats, not independent biological replicates. This is an exploratory post-main-grid mechanism intervention.
