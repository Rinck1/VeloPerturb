# W11: train the population-response interpretation, without new capacity

Hypothesis-driven follow-up to W10 frozen routing interventions, 2026-10-17.
At fixed K8/12000, replacing individual q with the source-cohort mean q retains
about 97% of the real-vs-static energy improvement. Hard argmax is worse; this
does not invalidate stochastic sampling from the soft predictive mixture.

Train q_cohort = mean_i softmax(router(z_i,v_i)) and broadcast q_cohort to all
source cells before weighted mean centering. This is a joint state/dynamics
population fingerprint, NOT mean velocity alone or a cell-fate classifier.
S-only experts and all 76,248 trainable parameters are unchanged. Entire source
cohort is used in both training and prediction. Individual routing MI is zero
by construction, so do not apply or claim that regularizer in this model.

Use W10 b2's full-source centering, direct-distance V-statistic, LR3e-4,
batch residual-L2, and fixed 1000/4000/12000 checkpoints. Three arms
(static, real dynamics, within-source shuffled dynamics), three matched seeds:
9 training jobs, not a new hyperparameter grid. Static and real both have the
same pooling mechanism and zero cell-wise MI. Compare against BOTH matched
cohort controls and the original local b2 and b0 as development references.

Reuse W6 features and W7 evaluation samples; r1–r3 optimizer only, r4 same-series
development. Mean predictor remains frozen and shared. Report every checkpoint;
do not claim equivalence to the frozen intervention before retraining is done.
This does not interrupt or replace the fixed W10 bridge or its continuation.
Use physical GPUs 1–3 only; concurrent small jobs may share those GPUs if the
main study is still running. Do not touch GPU0's unrelated project or GPUs4–7.
