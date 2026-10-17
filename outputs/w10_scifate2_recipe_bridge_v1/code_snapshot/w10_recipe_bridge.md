# W10: bounded recipe bridge, followed by matched continuation

User authorized continuous execution on 2026-10-17. Keep the current analytic
source dynamics and frozen mean predictor; no GFG/JVP, new ODE, or capacity sweep.
Use currently free physical GPUs 1–3. GPU0 has an unrelated job; never terminate it.

## Fixed experiment

K=8, width 128, rank 32, router width 64: 76,248 trainable parameters per arm.
Reuse immutable W6 features, W5 mean forecasts, and W7 evaluation samples/caches.
Train r1–r3 only. Evaluate the already-observed r4 hourly series as development.
Each recipe has static, real-velocity, and shuffled-velocity routers, seeds 0/1/2,
12000 steps, fixed checkpoints 1000/4000/12000: 54 training jobs, no new grid.

| Recipe | Centering | Energy estimator | Distance kernel | Regularizer population | LR |
|---|---|---|---|---|---|
| b0: original W8 | batch | V | original heuristic | batch | 3e-4 |
| b1 | batch | V | direct | batch | 3e-4 |
| b2 | full source | V | direct | batch | 3e-4 |
| b3 | full source | source-draw U | direct | batch | 3e-4 |
| b4 | full source | source-draw U | direct | full source | 3e-4 |
| b5: W9 small | full source | source-draw U | direct | full source | 1e-4 |

Adjacent comparisons isolate one local recipe change, conditional on earlier
changes; this is NOT a full factorial decomposition or proof of absence of
interactions. A V-statistic minibatch objective is a valid empirical objective,
not a data-leak bug. U-statistic unbiasedness requires the deterministic full
source population used by b3–b5; do not call batch-centered U training unbiased.

Shuffle source velocity rows once per source cohort, separately for train and
prediction, using a separate fixed RNG stream and the same permutation across
recipes. It preserves the velocity marginal and destroys its cell-level pairing.
Sampling, initialization, parameter count, budget, and frozen mean are matched.
Only local routing is shuffled: even the static-router arm keeps the SAME frozen
analytic mean predictor. Thus this compares local routing, not whole-model S-only
versus S/U, and cannot establish velocity-specificity relative to every possible
alternative representation of U or S.

## Numerical anchors and diagnostics

CPU tests must reproduce W6 and W9 optimizer trajectories for b0 and b5, preserve
the independent-source U-statistic definition, and verify deterministic shuffling
without changing optimizer RNG. Store source-group IDs in logs: the old every-200
step logging covered only three of the fifteen training groups. W10 logs every
300 steps on a rotating offset so all fifteen groups are visible over time.
Historical W8/W9 numerical endpoints are comparisons, not newly fitted targets.
Report all intervals and seeds, mean anchor error, covariance/variance, effective
expert counts, real-vs-static and real-vs-shuffled differences. Do not label seeds
or the three windows independent biological replication; no significance claim.

## Continuous execution rule, fixed before new results

After the complete 12000-step grid, select the lowest-energy real-velocity recipe
that beats BOTH matched controls on the equal-window, three-seed predictive-mixture
energy. If one exists, continue ALL THREE matched arms, ALL THREE seeds to 24000
and 36000 with Adam and RNG restored, keeping mean/data/K fixed. Publish every
checkpoint, not only the best. This is a bounded development expansion, not a
confirmation test or permission for indefinite outcome-driven tuning.

If none exists, record that local velocity pairing has not delivered a reproducible
advantage in this recipe family; that is a substantive blocker for the dynamic-router
claim. Do not silently change the success rule or spend the next round inflating
parameters. Numerical failure, source/evaluation contract failure, or low disk
space also halts the affected job and is reported, with all existing files retained.

Only compact small-model checkpoints/predictions are added (no raw data copies).
No GitHub upload is authorized by this execution request.
