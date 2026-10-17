# W13: direct-distance continuation and a genuine time model

Requested scope: execute steps 1, 2, and 4; omit step 3 (CellFlow/scDFM and
direct-U adapters). Internal static and shuffled-kinetics arms remain included.

1. Resume b1_direct_distance from its fixed 12,000-step Adam/RNG snapshots to
   24,000 and 36,000 steps, for three arms and seeds 0/1/2. Preserve the frozen
   global mean, feature pack, and all other hyperparameters.
2. Freeze the completed endpoint models, then evaluate all six available windows:
   5->10, 10->15, 15->20, 5->15, 10->20, and 5->20 hours. Report results at
   5/10/15-hour horizons separately, not as independent biological replicates.
3. Implement a small genuinely integrated time-dependent mixture field. Routing
   is source-defined and fixed through integration; each expert follows its own
   trajectory. The weighted residual field is centered so that the mean branch
   is unchanged. Train through RK4; do not relabel endpoint extrapolation as ODE.
   Include a parameter-/update-matched endpoint time-field control, static and
   shuffled arms, and three seeds. Fix numerical settings and training budgets
   after an unevaluated training-only timing/solver smoke test, before examining
   new prediction scores.

The mean follows the existing frozen elapsed-time kernel. Intermediate restarts
must retain the original elapsed time and source routing; resetting the kernel
would define a different process. Kinetic features are analytical source S/U
residuals, not an original GFG checkpoint, JVP, or a calibrated physical velocity.
Daily training cohorts remain r1-r3; r4 is same-series development only. No r4
targets enter training. A negative ODE result leaves the endpoint model as the
working predictor; it does not justify repeated post-hoc horizon tuning.

GPU allocation: use physical GPUs 1/2/3 while GPU0 runs an unrelated job; never
touch GPUs4-7. Keep existing artifacts and save new RESULTS.md/CSV/provenance.
