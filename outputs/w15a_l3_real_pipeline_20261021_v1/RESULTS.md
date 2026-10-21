# L3 real-data pipeline implementation

21 CPU tests passed in 16.50 seconds. No real-data model or GPU job was launched.

The new train/predict/controller files leave the frozen MoE core and active 120-fit synthetic runner unchanged. The actual CPU fixture traverses official source-only OT training, model serialization, independent Tsit5 K8 trajectories, five common prediction exports, full prediction freezing, and common target scoring. Separate mocked-boundary tests cover the production `train_real` artifact writer and the complete registered 18-model orchestration, without fitting real data.

Production is fixed at six variants times three seeds. The real kinetic variant also exports cohort/uniform/current-U interventions without retraining: 18 primary plus 12 diagnostic prediction manifests, each containing the same 12 W13 windows. Cohort weights use every source cell before the canonical512 subset.

The deferred controller requires the complete unchanged 120-fit synthetic engineering gate and the complete L2 launcher status. It trains and freezes on GPU0 only after that process exits, then releases GPU0 before launching common evaluation on CPU. Synthetic gate failure writes a terminal not-started result; it does not retry, change loss, or run real training.

This is implementation evidence, not a biological or performance result. Root review and launch authorization remain pending.
