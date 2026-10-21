# W14 cross-replicate results

P1: True

P2: False

Classification: PARTIALLY_CONFIRMED

24-hour internal validation; not a verdict on the 5–15h development claim.

{
  "status": "W14_GRID_EVALUATED",
  "P1": true,
  "P2": false,
  "classification": "PARTIALLY_CONFIRMED",
  "overall_gain_vs_static": 0.024957510318535908,
  "global_energy": {
    "static": 0.1610999712825949,
    "real": 0.15655462309351548,
    "direct_U": 0.15096932304809546,
    "nonlinear_SU": 0.16473940130458525,
    "shuffled_v_perm0": 0.15613556599256986,
    "shuffled_v_perm1": 0.15661822366221503,
    "shuffled_v_perm2": 0.15569062457604227,
    "shuffled_v_perm3": 0.156906932661469,
    "shuffled_v_perm4": 0.15650034373988167
  },
  "comparisons": [
    {
      "heldout": "r1",
      "control": "static",
      "mean_gain": -0.03775399472761398
    },
    {
      "heldout": "r1",
      "control": "direct_U",
      "mean_gain": -0.21575950389807202
    },
    {
      "heldout": "r1",
      "control": "nonlinear_SU",
      "mean_gain": -0.028155614174240083
    },
    {
      "heldout": "r1",
      "control": "shuffled_v_perm0",
      "mean_gain": -0.18040209134511564
    },
    {
      "heldout": "r1",
      "control": "shuffled_v_perm1",
      "mean_gain": -0.14392304030367326
    },
    {
      "heldout": "r1",
      "control": "shuffled_v_perm2",
      "mean_gain": -0.1317292598565345
    },
    {
      "heldout": "r1",
      "control": "shuffled_v_perm3",
      "mean_gain": -0.11369547682726651
    },
    {
      "heldout": "r1",
      "control": "shuffled_v_perm4",
      "mean_gain": -0.17488207052771895
    },
    {
      "heldout": "r2",
      "control": "static",
      "mean_gain": 0.019757412681428298
    },
    {
      "heldout": "r2",
      "control": "direct_U",
      "mean_gain": -0.022178607695104956
    },
    {
      "heldout": "r2",
      "control": "nonlinear_SU",
      "mean_gain": -0.03246626651506153
    },
    {
      "heldout": "r2",
      "control": "shuffled_v_perm0",
      "mean_gain": 0.005603456961145792
    },
    {
      "heldout": "r2",
      "control": "shuffled_v_perm1",
      "mean_gain": 0.007047669446667615
    },
    {
      "heldout": "r2",
      "control": "shuffled_v_perm2",
      "mean_gain": -0.007650411708995964
    },
    {
      "heldout": "r2",
      "control": "shuffled_v_perm3",
      "mean_gain": 0.006159225569807436
    },
    {
      "heldout": "r2",
      "control": "shuffled_v_perm4",
      "mean_gain": 0.008066141854354725
    },
    {
      "heldout": "r3",
      "control": "static",
      "mean_gain": 0.09286911300179342
    },
    {
      "heldout": "r3",
      "control": "direct_U",
      "mean_gain": 0.015215120527440961
    },
    {
      "heldout": "r3",
      "control": "nonlinear_SU",
      "mean_gain": 0.21344942774597095
    },
    {
      "heldout": "r3",
      "control": "shuffled_v_perm0",
      "mean_gain": 0.12121570395372813
    },
    {
      "heldout": "r3",
      "control": "shuffled_v_perm1",
      "mean_gain": 0.11312918364806519
    },
    {
      "heldout": "r3",
      "control": "shuffled_v_perm2",
      "mean_gain": 0.10893742448418123
    },
    {
      "heldout": "r3",
      "control": "shuffled_v_perm3",
      "mean_gain": 0.10691913893315655
    },
    {
      "heldout": "r3",
      "control": "shuffled_v_perm4",
      "mean_gain": 0.11573527306136107
    }
  ],
  "scope": "internal_fixed_recipe_cross_replicate_24h_not_new_short_horizon_replication",
  "biological_significance_claim": false,
  "independent_dataset": false,
  "total_training_jobs": 81
}
