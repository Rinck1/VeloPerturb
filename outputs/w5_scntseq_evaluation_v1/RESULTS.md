# scNT-seq minute-scale rolling forecasts

One nonnegative scale per calibrated arm, fitted only on 15→30. Test: 30→60 and 60→120. No biological p-values from cells/genes.

{
  "status": "SCNT_MINUTES_EVALUATED",
  "kind": "single_series_descriptive_mechanistic_validation",
  "freeze_commit": "659d76e2063a0daae9ce77de4d7005f420dc267d",
  "intervals": [
    [
      30,
      60
    ],
    [
      60,
      120
    ]
  ],
  "primary_velocity_arm": "vel_mean",
  "primary_descriptive_rule_met": false,
  "velocity_beats_identity_in_both_intervals": false,
  "velocity_beats_both_past_baselines": false,
  "velocity_beats_both_direct_U_baselines": false,
  "mean_mse": {
    "identity": 0.3366325026388139,
    "past_S": 1.9085144439281225,
    "past_S_calibrated": 0.3366325026388139,
    "vel_mean": 0.3366325026388139,
    "vel_pool": 0.3366325026388139,
    "direct_U_raw": 0.3366325026388139,
    "direct_U_delta0": 0.3366325026388139,
    "new_raw": 0.3366325026388139,
    "new_delta0": 3.9504745826442065
  },
  "mean_interval_gain_over_identity": {
    "identity": 0.0,
    "past_S": -13.672560198605566,
    "past_S_calibrated": 0.0,
    "vel_mean": 0.0,
    "vel_pool": 0.0,
    "direct_U_raw": 0.0,
    "direct_U_delta0": 0.0,
    "new_raw": 0.0,
    "new_delta0": -37.749836304322415
  },
  "biological_replicate_count": null,
  "significance_test_performed": false,
  "new_RNA_is_net_velocity": false,
  "router_enabled": false,
  "gfg_jvp_enabled": false,
  "calibration_interval_is_test": false,
  "inference_scope": "two_rolling_intervals_one_stimulus_series"
}
