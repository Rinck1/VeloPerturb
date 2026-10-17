# Post-hoc minute-scale direction audit

Oracle scales/ceilings use already observed future states and are NOT prediction results. Baseline-centered velocity is exploratory. Frozen primary forecasts remain unchanged.

{
  "status": "POSTHOC_MINUTE_DIRECTION_AUDIT_COMPLETE",
  "exploratory": true,
  "original_predictions_changed": false,
  "oracle_results_are_deployable": false,
  "gamma_zero_genes": 1025,
  "total_HVGs": 2000,
  "zero_gamma_reasons": {
    "insufficient_positive_S": 0,
    "zero_tail_cross_product": 1025,
    "other_zero_fit": 0,
    "U_never_observed_among_zero_gamma": 291
  },
  "velocity_directions": [
    {
      "source_minute": 15,
      "target_minute": 30,
      "feature": "vel_mean",
      "interval_role": "calibration",
      "exploratory_variant": false,
      "feature_norm": 5.013210950775402,
      "raw_direction_cosine": -0.43709079802426,
      "retrospective_nonnegative_scale": 0.0,
      "retrospective_1d_gain_ceiling": 0.0
    },
    {
      "source_minute": 15,
      "target_minute": 30,
      "feature": "vel_pool",
      "interval_role": "calibration",
      "exploratory_variant": false,
      "feature_norm": 2.019427740780066,
      "raw_direction_cosine": -0.4187342893481278,
      "retrospective_nonnegative_scale": 0.0,
      "retrospective_1d_gain_ceiling": 0.0
    },
    {
      "source_minute": 15,
      "target_minute": 30,
      "feature": "vel_mean_minus_baseline0",
      "interval_role": "calibration",
      "exploratory_variant": true,
      "feature_norm": 0.3610869456468692,
      "raw_direction_cosine": 0.3724850095879898,
      "retrospective_nonnegative_scale": 0.17193256773159032,
      "retrospective_1d_gain_ceiling": 0.13874508236776487
    },
    {
      "source_minute": 15,
      "target_minute": 30,
      "feature": "vel_pool_minus_baseline0",
      "interval_role": "calibration",
      "exploratory_variant": true,
      "feature_norm": 0.2086329601846249,
      "raw_direction_cosine": 0.13424093325698502,
      "retrospective_nonnegative_scale": 0.10724157411626273,
      "retrospective_1d_gain_ceiling": 0.01802062816170631
    },
    {
      "source_minute": 30,
      "target_minute": 60,
      "feature": "vel_mean",
      "interval_role": "previously_scored_test",
      "exploratory_variant": false,
      "feature_norm": 5.224150125497064,
      "raw_direction_cosine": -0.43527057472451225,
      "retrospective_nonnegative_scale": 0.0,
      "retrospective_1d_gain_ceiling": 0.0
    },
    {
      "source_minute": 30,
      "target_minute": 60,
      "feature": "vel_pool",
      "interval_role": "previously_scored_test",
      "exploratory_variant": false,
      "feature_norm": 1.9676234928114305,
      "raw_direction_cosine": -0.43513086012994856,
      "retrospective_nonnegative_scale": 0.0,
      "retrospective_1d_gain_ceiling": 0.0
    },
    {
      "source_minute": 30,
      "target_minute": 60,
      "feature": "vel_mean_minus_baseline0",
      "interval_role": "previously_scored_test",
      "exploratory_variant": true,
      "feature_norm": 0.5145477778626754,
      "raw_direction_cosine": 0.21203928302131472,
      "retrospective_nonnegative_scale": 0.07391656060429279,
      "retrospective_1d_gain_ceiling": 0.044960657544193205
    },
    {
      "source_minute": 30,
      "target_minute": 60,
      "feature": "vel_pool_minus_baseline0",
      "interval_role": "previously_scored_test",
      "exploratory_variant": true,
      "feature_norm": 0.3227279015349597,
      "raw_direction_cosine": 0.37931889521029244,
      "retrospective_nonnegative_scale": 0.21082353612732146,
      "retrospective_1d_gain_ceiling": 0.1438828242635568
    },
    {
      "source_minute": 60,
      "target_minute": 120,
      "feature": "vel_mean",
      "interval_role": "previously_scored_test",
      "exploratory_variant": false,
      "feature_norm": 5.1755054455580805,
      "raw_direction_cosine": 0.263416269201511,
      "retrospective_nonnegative_scale": 0.0018403615041871699,
      "retrospective_1d_gain_ceiling": 0.06938813088004292
    },
    {
      "source_minute": 60,
      "target_minute": 120,
      "feature": "vel_pool",
      "interval_role": "previously_scored_test",
      "exploratory_variant": false,
      "feature_norm": 2.183883978351432,
      "raw_direction_cosine": 0.2819012276676014,
      "retrospective_nonnegative_scale": 0.0046674612464454165,
      "retrospective_1d_gain_ceiling": 0.07946830216050084
    },
    {
      "source_minute": 60,
      "target_minute": 120,
      "feature": "vel_mean_minus_baseline0",
      "interval_role": "previously_scored_test",
      "exploratory_variant": true,
      "feature_norm": 0.7918982461276431,
      "raw_direction_cosine": 0.24859880288999742,
      "retrospective_nonnegative_scale": 0.011351231152840308,
      "retrospective_1d_gain_ceiling": 0.06180136479833979
    },
    {
      "source_minute": 60,
      "target_minute": 120,
      "feature": "vel_pool_minus_baseline0",
      "interval_role": "previously_scored_test",
      "exploratory_variant": true,
      "feature_norm": 0.2981420487710026,
      "raw_direction_cosine": 0.329109195337725,
      "retrospective_nonnegative_scale": 0.039914445549114955,
      "retrospective_1d_gain_ceiling": 0.10831286245584482
    }
  ],
  "baseline_field_norms": {
    "vel_mean": 5.218895604273299,
    "vel_pool": 2.0973590942641755
  }
}
