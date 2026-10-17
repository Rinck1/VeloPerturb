# U factorial attribution

Same original state/target for every arm. Equal-cell aggregation removes S-library weights; it does not causally separate technical depth from biological RNA content. All arms reported.

{
  "status": "EXPLORATORY_U_ATTRIBUTION_COMPLETE",
  "exploratory": true,
  "independent_confirmation": false,
  "biological_significance_test": false,
  "technical_vs_biological_total_identified": false,
  "original_results_changed": false,
  "models": 20,
  "mse_by_interval": {
    "identity": [
      0.5791282710352269,
      0.09413673424240092
    ],
    "v2_candidate": [
      0.5325883197976305,
      0.08536329800028165
    ],
    "depth_weighted__composition_current__total_current__velocity": [
      0.5325883197976305,
      0.08536329800028165
    ],
    "depth_weighted__composition_current__total_current__U_derivative": [
      0.5319137951034068,
      0.08404818051876366
    ],
    "depth_weighted__composition_current__total_baseline__velocity": [
      0.5549500558095367,
      0.08813663247730047
    ],
    "depth_weighted__composition_current__total_baseline__U_derivative": [
      0.557521089103835,
      0.08759712836937611
    ],
    "depth_weighted__composition_baseline__total_current__velocity": [
      0.49972836380191843,
      0.09289419063452042
    ],
    "depth_weighted__composition_baseline__total_current__U_derivative": [
      0.48346122477279346,
      0.09455941837038713
    ],
    "depth_weighted__composition_baseline__total_baseline__velocity": [
      0.5358524922386004,
      0.09171568855235672
    ],
    "depth_weighted__composition_baseline__total_baseline__U_derivative": [
      0.5311639919437213,
      0.09263873554519089
    ],
    "equal_cell__composition_current__total_current__velocity": [
      0.5148539237035664,
      0.09017870425541204
    ],
    "equal_cell__composition_current__total_current__U_derivative": [
      0.5148628931678101,
      0.09177736312932785
    ],
    "equal_cell__composition_current__total_baseline__velocity": [
      0.5580890420104179,
      0.08831912453360032
    ],
    "equal_cell__composition_current__total_baseline__U_derivative": [
      0.5635173896975187,
      0.08730003401718044
    ],
    "equal_cell__composition_baseline__total_current__velocity": [
      0.4811981520144577,
      0.10205582002452684
    ],
    "equal_cell__composition_baseline__total_current__U_derivative": [
      0.4682948798686332,
      0.10749403434410576
    ],
    "equal_cell__composition_baseline__total_baseline__velocity": [
      0.5276329542785588,
      0.09838291411994464
    ],
    "equal_cell__composition_baseline__total_baseline__U_derivative": [
      0.5237235062984517,
      0.10079962564287466
    ],
    "depth_weighted__S_kinetic": [
      0.5516418157881829,
      0.09069525540626629
    ],
    "equal_cell__S_kinetic": [
      0.5395669132774558,
      0.09332596625582683
    ]
  },
  "factor_contrasts": [
    {
      "aggregation": "depth_weighted",
      "field": "velocity",
      "contrast": "remove_current_composition",
      "control_mse_excess_by_interval": [
        -0.03285995599571212,
        0.00753089263423877
      ],
      "positive_means_current_factor_helped": true
    },
    {
      "aggregation": "depth_weighted",
      "field": "velocity",
      "contrast": "remove_current_total",
      "control_mse_excess_by_interval": [
        0.022361736011906164,
        0.0027733344770188156
      ],
      "positive_means_current_factor_helped": true
    },
    {
      "aggregation": "depth_weighted",
      "field": "U_derivative",
      "contrast": "remove_current_composition",
      "control_mse_excess_by_interval": [
        -0.04845257033061329,
        0.010511237851623473
      ],
      "positive_means_current_factor_helped": true
    },
    {
      "aggregation": "depth_weighted",
      "field": "U_derivative",
      "contrast": "remove_current_total",
      "control_mse_excess_by_interval": [
        0.02560729400042827,
        0.003548947850612452
      ],
      "positive_means_current_factor_helped": true
    },
    {
      "aggregation": "equal_cell",
      "field": "velocity",
      "contrast": "remove_current_composition",
      "control_mse_excess_by_interval": [
        -0.033655771689108716,
        0.011877115769114807
      ],
      "positive_means_current_factor_helped": true
    },
    {
      "aggregation": "equal_cell",
      "field": "velocity",
      "contrast": "remove_current_total",
      "control_mse_excess_by_interval": [
        0.04323511830685145,
        -0.0018595797218117216
      ],
      "positive_means_current_factor_helped": true
    },
    {
      "aggregation": "equal_cell",
      "field": "U_derivative",
      "contrast": "remove_current_composition",
      "control_mse_excess_by_interval": [
        -0.04656801329917687,
        0.015716671214777905
      ],
      "positive_means_current_factor_helped": true
    },
    {
      "aggregation": "equal_cell",
      "field": "U_derivative",
      "contrast": "remove_current_total",
      "control_mse_excess_by_interval": [
        0.04865449652970866,
        -0.0044773291121474185
      ],
      "positive_means_current_factor_helped": true
    }
  ],
  "fixed_readout_parameters": 2,
  "router_enabled": false
}
