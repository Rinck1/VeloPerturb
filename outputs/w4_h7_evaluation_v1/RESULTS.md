# W4 H7: direct U versus kinetic features

Prespecified auxiliary comparisons on the same frozen 16-TF, two-interval grid. These do not modify the H1 decision.

{
  "status": "H7_AUXILIARY_REPORTED",
  "n_TFs": 16,
  "H1_supported": false,
  "H1_success_rule_changed": false,
  "comparisons": {
    "direct_U_vs_static": {
      "gain": 0.0182167653497766,
      "ci_low": 0.00987739568081941,
      "ci_high": 0.02728070738597124,
      "wilcoxon_p": 0.0005035400390625,
      "by_interval": [
        0.023430926512254476,
        0.013002604187298721
      ],
      "independent_TFs": 16,
      "denominator_floors": 0,
      "supported": true
    },
    "velocity_vs_direct_U": {
      "gain": -0.011230805172886955,
      "ci_low": -0.01628206167238478,
      "ci_high": -0.0056552631462781005,
      "wilcoxon_p": 0.998931884765625,
      "by_interval": [
        -0.01112059453187055,
        -0.011341015813903359
      ],
      "independent_TFs": 16,
      "denominator_floors": 0,
      "supported": false
    }
  },
  "interval_relative_gain_over_static": {
    "static": [
      0.0,
      0.0
    ],
    "vel_pool": [
      0.012310331980383933,
      0.001661588373395361
    ],
    "vel_mean": [
      0.014095050317805167,
      0.0057691152449082005
    ],
    "vel_pool_Uswap": [
      0.004902328176939324,
      0.0027163459485923585
    ],
    "pseudo_S": [
      0.017857491875485566,
      0.00987169603524831
    ],
    "direct_U_pool": [
      0.023430926512254476,
      0.013002604187298721
    ]
  },
  "intervals": [
    [
      2,
      3
    ],
    [
      3,
      4
    ]
  ],
  "inference_unit": "TF_after_interval_averaging",
  "scope": "prespecified_auxiliary_no_new_model_or_subset"
}
