# Exploratory v2 mechanism controls

The nonlinear S-only arm uses no current U. The U-total control uses current U depth but baseline gene composition. No v2 primary score is changed.

{
  "status": "EXPLORATORY_REPAIR_MECHANISM_COMPLETE",
  "exploratory": true,
  "original_grid_changed": false,
  "independent_confirmation": false,
  "matched_parameters": 2,
  "candidate_beats_control_mean": {
    "S_kinetic": true,
    "U_derivative": false,
    "U_total_only": false
  },
  "candidate_beats_control_both": {
    "S_kinetic": true,
    "U_derivative": false,
    "U_total_only": false
  },
  "mse_by_interval": {
    "candidate": [
      0.5325883197976305,
      0.08536329800028165
    ],
    "S_kinetic": [
      0.5516418157881829,
      0.09069525540626629
    ],
    "U_derivative": [
      0.5319137951034068,
      0.08404818051876366
    ],
    "U_total_only": [
      0.4997283638019186,
      0.09289419063452042
    ]
  },
  "calibration_interval": [
    15,
    30
  ],
  "readouts": {
    "S_kinetic": {
      "coefficients": [
        -0.006764522890797603,
        0.027070879806366813
      ],
      "source_column_norms": [
        1.8996134968801373,
        0.047492819165894054
      ],
      "positive_columns": [
        false,
        false
      ],
      "ridge": 1.0,
      "parameter_count": 2
    },
    "U_derivative": {
      "coefficients": [
        -0.006693346472881205,
        0.05171405145242505
      ],
      "source_column_norms": [
        1.8996134968801373,
        0.2022088689807193
      ],
      "positive_columns": [
        false,
        false
      ],
      "ridge": 1.0,
      "parameter_count": 2
    },
    "U_total_only": {
      "coefficients": [
        -0.005087914884829368,
        0.06091978063730059
      ],
      "source_column_norms": [
        1.8996134968801373,
        0.21631468587203737
      ],
      "positive_columns": [
        false,
        true
      ],
      "ridge": 1.0,
      "parameter_count": 2
    }
  },
  "U_total_control_uses_current_total_U_but_not_current_U_gene_composition": true
}
