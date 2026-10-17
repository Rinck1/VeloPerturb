# W4 locked H1 evaluation

{
  "status": "H1_REPLICATION_EVALUATED",
  "kind": "formal_replication",
  "H1_TFs": [
    "ETS2",
    "ETV4",
    "FOXH1",
    "ID1",
    "JARID2",
    "LIN28A",
    "MYC",
    "MYCN",
    "NANOG",
    "NR5A2",
    "PDLIM1",
    "POU5F1",
    "PRDM14",
    "RUNX1T1",
    "TRIM24",
    "TRIM25"
  ],
  "n_TFs": 16,
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
  "arms": [
    "static",
    "vel_pool",
    "vel_mean",
    "vel_pool_Uswap",
    "pseudo_S",
    "direct_U_pool"
  ],
  "algorithm_seeds": [
    0,
    1,
    2,
    3
  ],
  "seeds_averaged_before_scoring": true,
  "independent_inference_unit": "TF; paired intervals averaged within TF; seeds/cells are not independent units",
  "primary_comparisons": {
    "static": {
      "gain": 0.0069859601768896456,
      "ci_low": -4.8489859331748313e-05,
      "ci_high": 0.014478452285054094,
      "wilcoxon_p": 0.0583343505859375,
      "by_interval": [
        0.012310331980383933,
        0.001661588373395361
      ],
      "independent_TFs": 16,
      "denominator_floors": 0,
      "supported": false
    },
    "vel_pool_Uswap": {
      "gain": 0.0031766231141238032,
      "ci_low": -0.00332921907330952,
      "ci_high": 0.009279031361069503,
      "wilcoxon_p": 0.1612548828125,
      "by_interval": [
        0.007408003803444609,
        -0.0010547575751969995
      ],
      "independent_TFs": 16,
      "denominator_floors": 0,
      "supported": false
    }
  },
  "H1_supported": false,
  "mean_TF_interval_sse": {
    "static": 1.218942494420531,
    "vel_pool": 1.2075653688007926,
    "vel_mean": 1.2039375657971612,
    "vel_pool_Uswap": 1.2161979271566539,
    "pseudo_S": 1.2041774722943734,
    "direct_U_pool": 1.194836984950556
  },
  "mean_TF_relative_gain_over_static": {
    "static": 0.0,
    "vel_pool": 0.0069859601768896456,
    "vel_mean": 0.009932082781356684,
    "vel_pool_Uswap": 0.0038093370627658423,
    "pseudo_S": 0.013864593955366936,
    "direct_U_pool": 0.0182167653497766
  },
  "frozen_grid_sha256": "e9c67d69c802d80f8f3b7509c7eb2ea0df7293e84d4b511046fc4f0a938e0c87",
  "protocol_lock_sha256": "8788a75da0f8ec5c86c936f4f4dbcb4333ce729f88d633bed28922aedbed0dfb",
  "formal_biological_claim": true
}
