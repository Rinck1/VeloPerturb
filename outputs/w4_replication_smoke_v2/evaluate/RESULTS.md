# W4 locked H1 evaluation

{
  "status": "SYNTHETIC_H1_PIPELINE_EVALUATED",
  "kind": "synthetic_fixture",
  "H1_TFs": [
    "SYNTH_TF0",
    "SYNTH_TF1",
    "SYNTH_TF2",
    "SYNTH_TF3",
    "SYNTH_TF4"
  ],
  "n_TFs": 5,
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
      "gain": -0.06732513009082164,
      "ci_low": -0.22542951560184635,
      "ci_high": 0.024390613995936526,
      "wilcoxon_p": 0.59375,
      "by_interval": [
        0.16548674812286837,
        -0.3001370083045116
      ],
      "independent_TFs": 5,
      "denominator_floors": 0,
      "supported": false
    },
    "vel_pool_Uswap": {
      "gain": -0.1441735535897704,
      "ci_low": -0.4553747498422431,
      "ci_high": 0.021961942243166114,
      "wilcoxon_p": 0.5,
      "by_interval": [
        0.18776622365864548,
        -0.4761133308381863
      ],
      "independent_TFs": 5,
      "denominator_floors": 0,
      "supported": false
    }
  },
  "H1_supported": false,
  "mean_TF_interval_sse": {
    "static": 0.33453469251481127,
    "vel_pool": 0.31312716846833755,
    "vel_mean": 0.3126446344065635,
    "vel_pool_Uswap": 0.29201424998934017,
    "pseudo_S": 0.1725563047629866,
    "direct_U_pool": 0.22337079493275516
  },
  "mean_TF_relative_gain_over_static": {
    "static": 0.0,
    "vel_pool": -0.06732513009082162,
    "vel_mean": -0.06391492910550002,
    "vel_pool_Uswap": 0.0768484234989488,
    "pseudo_S": 0.08449044824919948,
    "direct_U_pool": 0.05950971088020768
  },
  "frozen_grid_sha256": "57703129ebb2931ade024cb3e9626e452feda826a1dd90356ca4312dea75471c",
  "protocol_lock_sha256": "a323a7251370b4b05e1f7e39e768a774ecb2eeaa798f5bc717b1b8c0d767c21c",
  "formal_biological_claim": false
}
