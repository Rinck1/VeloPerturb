# W4 day4→5 development

Four previously used development TFs; descriptive results only. Four algorithmic seed predictions are averaged before scoring.

{
  "status": "DEVELOPMENT_EVALUATED",
  "kind": "development",
  "inference": "descriptive_only_four_preused_TFs",
  "mse": {
    "static": 0.04931723579930698,
    "vel_pool": 0.048185150984493566,
    "vel_mean": 0.04887694370826023,
    "vel_pool_Uswap": 0.05024460476489749,
    "pseudo_S": 0.048817655324719225,
    "direct_U_pool": 0.048152167939122664
  },
  "relative_gain_over_static": {
    "static": 0.0,
    "vel_pool": 0.02295515546370752,
    "vel_mean": 0.008927752821315638,
    "vel_pool_Uswap": -0.018804155394360977,
    "pseudo_S": 0.010129936653805238,
    "direct_U_pool": 0.023623948935935557
  },
  "mean_TF_relative_gain": {
    "static": 0.0,
    "vel_pool": 0.03277038201363572,
    "vel_mean": 0.013163471617805282,
    "vel_pool_Uswap": -0.007786954265137343,
    "pseudo_S": 0.015389094157887827,
    "direct_U_pool": 0.028164522868071976
  },
  "per_TF_sse": {
    "static": {
      "LIN28A": 0.771259443421087,
      "NANOG": 0.7584734572910088,
      "POU5F1": 6.978909481627832,
      "ZIC3": 1.354804777521469
    },
    "vel_pool": {
      "LIN28A": 0.7590315732433388,
      "NANOG": 0.701966174046656,
      "POU5F1": 6.851710139876604,
      "ZIC3": 1.3243223097321133
    },
    "vel_mean": {
      "LIN28A": 0.7543432705897016,
      "NANOG": 0.7427885383210426,
      "POU5F1": 6.926973695921935,
      "ZIC3": 1.3512832368193681
    },
    "vel_pool_Uswap": {
      "LIN28A": 0.7731461534094316,
      "NANOG": 0.7549125816513628,
      "POU5F1": 7.154995158748716,
      "ZIC3": 1.3658670591699877
    },
    "pseudo_S": {
      "LIN28A": 0.7523312938407644,
      "NANOG": 0.741109608863662,
      "POU5F1": 6.923699170623991,
      "ZIC3": 1.346390991615425
    },
    "direct_U_pool": {
      "LIN28A": 0.7527533726952543,
      "NANOG": 0.7262336695836006,
      "POU5F1": 6.830333021334163,
      "ZIC3": 1.3211135242115144
    }
  },
  "seeds_averaged_before_scoring": true,
  "no_formal_replication_claim": true
}
