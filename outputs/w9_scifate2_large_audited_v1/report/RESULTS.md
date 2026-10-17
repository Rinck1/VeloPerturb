# sci-FATE2 router scaling

One router, K response experts; all K/width/checkpoint results are reported. The global mean and W6 evaluation samples are fixed. This is same-series development, not a new independent holdout. Seed ensembles mix weighted candidates, not endpoint averages.

{
  "status": "SCIFATE2_ROUTER_SCALING_COMPLETE",
  "training_jobs": 12,
  "cases": 12,
  "best_observed_development_configuration": {
    "setting": "large_k8_w1216",
    "experts": 8,
    "hidden": 1216,
    "arm": "velocity_router",
    "step": 1000,
    "energy_distance": 0.6143305972094613,
    "mean_mse": 0.13660101223499263,
    "absolute_log_variance_error": 0.17499128748960213,
    "covariance_relative_error": 0.518868166790825,
    "parameters": 18516952,
    "training_seconds": 25.837961796671152,
    "peak_allocated_bytes": 1360543744,
    "mean_active_modes": 8.0,
    "mean_effective_modes": 5.081021169285597,
    "mean_routing_information": 0.105901847204233
  },
  "velocity_comparisons": [
    {
      "setting": "small_k8_w128",
      "experts": 8,
      "hidden": 128,
      "step": 1000,
      "velocity_energy": 0.6269850737918077,
      "static_energy": 0.6290703559211354,
      "relative_velocity_gain": 0.003314863130490786
    },
    {
      "setting": "small_k8_w128",
      "experts": 8,
      "hidden": 128,
      "step": 4000,
      "velocity_energy": 0.6210221411383898,
      "static_energy": 0.6201804626397447,
      "relative_velocity_gain": -0.0013571509412964033
    },
    {
      "setting": "small_k8_w128",
      "experts": 8,
      "hidden": 128,
      "step": 12000,
      "velocity_energy": 0.6210742134745187,
      "static_energy": 0.6205250081418333,
      "relative_velocity_gain": -0.0008850655903942517
    },
    {
      "setting": "large_k8_w1216",
      "experts": 8,
      "hidden": 1216,
      "step": 1000,
      "velocity_energy": 0.6143305972094613,
      "static_energy": 0.6186188999946906,
      "relative_velocity_gain": 0.006932059116309119
    },
    {
      "setting": "large_k8_w1216",
      "experts": 8,
      "hidden": 1216,
      "step": 4000,
      "velocity_energy": 0.6250849322608847,
      "static_energy": 0.6228893185736464,
      "relative_velocity_gain": -0.003524885757017149
    },
    {
      "setting": "large_k8_w1216",
      "experts": 8,
      "hidden": 1216,
      "step": 12000,
      "velocity_energy": 0.6235484546228834,
      "static_energy": 0.6219952249308905,
      "relative_velocity_gain": -0.002497173016345
    }
  ],
  "K_doubling_curve": [],
  "practical_K_saturation_candidate": null,
  "saturation_rule": {
    "relative_improvement_threshold": 0.002,
    "consecutive_doublings": 2
  },
  "no_saturation_means_tested_range_only": true,
  "global_mean_frozen": true,
  "scope": "same_series_development",
  "mean_anchor_max_error": 6.536331624022473e-08,
  "frozen_evaluation_summaries": [
    {
      "path": "outputs/w9_scifate2_large_audited_v1/evaluations/small_k8_w128/static_router/step1000/summary.json",
      "sha256": "ca3d6cc27e67277c6bb8fb62a787a8f6ba1067d3fa9c442ad3ab6baed75ac1f4"
    },
    {
      "path": "outputs/w9_scifate2_large_audited_v1/evaluations/small_k8_w128/static_router/step4000/summary.json",
      "sha256": "f5602a6c07b9fb520392e8f1d56f2c5bd74727d163ba36541fd4b7178e81ae85"
    },
    {
      "path": "outputs/w9_scifate2_large_audited_v1/evaluations/small_k8_w128/static_router/step12000/summary.json",
      "sha256": "8274bfe31c06e141a29cb3a488647fbfbe2761d435b980a3f078ab0160ab62bb"
    },
    {
      "path": "outputs/w9_scifate2_large_audited_v1/evaluations/small_k8_w128/velocity_router/step1000/summary.json",
      "sha256": "066639824c7d57ef91a304f67b2cbf315be2793f11d5199fe995868c134a281d"
    },
    {
      "path": "outputs/w9_scifate2_large_audited_v1/evaluations/small_k8_w128/velocity_router/step4000/summary.json",
      "sha256": "2141ffcf4e095d98096aada34551955eb1bf3e9b7f1dd0a7ee8cb6a9b370b165"
    },
    {
      "path": "outputs/w9_scifate2_large_audited_v1/evaluations/small_k8_w128/velocity_router/step12000/summary.json",
      "sha256": "0da28d4c8d2d43c9567e93d3b3bd0998197c30138a45053cd3590c0c93221b4b"
    },
    {
      "path": "outputs/w9_scifate2_large_audited_v1/evaluations/large_k8_w1216/static_router/step1000/summary.json",
      "sha256": "a6ed7086540c681b007a19088944e48aa6ccb90adf95ba65fa26276cff3eabd1"
    },
    {
      "path": "outputs/w9_scifate2_large_audited_v1/evaluations/large_k8_w1216/static_router/step4000/summary.json",
      "sha256": "a5f57015b24867bb0ce960cbf37033d0b6421cde282afa51d7aa2508bd7882cc"
    },
    {
      "path": "outputs/w9_scifate2_large_audited_v1/evaluations/large_k8_w1216/static_router/step12000/summary.json",
      "sha256": "2933c5a861c26b1da22aa2c00eb5f385920954d5c02882ed977fd2a81cd92d4b"
    },
    {
      "path": "outputs/w9_scifate2_large_audited_v1/evaluations/large_k8_w1216/velocity_router/step1000/summary.json",
      "sha256": "d57ce2db88153b5eb32623a607e096476df28d696946afaff90c839bb6fa0a49"
    },
    {
      "path": "outputs/w9_scifate2_large_audited_v1/evaluations/large_k8_w1216/velocity_router/step4000/summary.json",
      "sha256": "707b3dab9b96f1bb98d3a8d23b55de62f7a0984c6eb81e615ee23d5f46b1641e"
    },
    {
      "path": "outputs/w9_scifate2_large_audited_v1/evaluations/large_k8_w1216/velocity_router/step12000/summary.json",
      "sha256": "550d9a460f2075581ad3669419924ef7c6447f468bfa5494d167728e520af44d"
    }
  ]
}
