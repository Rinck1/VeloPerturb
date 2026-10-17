# Local velocity router expansion

The original analytic population mean is frozen; experts change only distribution structure. Energy uses fixed source/target subsamples and an equally weighted seed mixture.

{
  "status": "LOCAL_VELOCITY_ROUTER_EXPANSION_COMPLETE",
  "metrics": {
    "identity": {
      "energy_distance": 0.782598466372607,
      "mean_mse": 0.17187010785208526,
      "variance_ratio": 1.1367842014163545,
      "covariance_relative_error": 0.5931846726206381
    },
    "global_translation": {
      "energy_distance": 0.6351569889677151,
      "mean_mse": 0.1366010130474364,
      "variance_ratio": 1.1368235824676676,
      "covariance_relative_error": 0.5931836217005121
    },
    "k1": {
      "energy_distance": 0.6238460198644061,
      "mean_mse": 0.13660101227085228,
      "variance_ratio": 1.0178143030106133,
      "covariance_relative_error": 0.5304060674745762
    },
    "static_router": {
      "energy_distance": 0.6234815575006701,
      "mean_mse": 0.13660101244865766,
      "variance_ratio": 1.0176436493662033,
      "covariance_relative_error": 0.5376960893488566
    },
    "velocity_router": {
      "energy_distance": 0.6162837534842277,
      "mean_mse": 0.13660101311893333,
      "variance_ratio": 1.009663726609051,
      "covariance_relative_error": 0.5281168337109531
    }
  },
  "velocity_router_energy_gain": {
    "identity": 0.2125160219892308,
    "global_translation": 0.029714284517534773,
    "k1": 0.012122007898394682,
    "static_router": 0.01154453396391708
  },
  "velocity_router_energy_gain_by_interval": {
    "global_translation": [
      0.05705666678668608,
      -0.009758560029378582,
      -0.014870157216321678
    ],
    "k1": [
      0.01975533986214395,
      -0.009219539819617983,
      0.011821622383963004
    ],
    "static_router": [
      0.010387744509458896,
      0.010104506329128737,
      0.01650346880280773
    ]
  },
  "mean_anchor_max_error": 6.779486986600494e-08,
  "seeds": [
    0,
    1,
    2
  ],
  "weighted_prediction_ensemble": true,
  "r4_used_for_fitting": false,
  "gfg_jvp_enabled": false,
  "individual_fate_truth_used": false,
  "global_mean_frozen": true,
  "distribution_energy_uses_fixed_subsamples": true
}
