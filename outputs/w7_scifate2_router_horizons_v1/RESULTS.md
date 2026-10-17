# Frozen router forecast horizons

W6 K4 models, no retraining; fixed daily-calibrated decay kernel. All origin-to-future windows within the observed r4 series, at most 15 hours. Predictions are built before evaluation.

[
  {
    "source_hour": 5,
    "target_hour": 10,
    "horizon_hours": 5,
    "router_gain_vs_static": 0.010387744379993014,
    "router_gain_vs_global_translation": 0.05705667724369223,
    "analytic_mean_gain_vs_static": -0.009220384306533491
  },
  {
    "source_hour": 10,
    "target_hour": 15,
    "horizon_hours": 5,
    "router_gain_vs_static": 0.010104519813907253,
    "router_gain_vs_global_translation": -0.009758577038289085,
    "analytic_mean_gain_vs_static": 0.34492302007578146
  },
  {
    "source_hour": 15,
    "target_hour": 20,
    "horizon_hours": 5,
    "router_gain_vs_static": 0.016503489569013374,
    "router_gain_vs_global_translation": -0.014870137255345695,
    "analytic_mean_gain_vs_static": 0.0755463131430254
  },
  {
    "source_hour": 5,
    "target_hour": 15,
    "horizon_hours": 10,
    "router_gain_vs_static": 0.01135828093979041,
    "router_gain_vs_global_translation": 0.08289169449638456,
    "analytic_mean_gain_vs_static": -0.012947841738847998
  },
  {
    "source_hour": 10,
    "target_hour": 20,
    "horizon_hours": 10,
    "router_gain_vs_static": 0.01206167135991787,
    "router_gain_vs_global_translation": -0.023705058410252056,
    "analytic_mean_gain_vs_static": 0.3900147877618567
  },
  {
    "source_hour": 5,
    "target_hour": 20,
    "horizon_hours": 15,
    "router_gain_vs_static": 0.008180177243966247,
    "router_gain_vs_global_translation": 0.08962328212237713,
    "analytic_mean_gain_vs_static": -0.03929132236247468
  }
]
