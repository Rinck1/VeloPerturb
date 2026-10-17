# Source-only no-JVP head: day4→5 development

Fixed three-seed mean predictions. Matched 1600-parameter heads for velocity, direct U, and S-only population features. No future labels in head training; future training TFs supervise ridge only.

{
  "status": "NOJVP_HEAD_DEVELOPMENT_EVALUATED",
  "kind": "development",
  "mse": {
    "static": 0.04931723579930698,
    "vel_pool": 0.048185150984493566,
    "vel_mean": 0.04887694370826023,
    "vel_pool_Uswap": 0.05024460476489749,
    "pseudo_S": 0.048817655324719225,
    "direct_U_pool": 0.048152167939122664,
    "learned_velocity": 0.04846624606409983,
    "learned_velocity_Uswap": 0.04949827108138112,
    "learned_direct_U": 0.0480944854756333,
    "learned_S_only": 0.04882341397154426
  },
  "relative_gain_over_static": {
    "static": 0.0,
    "vel_pool": 0.02295515546370752,
    "vel_mean": 0.008927752821315638,
    "vel_pool_Uswap": -0.018804155394360977,
    "pseudo_S": 0.010129936653805238,
    "direct_U_pool": 0.023623948935935557,
    "learned_velocity": 0.017255422397763565,
    "learned_velocity_Uswap": -0.0036708318935563256,
    "learned_direct_U": 0.024793569709575247,
    "learned_S_only": 0.010013169224899254
  },
  "mean_TF_relative_gain": {
    "static": 0.0,
    "vel_pool": 0.03277038201363572,
    "vel_mean": 0.013163471617805282,
    "vel_pool_Uswap": -0.007786954265137343,
    "pseudo_S": 0.015389094157887827,
    "direct_U_pool": 0.028164522868071976,
    "learned_velocity": 0.027307773786311707,
    "learned_velocity_Uswap": 9.70805337117698e-05,
    "learned_direct_U": 0.02803287848177373,
    "learned_S_only": 0.016359257345020608
  },
  "learned_velocity_gain_over_analytic": -0.005833645300742596,
  "no_formal_replication_claim": true,
  "head_future_supervision": false,
  "router_enabled": false,
  "interpretation": "source-population feature denoising, not independent true-velocity validation"
}
