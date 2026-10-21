# 最终小 ODE：动力学路由在起什么作用

2026-10-21，冻结 W13 最终 ODE，不更新任何参数。六个开发窗口、三 seed，
所有42份预测先冻结再评分；原 soft/static/shuffled 的18项 Energy 与原记录完全一致（差为0）。

## 核心结果

下表均为相对本线 static 的 Energy 改善，正值更好。

| 预测视野 | 原逐细胞 soft 路由 | 群体平均后重新积分 | 均匀路由后重新积分 |
|---|---:|---:|---:|
| 5h | +3.723% | +3.735% | +1.094% |
| 10h | +6.358% | +6.449% | −3.367% |
| 15h | +2.415% | +1.569% | −33.348% |

群体平均路由分别保留原收益的100.33%、101.43%、64.96%。只在终态改变权重、不重新积分，
分别保留100.44%、102.15%、68.95%；两种干预方向一致。

因此，5–10h的主要优势不依赖每颗细胞不同的q，而依赖源群体提供的非均匀专家配比。
15h逐细胞权重仍有部分额外贡献，但只有一个重叠序列窗口，不能由此认定真实单细胞命运。
均匀化明显损害长视野，说明训练所得专家并不是可以任意等权互换的分量；
这也不等于这些专家已经对应生物学谱系。

这支持后续 L2 优先检验动力学群体 token；L3 保留逐细胞 router，并继续与群体平均路由比较。
本实验不单独证明 velocity 相对所有 U/S 表示的特异性，也不是独立数据复现。

## 实验口径与记录

重加权只改变终态q并做共同均值平移；重新积分则在每次 RK4 群体中心化中使用新q。
全源群体积分后才抽样，三 seed 等权混合，沿用原均值锚、输入与模型哈希。

[逐窗口指标](../../outputs/w15_l1_routing_20261021_v1/metrics.csv) ·
[视野汇总](../../outputs/w15_l1_routing_20261021_v1/horizon_summary.csv) ·
[原结果精确复现](../../outputs/w15_l1_routing_20261021_v1/reference_reproduction.json)

<details>
<summary>原始运行汇总</summary>

```json

{
  "status": "FROZEN_FINAL_L1_ROUTING_INTERVENTIONS_COMPLETE",
  "optimizer_steps": 0,
  "scope": "same_series_development",
  "interventions": [
    "soft",
    "cohort_reweight",
    "uniform_reweight",
    "cohort_reintegrate",
    "uniform_reintegrate",
    "static",
    "shuffled"
  ],
  "horizon_summary": [
    {
      "horizon_hours": 5,
      "variant": "soft",
      "energy": 0.58950417173402,
      "gain_vs_static": 0.037225267182402155,
      "retained_gain_fraction": 1.0
    },
    {
      "horizon_hours": 5,
      "variant": "cohort_reweight",
      "energy": 0.5894029272423656,
      "gain_vs_static": 0.03739061908842023,
      "retained_gain_fraction": 1.0044419266410596
    },
    {
      "horizon_hours": 5,
      "variant": "uniform_reweight",
      "energy": 0.6053317611859613,
      "gain_vs_static": 0.011375741536271744,
      "retained_gain_fraction": 0.3055919378773325
    },
    {
      "horizon_hours": 5,
      "variant": "cohort_reintegrate",
      "energy": 0.58942898466572,
      "gain_vs_static": 0.037348062258442005,
      "retained_gain_fraction": 1.003298702342099
    },
    {
      "horizon_hours": 5,
      "variant": "uniform_reintegrate",
      "energy": 0.6056003684109962,
      "gain_vs_static": 0.010937053802213659,
      "retained_gain_fraction": 0.29380726130513934
    },
    {
      "horizon_hours": 5,
      "variant": "static",
      "energy": 0.6122970946784333,
      "gain_vs_static": 0.0,
      "retained_gain_fraction": 0.0
    },
    {
      "horizon_hours": 5,
      "variant": "shuffled",
      "energy": 0.6180581157465573,
      "gain_vs_static": -0.00940886559513987,
      "retained_gain_fraction": -0.2527548170181519
    },
    {
      "horizon_hours": 10,
      "variant": "soft",
      "energy": 1.0270316598504134,
      "gain_vs_static": 0.06358379977414275,
      "retained_gain_fraction": 1.0
    },
    {
      "horizon_hours": 10,
      "variant": "cohort_reweight",
      "energy": 1.025533941142366,
      "gain_vs_static": 0.06494937409519252,
      "retained_gain_fraction": 1.0214767649291243
    },
    {
      "horizon_hours": 10,
      "variant": "uniform_reweight",
      "energy": 1.130265051515848,
      "gain_vs_static": -0.030541263881493697,
      "retained_gain_fraction": -0.4803309017388063
    },
    {
      "horizon_hours": 10,
      "variant": "cohort_reintegrate",
      "energy": 1.0260347456338623,
      "gain_vs_static": 0.06449275580647174,
      "retained_gain_fraction": 1.0142954028472302
    },
    {
      "horizon_hours": 10,
      "variant": "uniform_reintegrate",
      "energy": 1.1336983678132517,
      "gain_vs_static": -0.03367165715666931,
      "retained_gain_fraction": -0.5295634623327806
    },
    {
      "horizon_hours": 10,
      "variant": "static",
      "energy": 1.096768359627589,
      "gain_vs_static": 0.0,
      "retained_gain_fraction": 0.0
    },
    {
      "horizon_hours": 10,
      "variant": "shuffled",
      "energy": 1.1093255659135615,
      "gain_vs_static": -0.011449278396612694,
      "retained_gain_fraction": -0.18006596707466208
    },
    {
      "horizon_hours": 15,
      "variant": "soft",
      "energy": 1.0078334305557366,
      "gain_vs_static": 0.024147598138747894,
      "retained_gain_fraction": 1.0
    },
    {
      "horizon_hours": 15,
      "variant": "cohort_reweight",
      "energy": 1.0155763545216399,
      "gain_vs_static": 0.016650376156947533,
      "retained_gain_fraction": 0.6895251470261088
    },
    {
      "horizon_hours": 15,
      "variant": "uniform_reweight",
      "energy": 1.3697695991271939,
      "gain_vs_static": -0.326303447354114,
      "retained_gain_fraction": -13.51287384688246
    },
    {
      "horizon_hours": 15,
      "variant": "cohort_reintegrate",
      "energy": 1.0165717691977267,
      "gain_vs_static": 0.015686548431991354,
      "retained_gain_fraction": 0.649611126616369
    },
    {
      "horizon_hours": 15,
      "variant": "uniform_reintegrate",
      "energy": 1.3771803632532897,
      "gain_vs_static": -0.33347904974318165,
      "retained_gain_fraction": -13.810029793732243
    },
    {
      "horizon_hours": 15,
      "variant": "static",
      "energy": 1.0327724035248433,
      "gain_vs_static": 0.0,
      "retained_gain_fraction": 0.0
    },
    {
      "horizon_hours": 15,
      "variant": "shuffled",
      "energy": 1.0239387541549334,
      "gain_vs_static": 0.008553335991318911,
      "retained_gain_fraction": 0.35421063172299505
    }
  ],
  "reweighting_and_reintegration_separate": true,
  "new_biological_replication": false
}
```

</details>
