# Velocity 步骤 1–3 结果（2026-09-23）

本轮按既定顺序完成：GFG 尺度校准、源侧测量稳定性、相邻时间短时几何预测。三项均不读取 RENGE confirmation；短时实验在读取 validation target 前先冻结 source vectors 与训练折尺度。

## 1. GFG 尺度校准

数据为 RENGE train day4→day5 的 14 个训练 TF 条件，使用条件 centroid 位移拟合正标量，未使用 validation/confirmation target。

| GFG 版本 | native 平均范数 | 训练折尺度 alpha | 校准后范数 | centroid cosine |
|---|---:|---:|---:|---:|
| frozen native | 34.57 | 0.0781（95% bootstrap 0.0710–0.0807） | 2.60 | 0.2063 |
| joint corrected v2 | 5.25 | 0.7754（0.7302–0.8835） | 2.48 | 0.2020 |

day4→day5 的真实条件 centroid 位移范数为 2.586。native scale=1 的预测范数为 33.35，说明原始 GFG 输出不能直接按天积分。校准修正了量纲，但真实 U 与局部 U shuffle 的方向几乎相同（0.2063 vs 0.2061），因此这是工程修正，不是 velocity 特异增益。

产物：`/data/yuchang/veloroute_velocity_scale_calibration_20260923/`。

## 2. 源侧测量稳定性

使用 RENGE train day4 的 1,200 个源细胞、2,000 个固定基因和同一个冻结 GFG checkpoint。除压力扰动外没有读取任何 target。

- raw S/U binomial count split（在 processed h5ad 的整数层上、再做每细胞 library normalization）：velocity 方向 cosine 中位数 **0.9663**，q10–q90 = 0.9509–0.9779，norm rank correlation 0.881。
- 局部 pre-GFG U shuffle：cosine 中位数约 **0.99983**，norm rank correlation 约 **0.993**。
- 归一化矩阵上的 U dropout（辅助压力测试）：cosine 约 0.99991–0.99998。

结论：GFG 对一次测序 count split 有中等到较高的测量稳定性；但对局部 U 细胞对应打乱几乎不变，提示当前 checkpoint 的输出主要由平滑状态/表达结构决定，不能把“稳定”解释为“携带命运信息”。raw-UMI count split 结果已经完成；此前的 normalized dropout 只作辅助，不替代 count split。

产物：`/data/yuchang/veloroute_velocity_stability_20260923_v4/`。

## 3. RENGE day4→day5 短时几何预测

这是当前可用的 ER-short 探针，不是 day2/day3，也不是逐细胞配对真值。尺度只在 14 个训练 TF 条件拟合；4 个 validation TF 只在冻结边界后用于评估。

validation 条件平均结果：

| 方法 | Energy V-statistic ↓ | pseudobulk MSE ↓ | NN-barycenter cosine |
|---|---:|---:|---:|
| identity | 0.736 | 0.129 | 0.000 |
| training mean-shift | **0.257** | **0.044** | 0.369 |
| GFG frozen native | 1.273 | 0.249 | 0.318 |
| GFG joint corrected v2 | 1.123 | 0.252 | 0.005 |
| z-only ridge | 2.154 | 0.229 | **0.859** |

动态特异性比较（正值才表示 real GFG 优于 shuffled-U）：

- frozen native vs pre-GFG U shuffle：energy gain **−0.00143**，95% exploratory CI [−0.00186, −0.00093]；pseudobulk gain −0.000161。
- joint corrected v2 vs pre-GFG U shuffle：energy gain **−0.02418**，CI [−0.03437, −0.00941]；pseudobulk gain −0.00596。

因此当前短时探针没有发现 GFG 优于 shuffled-U 的方向/分布增量；training mean-shift 反而明显更强。上述 CI 只有 4 个 validation TF，属于 exploratory condition bootstrap，不能当最终确认性统计。

产物：`/data/yuchang/veloroute_velocity_short_horizon_20260923_v2/`。

## 当前判定与下一步

步骤 1–3 没有给 router “可以使用 velocity 做命运路由”的阳性证据，但给出了两个可用的工程结论：

1. GFG 必须采用训练折校准尺度或单位方向，禁止 native scale=1 的天级积分；
2. count-split 可作为可靠度门控输入，稳定性与任务增量必须分开报告。

在完成 MeRLin Day0→Day21 或真正小时/24 小时纵向数据前，router 的 velocity 入口应保持 reliability-weighted prior / fail-closed gate，不应升级为 fate label 或长期内源场。

## 4. 最小替代：静态状态×条件 router pilot

为验证多峰路由是否必须依赖 velocity，另跑了一个 400-step RENGE pilot：router 只接收 `z0` 和条件 embedding，专家网络和 OT/EM 保留，GFG 完全不进入 forward。

4 个 validation TF 的平均 energy distance 为 **0.334**，variance ratio 为 **0.924**。这只是小模型、单 seed、400-step exploratory pilot，不能和 full-budget 三臂结果直接比较；它的意义是确认“状态×条件 router”可以独立运行并产生有结构的混合分布。

产物：`/data/yuchang/veloroute_static_state_condition_pilot_20260923_retry/`。

若要继续扩大 router 主线，优先做参数/预算匹配的 `z0×c` router 与原 GFG router 对照；velocity 先作为弱调制项，而不是主命运输入。
