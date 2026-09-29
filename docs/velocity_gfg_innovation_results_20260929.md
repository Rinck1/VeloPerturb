# GFG velocity 残差信息诊断：稳定输出是否包含任务相关方向？

日期：2026-09-29  
性质：source-side GFG 诊断，探索性，不打开 confirmation。

## 结论

在当前 RENGE 与 pancreas frozen pack 上，GFG decoder-JVP velocity 的大部分输出可以由源侧 latent state `z` 预测。将这部分 `E[v|z]` 去掉后，剩余 velocity 与 held-out target 的局部邻域方向没有稳定正向一致性。

这进一步把项目的阶段性结论收窄为：

> GFG velocity 的输出具有结构性和可重复性，但当前没有证据表明其 z-不可解释残差沿着扰动后的局部变化方向。

这不是条件互信息为零的证明。诊断中的 target 方向是同条件 target latent 的 kNN 几何代理，不是逐细胞配对真值；结果用于定位机制，不用于 fate claim。

## RENGE：GFG decoder-JVP

数据范围：day4 source，14 个训练 TF、4 个 validation TF；velocity 由 frozen GFG checkpoint 的 decoder-JVP 产生。训练集用 5-fold cross-fit 拟合 `z → v`，validation 只使用 train-fit 的 Ridge，不读取 validation target 进行任何拟合。

| split | z→v cross-fit / train-fit R² | velocity RMS | residual 与目标局部方向 cosine |
|---|---:|---:|---|
| train | 0.893 | 4.943 | −0.179 至 −0.028（14 个条件） |
| validation | 0.890 | 5.029 | LIN28A −0.048；NANOG −0.044；POU5F1 −0.033；ZIC3 −0.105 |

validation 的原始 velocity 与目标局部方向 cosine 为：

- LIN28A：`+0.327`
- NANOG：`+0.332`
- POU5F1：`+0.290`
- ZIC3：`+0.324`

但对应的 `E[v|z]` cosine 为 `+0.330、+0.336、+0.293、+0.331`，与原始 velocity 几乎相同；去掉 z 可预测部分后的 residual cosine 变为 `−0.048、−0.044、−0.033、−0.105`。

因此目前观察到的方向一致性主要由 state-predictable component 提供，而不是由 velocity 的 z-残差提供。

## Pancreas：自然转变对照

pancreas 数据作为非扰动发育转变对照，训练条件为 Alpha/Beta，validation 条件为 Delta/Epsilon。

| split | z→v R² | velocity RMS | residual cosine |
|---|---:|---:|---|
| train | 0.526 | 15.669 | Alpha −0.041；Beta −0.067 |
| validation | 0.305 | 28.583 | Delta −0.001；Epsilon +0.114 |

Pancreas validation 的 Epsilon 出现了轻微正向 residual cosine，但只有一个条件，不能构成稳定机制证据。它提示真正转变系统中可能存在局部动态残差，但仍需要时间留出或独立数据复现。

## 加入扰动条件后的对照

为了区分“velocity 被 z 吸收”和“velocity 只是没有加入条件 c”这两个解释，进一步用冻结的 ESM2 条件嵌入拟合 `(z, c) → v`。条件嵌入只来自训练/验证输入条件表，不使用目标表达。

| predictor | train R² | validation R² | validation residual RMS |
|---|---:|---:|---:|
| `z → v` | 0.89256 | 0.89042 | 0.47821 |
| `(z,c) → v` | 0.89228 | 0.89202 | 0.47470 |

`(z,c)` 只带来约 0.0016 的 validation R² 增加。四个 held-out TF 的 residual cosine（`(z,c) → v`）为：

- LIN28A：−0.0251
- NANOG：−0.0364
- POU5F1：−0.0205
- ZIC3：−0.0955

因此当前结果不支持“GFG 失败只是因为 velocity encoder 没有看到条件 c”这一解释。条件输入对 GFG 输出的额外解释很小，且去除 `(z,c)` 可解释部分后的方向仍未呈现稳定正向目标对齐。

## 机制解释

目前结果支持以下排序：

1. `z` 已吸收 GFG velocity 中的大部分可预测结构；
2. GFG velocity 与目标局部方向的正向 cosine 主要来自 state-predictable component；
3. z-残差目前没有稳定沿着目标方向；
4. 自然发育数据中的单条件弱阳性值得继续检验，但不能外推为扰动命运信号。

这比“velocity 是噪声”更精确，也比“velocity 已携带 hidden fate information”更谨慎：当前数据只支持输出结构化，不支持残差任务增量。

## 下一步判别实验

1. 在 RENGE 上加入条件编码 `c`，比较 train-only cross-fit 的 `z→v` 与 `(z,c)→v`；如果 `(z,c)` 已解释残差，说明 router 的条件输入可以吸收该信号。
2. 加入 pre-GFG U shuffle、U zero 和 post-velocity local shuffle；区分输入层、GFG 表示层和 router 配对层的作用。
3. 不使用逐细胞 pseudo-pair，改用 held-out condition 的 energy、variance 和 pseudobulk residual 指标。
4. MeRLin 309 仅作为 Day21 技术复本/深度敏感性审计，不视为独立生物复本；309 定量完成后再做 frozen GFG replication。

## 运行边界

- 本结果只读取 train/validation source 与 validation target 的局部几何用于评估。
- 没有读取 MeRLin confirmation，也没有将 target U/S/velocity 作为输入。
- RENGE pack 的 `formal_ready=false`，因此本结果属于 exploratory mechanism diagnostic。
- 诊断代码：[`gfg_innovation_diagnostic.py`](../scripts/phase0/gfg_innovation_diagnostic.py)。
