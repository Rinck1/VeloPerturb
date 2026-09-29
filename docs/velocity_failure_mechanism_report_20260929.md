# VeloRoute 阶段汇报：GFG velocity 为什么尚未赋能扰动预测？

日期：2026-09-29  
汇报重点：实验结论、失效机制与下一步判别实验。

## 一、核心结论

**当前最清晰的发现是：GFG velocity 的输出可重复性，与它对扰动终点的预测增量发生了分离。**

在 RENGE 中，GFG 输出对计数拆分较稳定，但对局部 U 细胞对应打乱也几乎不变；在相邻时间预测中，real GFG 没有优于 shuffled-U。在 MeRLin 的 Day0→Day21 克隆留出预测中，加入 GFG velocity 没有优于只使用表达状态的模型。

这把问题从“网络能不能用 velocity”推进到了更具体的机制问题：

> GFG 输出究竟携带了多少超出表达状态的动态信息？这些信息是否与扰动造成的未来变化同向，又能保持多长时间？

当前不宜将 velocity 直接解释为长期位移或命运标签。局部方向先验、路径约束和可靠度门控是下一步要验证的用途，而不是已经得到阳性支持的功能。

## 二、关键实验结果

### 2.1 RENGE：可重复，但对局部 U 对应关系不敏感

使用 1,200 个训练源细胞、2,000 个固定基因及同一个冻结 GFG checkpoint。

| 检验 | 方向 cosine 中位数 | 范数 rank correlation |
|---|---:|---:|
| 原始整数计数的 binomial count split | 0.9663 | 0.881 |
| 局部 pre-GFG U shuffle | 约 0.99983 | 约 0.993 |

count split 的方向 cosine 第 10–90 百分位为 0.9509–0.9779。

**机制意义：高稳定性本身不足以证明估计到了真实生物速度。** 局部打乱 U 后输出几乎不变，提示当前输出可能较多依赖 S、邻域平滑状态或 checkpoint 的流形先验，而较少依赖逐细胞的 S/U 配对。局部 shuffle 仍保留邻域结构，因此这一结果也不能单独证明模型完全忽略 U。

### 2.2 RENGE：数值尺度校准没有建立动态特异性

训练条件 day4→day5 的条件质心位移范数为 2.586。

| GFG 版本 | 原始输出平均范数 | 训练折校准系数 α | 校准后范数 | 与条件质心位移的 cosine |
|---|---:|---:|---:|---:|
| frozen native | 34.57 | 0.0781 | 2.60 | 0.2063 |
| joint corrected v2 | 5.25 | 0.7754 | 2.48 | 0.2020 |

frozen native 的 α bootstrap 95% CI 为 [0.0710, 0.0807]。真实 U 与局部 U shuffle 的质心方向 cosine 为 **0.2063 vs 0.2061**。

**机制意义：数值尺度可以校准，方向中的动态特异性却没有随之出现。** 原始范数不能直接赋予“PCA 单位/天”的物理含义；34.57 与 2.586 的差异是当前表示下的尺度差异，不是生物速度快了约 13 倍的测量结论。

### 2.3 RENGE：相邻一天的预测也没有出现稳定 velocity 增量

任务为 day4→day5；训练折拟合尺度，4 个留出 TF 用于 validation。该实验评估群体分布，不是逐细胞配对轨迹。

| 方法 | Energy distance ↓ | Pseudobulk MSE ↓ |
|---|---:|---:|
| identity | 0.736 | 0.129 |
| training mean-shift | **0.257** | **0.044** |
| GFG frozen native 分支 | 1.273 | 0.249 |
| GFG joint corrected v2 分支 | 1.123 | 0.252 |
| z-only ridge | 2.154 | 0.229 |

这里的 native 指冻结预训练 GFG 的版本身份，不应将该表简单归因为“没有做尺度处理”。该短时探针使用训练折尺度，和前一节质心校准也不是完全相同的拟合目标。

定义 gain = 对照误差 − real GFG 误差，正值表示 real 更好：

| real 对 shuffled-U | Energy gain | 95% 条件 bootstrap CI |
|---|---:|---:|
| frozen native | −0.00143 | [−0.00186, −0.00093] |
| joint corrected v2 | −0.02418 | [−0.03437, −0.00941] |

**机制意义：不能只用“Day0→Day21 太长”解释所有阴性结果；目前 day4→day5 的探针也未得到支持。** GFG 优于某个较弱的 z-only ridge，并不足以证明 velocity 有效；它还需要胜过均值平移和自身的 shuffled-U 对照。该组仅有 4 个 validation TF，CI 用于探索性定位。

### 2.4 MeRLin：克隆关联的长期结局预测没有增量

使用 Day0 SRR33960310 和 Day21 SRR33960308，共 376 个两侧 QC 后共享克隆，按 clone 分为 train/validation/confirmation = 226/75/75。当前结果来自 75 个 validation 克隆、720 个 Day0 源细胞。

预测目标是同克隆 Day21 细胞的五类表达程序均值：stress-like、neural-crest-like、lipid metabolism、PI3K signaling、ECM remodeling。它提供克隆关联的未来结局，不是逐个 Day0 细胞的独立命运真值。

| 输入 | normalized cell MSE ↓ | clone-equal MSE ↓ |
|---|---:|---:|
| z-only | **0.994951** | **0.014816** |
| z + simple velocity | 1.049118 | 0.015454 |
| z + GFG velocity | 1.003549 | 0.014867 |
| z + GFG shuffled-U | 1.012126 | 0.014894 |

四臂均使用同类线性 ridge probe。normalized cell MSE 是细胞加权 MSE 除以 validation 目标围绕自身均值的方差，因此约 1 表示接近该均值参照的误差水平；clone-equal MSE 先在每个克隆内平均，再给各克隆等权。

GFG 相对 z-only 的 normalized MSE 增加 0.00860，相对 shuffled-U 降低 0.00858。**两种加权口径下，GFG 都没有超越 z-only；而且整个 probe 的可预测性接近均值参照。** real 比 shuffle 略好的点估计，尚不能证明 U 中存在可复现的命运信息。

这一结果同时指向两个待区分的解释：velocity 没有提供额外信息，或当前状态表示、线性 probe 与表达程序结局尚不足以解析长期克隆差异。不能据此推导条件互信息为零。

SRR33960309 也是 Day21，后续可用于一致性核查；是否构成独立生物复本需根据样本设计判定，不能把新增测序 run 直接计为独立重复。

### 2.5 全模型结果：局部改善存在，但没有形成跨条件稳定优势

RENGE 单 seed、4 个 validation TF 的完整模型探索结果：

| 方法 | Energy distance ↓ | Variance ratio（越接近 1 越好） |
|---|---:|---:|
| static | 0.3803 | 0.8826 |
| GFG joint | **0.3672** | 0.8815 |
| GFG shuffled-U | 0.3926 | **0.9125** |

GFG 相对 static 的 gain 为 +0.01313，95% CI [−0.03471, 0.08704]；相对 shuffled-U 为 +0.02545，CI [−0.00144, 0.07805]。相对 static 的平均改善主要由 1 个条件推动，另外 3 个条件的 gain 为负；方差指标也没有同步改善。

**这个结果支持寻找条件特异的有效场景，但尚不支持普遍的多峰结构增益。** 本节是训练完整模型的结果，不能与上一节短时向量探针的绝对误差直接横向排名。

Kang 完整模型的 8-donor 汇总：

| GFG 对照 | Energy gain | 98.75% donor bootstrap CI |
|---|---:|---:|
| static matched | +0.02847 | [−0.04972, 0.17672] |
| shuffled-U | +0.00538 | [−0.00458, 0.01688] |

置信水平按该实验实际记录为 98.75%，不是 95%。预设主分析的多峰支持 donor 数为 0，表示当前模式定义和计数门槛下没有合格分析子集，不等于证明 Kang 没有多峰。全细胞类型的 2% 非劣检验也未通过。

## 三、失效机制：证据最集中在哪里？

### 机制 A：GFG 输出可能更接近状态依赖的切向表示，而非独立动态观测

项目中的 GFG velocity 通过编码器、解码器 JVP 和 PCA 投影得到。它是模型从 S/U 推断的局部向量，而非直接测量的未来表达导数。

最关键的证据组合是：count-split 一致性较高，同时局部 U shuffle 后几乎不变。这更值得优先调查，而不是把所有失败归结为测序噪声。

工作假说：S 和流形先验贡献了主要输出，U 中与细胞对应的增量较弱。需要通过 U 零化、不同范围的分层 shuffle、S-only 蒸馏和 U 残差探针来区分“输入冗余”与“估计器没有提取出增量”。

### 机制 B：内源 RNA 动态方向与扰动诱导位移没有自动对齐

研究对象可以概念性分解为：

$$
\Delta z = \int_{t_0}^{t_1} F_{\mathrm{base}}(z(t),t)\,dt
          + \int_{t_0}^{t_1} F_{\mathrm{response}}(z(t),t,c)\,dt.
$$

即使起点 velocity 能近似局部内源变化，它也不自动决定条件相关的响应场。RENGE 的质心 cosine 较低且 real/shuffle 相近，与这一解释一致。

需要进一步检验的是：velocity 能否预测去除共享/条件均值后的响应残差。不能仅靠整体方向相似性，把内源动态解释成扰动特异机制。

### 机制 C：局部方向的有效时间有限，但时间跨度不是唯一解释

起点导数不等于整段轨迹的平均速度：

$$
z(t_1)-z(t_0)=\int_{t_0}^{t_1} F(z(t),t,c)\,dt,
\qquad
F(z(t_0),t_0,c)(t_1-t_0)
$$

只是局部近似。Day0→Day21 可能跨越调控切换、增殖和选择等过程，clone-level 结局也包含这些累积作用。

但 RENGE day4→day5 已经出现阴性，因此不能简单将“缩短到一天”当作解决方案。需要同时检验时间跨度、方向敏感性和状态条件增量；更短时间窗是否有效仍待数据回答。

### 机制 D：当前结局与可观测状态之间的可预测性不足

MeRLin 中所有 normalized MSE 都接近 1。除了 velocity 冗余，还可能存在：训练/validation 状态差异、少量克隆细胞带来的结局均值噪声、线性 probe 表达能力不足，或表达程序并非最佳终点。

因此下一步要测量结局的可重复性与静态预测上限。只有基线能够稳定预测结局、velocity 仍无增量时，“velocity 的特异失效”才能与“整个任务难以预测”分离。

## 四、对 router 主线的含义

Router 的核心问题不是能否分出 K 个专家，而是源侧信息能否预测有意义的响应模式概率。

当前应分开两个贡献：

1. **静态状态 × 扰动条件能否改善多峰分布建模？** 这检验 router 本身。
2. **GFG velocity 能否进一步改善模式分配？** 这检验 velocity 的独立作用。

只有第二点成立，才有依据称为 velocity-assisted routing。专家出现分化、GFG 获得任务梯度或训练 loss 降低，都不能替代这一比较。

候选入口是可靠度加权的方向兼容度：

$$
q_k \propto q_{\mathrm{expr}}(k\mid z,c)
\exp\{\eta w_i\,\mathrm{compat}(v_i,d_k)\}.
$$

这里的模式方向与兼容度需由训练数据定义，并通过 real/shuffled 对照验证。该公式是待验证方案，不是现有阳性结果；尤其不能只因为 velocity 稳定，就令其可靠度权重变大。

## 五、下一步只做三个判别实验

| 实验 | 核心比较 | 能区分的机制 |
|---|---|---|
| 1. U 信息来源与 S 可预测性 | 固定 S，比较 real、局部/较宽范围分层 shuffle、zero-U；用 S 预测 GFG 输出，再测残差 | GFG 是否主要复现静态状态，或是否保留细胞特异动态信息 |
| 2. 相邻时间局部几何与预测 | 在已有 day4→day5 上测 real/null 局部转移；补齐 day2/day3 后独立复验；各时间对分别报告 | 动态方向是否对齐真实群体变化，以及效用随时间窗如何变化 |
| 3. MeRLin 结局可靠性与条件增量 | 先测 clone 结局的拆分一致性和静态预测上限，再做交叉拟合的 velocity 残差增量；按 clone bootstrap | 长期任务本身难预测，还是 velocity 在可预测部分之外没有增量 |

第三项先核查 Day21 308/309 的样本关系；若是技术重复，只作为测量一致性证据，不增加独立样本量。MeRLin confirmation 保留给事先固定的最终检验。

短期不再靠扩大 K 或增加 EM 复杂度解释阴性。首先得到“信息来源—方向—时间—结局”四个环节中具体失效的位置，再决定 velocity 进入 router、路径约束，还是仅作为质量特征。

## 六、向导师汇报的三句话

1. **我们观察到的不是简单的 velocity 数值不稳定，而是输出稳定性与未来响应预测能力脱节。**
2. **最强线索是：GFG 对局部 U 打乱近乎不变，短时预测没有 real/shuffle 优势，长期克隆预测也没有超越表达状态。**
3. **下一阶段要区分状态冗余、方向错配和结局可预测性三个机制；在此基础上再确定 velocity 的作用位置，而不是继续扩大模型。**

## 附：结果来源

正文数据来自项目现有实验，不使用其他论文的数值替代本项目结果。工作假说与后续用途均未当作已证机制。

- [Velocity 步骤 1–3 结果](velocity_steps123_results_20260923.md)：尺度校准、count-split、短时 probe。
- [MeRLin 纵向探针记录](merlin_day0_day21_results_20260929.md)：样本与任务说明。
- MeRLin 数值核对：`/data/yuchang/veloroute_merlin_day0_gfg_fate_probe_20260929_gpu_v6/summary.json`；指标定义见 [`run_merlin_day0_gfg_fate_probe.py`](../scripts/phase0/run_merlin_day0_gfg_fate_probe.py)。
- RENGE 短时任务与对照：`/data/yuchang/veloroute_velocity_short_horizon_20260923_v2/summary.json`。
- RENGE 完整模型探索：`/data/yuchang/veloroute_gainprobe_20260922/corrected_renge_v2/summary.json`。
- Kang donor-level 评估：`/data/yuchang/veloroute_kang_20260922/corrected_evaluation_v2/summary.json`。

本报告仅发布汇总数值和分析；原始细胞数据、模型权重和服务器结果目录不随文档上传。
