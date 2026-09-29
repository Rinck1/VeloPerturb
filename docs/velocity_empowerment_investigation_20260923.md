# Velocity 如何赋能扰动任务：身份、边界与验证方案

版本：2026-09-23  
对象：VeloRoute，速度估计器为 GFG decoder-JVP

## 结论先行

当前没有证据证明 GFG velocity 能在真实扰动终点预测中提供稳定的逐细胞命运信息。最稳妥、也最有机会产出正结果的身份是：

> GFG velocity 是一个由源侧 spliced/unspliced 状态估计出的、带可靠度的局部切向动态先验（source-side, state-conditioned dynamical prior）。

它不是：

- 扰动后的位移真值；
- 天级终点的速度标尺；
- 逐细胞命运标签；
- 扰动特异响应方向；
- 可直接积分数天的常数内源力。

它最可能稳定发挥的作用按可信度排序为：

1. 短时程状态变化预测（小时至 24 小时）；
2. 流形切向/路径一致性约束；
3. velocity 适用性判断与 fail-closed 可靠度门控；
4. 近邻转移方向证据，辅助而非替代表达相似性；
5. 在被独立纵向数据验证后，才进入 response-mode router。

当前 router 直接利用 velocity 做命运路由，属于尚未被数据支持的高风险解释。

## 1. 文献给出的真实边界

La Manno 等最初将 RNA velocity 定义为 spliced RNA 的时间导数，并强调它主要预测未来数小时的状态变化，而不是任意时间跨度的终点位移。[RNA velocity of single cells](https://www.nature.com/articles/s41586-018-0414-6)

scVelo 将模型扩展到非稳态动力学，但仍依赖转录、剪接、降解过程的可识别性和足够的动态覆盖。veloVI 的重要贡献不是“速度一定正确”，而是给出后验不确定度和适用性判断；这意味着 velocity 本身必须经过 reliability audit。[veloVI](https://www.nature.com/articles/s41592-023-01994-w)

Dynamo 和 CellRank 展示了 velocity 更可靠的用法：将局部箭头作为邻域转移 kernel 的方向证据，再通过聚合得到转移概率或 fate probability，而不是把每个细胞的速度直接当成未来终点。[Dynamo](https://pmc.ncbi.nlm.nih.gov/articles/PMC9332140/)，[CellRank](https://www.nature.com/articles/s41592-021-01346-6)

速度估计还存在明显的技术边界：U 稀疏、ambient RNA、doublet、capture efficiency、bursting、邻域平滑和模型选择都会影响方向。近期 benchmark 甚至显示局部 coherence 可能被计数噪声或平滑机制人为抬高。因此，重复测量一致性和 count-split null 是必要条件，不是可选附录。

## 2. GFG velocity 的实现级身份

VeloRoute 中的 GFG 并不是把观测 U/S 代入一个已知的物理 ODE 后直接得到真值速度。实际流程是：

1. GFG encoder 将每个基因的 U/S 映射到 manifold latent 和 velocity latent；
2. codebook 对两个 latent 做软量化；
3. decoder 将 manifold latent 解码回基因级 U/S；
4. 对 decoder 做 JVP：沿 velocity latent 的方向求解码表达的切向导数；
5. 取 spliced 导数，除以 `1+S`，再投影到 PCA 空间，得到 `v_GFG`；
6. 另有一个 reconstruction-error proxy `rho`。

因此更准确的数学描述是：

\[
v_{GFG}(x) = P_{PCA}\left[\frac{J_{D}(h_s(x))\,h_v(x)}{1+S}\right].
\]

它是“学习到的表达流形上的局部切向向量”，而不是唯一可识别的物理 `dS/dt`。GFG 的 manifold constraint 能减少 off-manifold 运动、提高 JVP 数值一致性，但不能自动保证方向是因果的、速度大小有绝对单位、或能外推到天级扰动终点。GFG 官方定位也强调 tangent-bundle dynamics，而非唯一潜变量分解或绝对速度标尺。[GFG overview](https://sungsoo.github.io/2026/07/08/gfg.html)

一个必须优先修复的实现问题：`GFGDynamics.prepare()` 在真实数据路径只拟合 U/S 均值、标准差和 PCA components，`velocity_scale` 保持默认 1；目前只有 synthetic 路径显式按 projected RMS 校准。因此真实 `v_GFG` 的绝对范数没有跨数据集或跨 run 的物理意义。当前只能谈方向、相对排序和可靠度，不能谈“单位/天”或把范数直接积 2–5 天。

本地代码审计还显示 native RNA ODE projection 是每个基因 2 个观测方程、3 个 rate 参数的欠定正则，低 native loss 不能被当成独立 velocity 真值。

本地实现中还必须区分三件事：

- GFG reconstruction/JVP 数值正确；
- GFG 速度在细胞重复采样下稳定；
- GFG 速度对扰动任务有独立增量。

目前第一项已基本通过，第二项只有工程级线索，第三项尚未建立。

## 3. Velocity 能扮演的角色分层

| 层级 | 角色 | 需要验证的命题 | 当前状态 |
|---|---|---|---|
| L0 | 数值估计器 | JVP 与参考实现一致、梯度可传 | 已通过 |
| L1 | 几何切向先验 | 方向更平滑、路径更少离开表达流形 | 有工程线索，缺独立生物验证 |
| L2 | 短时动态预测 | `z(t+Δ)≈z(t)+Δv_GFG` 对相邻时间有效 | 尚未完成 |
| L3 | 群体 transport 先验 | velocity 改善中间边际/均值/方差 | 旧探索不稳定，不能归因逐细胞 velocity |
| L4 | mode router 输入 | 在给定 `z,c` 后，`v_GFG` 对未来模式有独立信息 | 当前未证 |
| L5 | 个体命运/因果变量 | `v_GFG` 可解释 persister、分化或响应命运 | 当前不支持 |

真正可以称为“velocity 赋能扰动预测”的最低标准是 L4；L0-L2 只能称为动态或几何先验。

## 4. 本项目现有证据的重新解释

### 4.1 真实数据

- Norman：静态稳态系统中速度与运输方向近似正交，matched/shuffled 几乎无差异；这更像数据处于 velocity 不适用区，而不是速度物理原理被证伪。
- RENGE：三 seed 的 GFG-vs-static energy gain 为 `-0.00342`，99.44% CI `[-0.01055, 0.00243]`；GFG-vs-shuffled 为 `+0.000143`，CI `[-0.000419, 0.000758]`。条件均值 MSE 有 `3.02e-6` 的微弱改善，但不能归因于 velocity 特异信息。
- Kang：八个 donor 的 GFG-vs-static gain 为 `-1.8e-5`，CI 跨零；GFG-vs-shuffled 为 `+6.98e-5`，CI `[-0.000861, 0.001111]`。均值平移基线还显著优于 GFG，说明当前动态分支没有捕获稳定任务增量。
- MeRLin Day21：2445 个细胞、713 个克隆（有效 1777 个细胞、153 个多细胞克隆）；GFG 克隆残差 `0.102477`，shuffled-U 为 `0.102722`，z-only 为 `0.173223`。五个程序的 velocity 最大绝对相关为 `0.060–0.167`，shuffled 为 `0.067–0.170`，几乎相同。Day21 程序探针三 seed 的平均 MSE 为 z-only `0.47927`、z+GFG `0.47526`、z+shuffled `0.44804`；因此不能替代 Day0→Day21。

### 4.2 合成数据

在人为构造“相同表达、不同 U/S 且 velocity 决定分支”的系统中，GFG-router 可以恢复隐藏分支；在位置决定分支和无增量系统中不会强行制造优势。这证明架构具有可识别性，但不是现实数据的阳性证据。它只说明：若真实系统满足隐藏动态状态假设，router 有能力利用它。

### 4.3 工程阳性不等于科学阳性

GFG reference parity、JVP 梯度、联合训练 task gradient、专家参数分化都已出现。这些证明代码和梯度链工作，不证明 `v_GFG` 携带未来响应信息。当前专家 router 熵接近均匀，且真实 router 与 shuffled router 的下游差异很小，说明“能路由”不等于“路由的是生物动态信号”。

## 5. 为什么当前主线容易失败

### 时间尺度错配

RNA velocity 主要是小时级局部导数，而 RENGE/Kang/Norman 的扰动终点常在天级甚至稳态。把 `v_GFG` 作为常数 `v_int` 积分 2–5 天，会把短期动量误当成长程内源力。

### 速度和扰动位移不是同一个向量

应拆成：

\[
\Delta(z,c)=f_0(z)+g(z,c)+\epsilon,
\]

其中 `v_GFG` 最多近似 `f_0` 的局部方向；真正的扰动响应 `g(z,c)` 必须由条件场学习。当前 Norman/Kang 中 `v_GFG` 与终点运输方向不一致，正是这个分解的预期结果。

### GFG 的联合训练存在身份漂移风险

如果任务损失过强，GFG encoder 可能学会编码“有利于 router 的任意表示”，不再是 RNA 动力学估计器。因此必须同时保存 frozen-native GFG、joint-adapted GFG，并比较二者的 count-split 稳定性和生物方向；否则“GFG 联训成功”可能只是标签编码。

### `rho` 不是后验速度不确定度

当前 `rho` 主要来自 reconstruction-error proxy，不等价于 VeloVI posterior uncertainty。它可以作为质量特征，但不能直接写成“速度置信区间”。

## 6. 激进但可证伪的实验计划

### E0：GFG 估计器身份实验

在每个数据集做：

- count splitting / thinning 重复，测 `cos(v_a,v_b)`；
- U 零化、U 行内打乱、基因块打乱、S/U 配对打乱；
- 细胞深度分层、U 阳性基因数分层；
- GFG seed、checkpoint、PCA 拟合折变化；
- frozen-native 与 joint-adapted 的方向一致性；
- 速度大小校准和 sign-flip null。

硬门槛：真实重复 cosine 必须显著高于 matched null；否则 velocity 只允许作为门控负证据，不能进 router。

### E1：近端预测，而不是直接做 day-level endpoint

优先测试相邻时间：

\[
S_{t+\Delta} - S_t \quad \text{vs.} \quad v_{GFG}(S_t,U_t)
\]

候选 `Δ` 为数小时、24 小时、相邻采样间隔。RENGE 的 day2→day3→day4→day5 可做粗粒度版本；MeRLin 的 Day0→Day21 只能用于模式/克隆长期验证，不能直接检验恒定速度积分。

### E2：MeRLin 决定性纵向实验

以 Day0 source、Day21 target、克隆条码为外部锚点，做 clone-held-out：

1. `S-only static`；
2. `S + GFG velocity`，但 velocity 只影响 routing prior；
3. `S + shuffled-U`。

评价未来 persister program 分布、克隆比例、mode proportion、energy distance、variance ratio。必须比较：

\[
I(Y_{mode};v_{GFG}\mid z,c) > 0
\]

而不是只比较 `cos(v, target-source)`。

### E3：四种作用位置分别消融

- 直接 endpoint additive：预期最不稳定；
- ODE 初始方向/短时 prior：主推候选；
- along-path/tangent regularizer：第二候选；
- reliability gate：最稳妥保险丝。

每一臂必须与 static 参数量、训练预算、预训练状态匹配；同时保留 real-U、shuffled-U、zero-U、sign-flip 四个 null。

### E4：router 重新定义

不要让 router 直接把每个 velocity atom 当作 fate label。建议使用 CellRank 风格的方向兼容度：

\[
q_k \propto q_{expr}(k\mid z,c)\exp\{\eta\,w(v)\cos(v_{GFG},d_k)\},
\]

其中 `d_k` 是训练折定义的未来模式方向，`w(v)` 由 count-split coherence、U 覆盖度、rho 和 OOD 距离决定。`w(v)=0` 时严格退化为 static。只有 MeRLin 纵向实验确认增量后，才允许 velocity 主导 `q_k`。

### E5：时间衰减

不再把 `v_GFG` 作为 2–5 天恒定内源场。使用：

\[
v_{int}(t)=\alpha(z)\exp(-t/\tau_v)\,\mathrm{unit}(v_{GFG})+r_\theta(z,t,c).
\]

短时由 velocity 提供方向，长时由 perturbation-conditioned field 负责。`tau_v` 必须由相邻时间数据校准，而不是手工解释。

## 7. 推荐的最终架构定位

### 默认部署版本

- `z` 和 `c` 决定主要扰动场；
- GFG velocity 只进入一个 reliability-weighted prior 分支；
- prior 影响初始 routing probability、沿途约束和 gate；
- gate 失败时严格回退 static；
- router 不把 GFG 模式索引解释成真实命运类别；
- 速度不作为 flow-matching target，不直接替换 endpoint displacement target。

### 只有在纵向阳性后才启用

- velocity 直接进入 router logits；
- K>2 多专家命运承诺；
- 速度方向参与长期 ODE 积分；
- “hidden priming / bifurcation fate”标题主张。

## 8. 最终身份判定

| 说法 | 判定 |
|---|---|
| GFG velocity 是数值上可计算的 decoder-JVP | 支持 |
| GFG velocity 是表达流形局部切向先验 | 支持，且最合理 |
| GFG velocity 是短时程状态变化预测器 | 待验证，最值得做 |
| GFG velocity 是扰动响应方向 | 当前不支持 |
| GFG velocity 是逐细胞命运标签 | 当前不支持 |
| GFG velocity 是天级终点位移 | 不应这样使用 |
| GFG velocity 能稳定赋能 router | 尚未建立 |
| GFG velocity 能作为可靠度/适用性门控信号 | 中等可信，需 count-split 校准 |

## 9. 项目决策

项目不应再以“完整 router/专家网络训练完成”为成功标准。新的成功标准是：

> GFG velocity 在独立的、纵向的、克隆或时间留出任务中，提供超出 `z,c` 和 shuffled-U 的结构性增量；否则它只保留为几何先验和可靠度门控。

当前最高优先级仍是 MeRLin Day0→Day21，而不是继续扩大专家数或调 EM。若该实验也为阴性，最诚实且仍有产出的方向是“RNA velocity 在未配对扰动终点预测中的适用性边界与 fail-closed 评估”，而不是继续维护一个没有任务增量证据的 fate router。
