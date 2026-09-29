# VeloRoute 实验总账与声明边界

记录日期：2026-09-14。性质：已完成实验的追溯记录、讨论决策和下一轮建议；不是回填的正式预注册。

本文件依据已落盘配置、summary、CSV、checkpoint/预测选择清单及本日文献核查整理。本次只补文档和审计索引，不启动新模型训练、不打开最终确认集、不覆盖历史实验目录。机器可读审计见 [总账审计包](../outputs/veloroute_experiment_ledger_20260914/RESULTS.md)。

## 0. 当前结论：先读这一页

| 问题 | 当前证据状态 | 不应混淆的边界 |
|---|---|---|
| 全量组件是否实现？ | 已实现，约 20.02M 参数 GFG 全宽模型通过 GPU 短验收 | 不等于全量长训练或全部组件有收益 |
| 是否真用 GFG？是否联训？ | 实际双流/VQ/decoder JVP；router 任务梯度回传到 GFG | 不是旧 MLP 改名；初始化不是认证的官方 foundation release |
| velocity/router 是否有积极信号？ | 有：指定未配对合成系统的组分预测有明显增益 | 这是机制可行性阳性，不只是工程阳性；不是实际生物学增量 |
| 真实 RENGE 上有增量吗？ | 两轮、合计 21 次固定预算训练后仍未建立稳定增量 | 不是宣称所有数据上无效；也不能把很小的正点估计写成成功 |
| 真实多峰优势成立吗？ | 14 个训练 TF 中 0 个通过本轮稳定性筛查 | 不是正式 D4/FDR；不能断言数据全部单峰 |
| 正式 G1 / 最终确认？ | G1 NOT_RUN；最终 5 TF 封存 | 开发 TF 已反复查看，不能当独立确认 |
| GFG 长时全量？ | 暂缓 A300/B2000/C12000 | 旧残差后端的长训练已经跑过；两者不是同一实验 |
| 数据扩展？ | day3 四条 raw 已完整获取；S/U 尚未定量 | Kang/McFarland 等只是文献与元数据候选，未跑增量实验 |

用户明确保留 velocity/GFG 主线，同时接受适当降低机制声明。当前决定是降低“要证明什么”的强度，不改变已经观察到的数字，不事后放宽阈值或把未成立的真实收益改称成立。

## 1. 执行顺序与修订历史

1. 2026-09-12：RENGE 文库、化学布局、day4/day5 S/U、guide 初步对齐、TF 切分和训练折潜空间落地。
2. 2026-09-13 至 14：在用户允许 G1 前探索的授权下，实现全量组件并完成旧 steady-state residual 后端探针、最小 router、1-seed 长时 full 和一次固定专家跟进。结果整体不支持 velocity/router 增量。
3. 2026-09-14：用户要求实际 GFG，并要求与后续模块联合训练；原 VeloVI 下一步被取代。核对原实现、严格加载本地 MouseBrain checkpoint、完成数值与梯度验收。
4. 开展训练侧多峰筛查和未配对合成实验。原后编码置换未形成完整阳性；分析其保留群体边际信息的原因后，另做前编码 U 置换实验。两个版本均保留。
5. 实际 GFG 四臂三 seed pilot 完成；全部 12 份预测冻结后统一评价，未建立真实增量。
6. 依据训练侧路由概率敏感性低、标签近均匀的检查，记录并执行唯一一轮“固定专家 + 精确混合分布损失”的直接 router 跟进。3 臂 × 3 seed 完成后统一评价，仍未建立增量。
7. 决定暂缓 GFG 长时全量，不继续在同一 ER-short 开发集上换损失/加 seeds 追阳性。补 day3 数据，讨论 Kang、McFarland、sci-fate 等外部场景。
8. 用户允许降低核心声明。本文件明确合成机制阳性、真实任务增量、GFG 特异性与生物学机制四个不同层级。

以上是实际研究过程；后续修复属于探索性设计，不追认其为原始预注册。原计划“诊断不过不建模”曾由用户明确放宽为允许开发侧实现与实验，不能据此跳过最终确认与泄漏约束。

## 2. 数据、任务和拟合范围

### 2.1 当前实际 RENGE 任务

任务为同 KO 条件下 day4 源侧到 day5 终态的 ER-short 预测，不是 day2 对照侧 de novo，不是 2→5 时间外推。

| 层级 | 已落地内容 |
|---|---|
| day4 / day5 全部发布 barcode 对齐细胞 | 7,582 / 6,910；每个时间点 60,668 基因 |
| U UMI 中位数 | day4 675；day5 836；不是 velocity 正确性的证据 |
| TF 切分 | 14 train / 4 development / 5 confirmation |
| 开发 TF | LIN28A、NANOG、POU5F1、ZIC3 |
| 训练源 / 训练目标 | 2,733 / 2,401；共 5,134 训练细胞 |
| 开发源 / 开发目标 | 1,133 / 944 |
| 实际表达工作空间 | 训练折选 2,000 基因，PCA-50；不是初始计划中的固定 5,045 基因 |
| 条件表示 | 23 × 2,560 冻结 ESM2-3B；模型内部条件维度 256 |
| guide 状态 | 初步 UMI dominance 归属；编辑逃逸、双细胞、guide 一致性尚未正式验收 |

源侧 S 按源细胞完整 S 文库归一化，log1p 后投影；U 使用同一 S 文库分母。GFG 读取所选基因的未取 log 的归一化 U/S，标准化参数只由训练源拟合。状态与位移解码使用冻结 PCA。

重要标识：

- 冻结变换标识：`2106a8bbc8c9082ec95ec5c954ebfa57c5215c1a19e4020e82aa485e09648204`。
- 拟合细胞 ID 标识：`52c02c8c8af6339a1303e31eb0d7e3cf396a2837a02a274f9c092962eeed1c5b`。
- 条件包标识：`951d4ecafdec552de1e1532c9dd79f686fdc065af9c8822c2a54a2142996155e`。

上述为对应产物记录中的标识；文件字节 SHA256 由总账审计包另行记录，不混淆对象标识与文件哈希。

依据：[冻结数据包](../outputs/veloroute_real_pipeline_20260912_v2/fold/RESULTS.md)、[GFG 输入](../outputs/veloroute_gfg_inputs_20260914/RESULTS.md)。确认数据包可能已经通过冻结变换生成并隔离存放，但训练、调参和本次汇编均不打开其响应数组。

### 2.2 原始文库处理的纠错

- RENGE 每天两条 mRNA/GEX 和两条 gRNA 文库，不是“小文件也都是同一 GEX 的低深度”。gRNA 不能送入表达 USA。
- day4/day5 的 SRA 提取实测 `_1` 为样本索引、`_2` 为 barcode/UMI、`_3` 为文库序列；不能机械使用 `_1/_2`。
- 首位偶有 N 不代表固定应剪掉 1 bp。day4 完整指定 gRNA run 的 barcode 比对支持从首碱基开始，而非右移 1 位。
- 以上布局必须按新数据集重新审计，不能把 RENGE 参数直接套到早期 10x 的 Kang。

## 3. 当前 GFG 实现及其限制

### 3.1 实际来源

- 原实现目录：`/data/yuchang/GFG/model/`。
- 初始化：`/data/yuchang/GFG/results/mousebrain_graphbatch_directed_v1_seed0/final.pth`。
- checkpoint SHA256：`a9d9fa39063312230e4ba64309e9264ba3c2cb7ac549f04544b366df57c1340b`。
- RNA 双流结构：隐藏层 256/512/512/256、gene dimension 8、双 soft-VQ 各 32 codes、共享基因 decoder 与 JVP。
- 早先 [候选前置配置](../configs/veloroute_gfg_prerequisites_20260914.yaml) 指向另一份约 14.64M corrected foundation pilot；它不是本轮实际加载的权重。本轮没有按照下游分数比较或挑选两者。

### 3.2 velocity 到底在哪里起作用

主要推理通路：源基因 U/S → GFG decoder JVP → `vS/(1+S)` → 冻结 PCA 旋转 → 50→64 表示 → router → 固定模式专家 → 终态分布。

- router 的任务损失经 JVP 回传到 GFG；扰动条件 c 不进入 GFG 前向接口。条件相关的训练梯度不等于把 c 当作前向输入。
- E-step 参考场、老师标签及门控特征 detach；FM 主目标仍是训练 OT 配对位移除以时间差，不是直接拟合 RNA velocity。
- 完整模型的参考场用于耦合方向兼容与沿途方向约束；可选内源场使用初始速度方向。不能假定源侧 RNA 方向必然等于施加扰动后的运输方向。
- 真实 pilot 关闭内源、门控、噪声和自适应 K；第二轮进一步冻结专家候选，只检查 GFG/router 对混合权重的贡献。
- 推理仍需源侧 U/S，因此当前方案不是“velocity 仅在训练时存在”。

代码依据：[GFG](../src/veloroute/gfg.py)、[上下文与 router](../src/veloroute/full_model.py)、[训练代价与损失](../src/veloroute/full_training.py)、[直接 router](../src/veloroute/gfg_direct_router.py)。

### 3.3 明确记录的适配与未解决问题

1. 禁用 dropout，使 decoder 与 JVP 对应同一确定函数；RNA ODE 求解用 float64，并按 RNA 量纲缩放残差。
2. 原生损失包括 U/S 重构、VQ 和 RNA ODE 约束；本轮不做原预训练中的表达邻居 moments 平滑，存在预处理及跨组织域迁移。
3. 逐细胞/逐基因的局部 RNA 约束为 2 方程、3 速率参数，欠定。低原生损失不证明真实速度方向或绝对速率正确。
4. rho 学的是源 RNA 重构误差代理，不是实际 velocity 误差的校准后验。corruption 门控也不保证预测非劣。
5. 状态编码器为 PCA 坐标上的残差去噪器，A 后冻结；参考场只依赖训练源侧，不是已验证的扰动后真实物理场。
6. 联训后的动态表示即使有用，也可能只是有用的 S/U 表示。要解释为 velocity 特异收益，需要等容量 S+U 编码器与独立时间/标记证据。

## 4. 指标、统计和无泄漏契约

当前分布指标为 PCA-50 中的 energy V-statistic：`2 E||X-Y|| - E||X-X'|| - E||Y-Y'||`，不取平方根。它不是整个 CellFlow 官方 15 指标套件。

- pilot：按同一预测机制抽样的终态分布计算 energy；方差误差为 `abs(log(variance_ratio))`，比值由总潜空间方差计算。
- 直接 router 跟进：在每颗源细胞的固定专家候选及 q 上计算精确混合 energy，消除该有限候选混合的模式抽样噪声；不消除有限细胞或训练不确定性。同时保留 sampled energy 作诊断。
- 条件中心化基因均值 MSE：分别去掉预测与真实的跨条件共享均值，再比较条件特异部分；它是均值层指标，不是模式保真指标。
- 增益统一定义为 `对照误差 - joint GFG 误差`，正值才有利。
- 先按条件平均 3 seeds，再以条件为单位配对 bootstrap 10,000 次，bootstrap seed 20260914；不能把细胞数或 seeds 当额外生物学重复。
- pilot 9 项比较校正置信水平为 `1 - 0.05/9 = 0.994444...`；直接 router 6 项为 `1 - 0.05/6 = 0.991666...`。
- 每个实验族的全部预测先冻结并核验哈希，再统一读取开发目标。不同实验族预算、预测器与部分指标口径不同，不组成跨族公平排行榜。
- 只有 4 个反复查看的开发 TF，置信区间和推广性有限；零附近且跨零的结果应写“未建立增量”，不是已证明精确等效。
- 最终确认 TF 不拟合 HVG/PCA/GFG/邻居/模式、不参与早停与调参、不读取响应效果；OT/EM 伪标签不是逐细胞真实命运证据。

依据：[指标实现](../src/veloroute/metrics.py)、[pilot 评价](../scripts/phase0/evaluate_gfg_pilot.py)、[精确混合评价](../scripts/phase0/evaluate_gfg_direct_router.py)。

## 5. 历史实验：旧残差 velocity，不是 GFG

已完成 A/B 七臂三 seeds、最小 router 五训练臂三 seeds、一次旧 full 长训练及一次固定候选专家跟进。原估计器是训练折 steady-state residual，不是 GFG 或 VeloVI。

| 实验/比较 | 结果摘要 | 判断 |
|---|---|---|
| velocity 探针 real 对 S-only | energy gain -0.017641；校正区间跨零 | 无稳定正增量 |
| velocity 探针 real 对 shuffled | gain -0.031586；区间跨零 | 无独立信息收益证据 |
| S+U 对 S-only | gain -0.022770；区间为负 | 该协议下退化 |
| 最小 velocity-router 对 static-router | gain 约 -0.0000164 | 极小负差异，不是实际改善 |
| 最小模型对 identity / 均值位移 | 好于 identity，差于训练全局 mean-shift | 学到变化不等于 velocity 有用 |
| 固定专家后仅路由 | 未胜过 static、shuffled 或 uniform | 专家分开不保证路由有用 |

旧 full：A1000/B2000/C12000，1 seed，部署参数 18,541,718（含额外冻结静态回退），不是严格参数匹配的静态因果比较。

| 同一 checkpoint 的干预 | 平均 energy |
|---|---:|
| 完整模型 | 0.52162575 |
| 冻结静态回退 | 0.40255031 |
| 全动态分支 | 1.14350030 |

学习门控缓解了全动态退化，但未达到静态非劣；`g=0` 精确回退是工程性质，不是学习门控无损的实证保证。

历史完整表、误差区间与产物：[旧估计器研究进度](../outputs/veloroute_research_progress_20260914/RESULTS.md)。该旧文档中的“下一步 VeloVI / GFG 未接入”只代表当时状态，已由后续用户指定的 GFG 联训取代；不修改旧产物。

## 6. GFG 工程验收

| 检查 | 实测 | 能支持什么 |
|---|---|---|
| 原实现重构数值对齐 | 最大绝对差 4.172325e-7 | 移植实现一致 |
| 原实现 JVP 对齐 | 最大绝对差 3.576279e-7 | 实际使用原双流/JVP 路径 |
| 全宽全组件短验收 | 20,021,704 参数；8 专家；13 次老师更新；4 corruption batches | 组件可运行，不授予真实收益 |
| 短验收 task gradient | 最大范数 0.00589546 | router 任务梯度能到 GFG |
| 随机生成 / checkpoint | 输出有限、重载一致 | 数值工程检查通过 |
| 实际中断恢复 | C200→C400；207/207 张量逐位一致，trace 一致 | 恢复链可靠 |
| 回归测试 | 123 passed，v5 JUnit | 截至本轮已有测试通过，不证明科学假设 |

原始产物：[数值对齐](../outputs/veloroute_gfg_reference_parity_20260914/summary.json)、[全组件短验收](../outputs/veloroute_gfg_full_smoke_20260914/summary.json)、[恢复验收](../outputs/veloroute_gfg_actual_resume_verified_20260914/summary.json)、[123 项测试](../outputs/veloroute_gfg_all_tests_20260914_v5/junit.xml)。

## 7. 合成机制实验：重要阳性，但限定范围

### 7.1 设置

实际 GFG/JVP 的基因级 toy；四种隐藏状态/群体组成系统，四臂 static / frozen GFG / joint GFG / shuffled，seeds 0/1/2。每组 96 细胞，训练组分比例 0.2/0.4/0.6/0.8，测试 0.1/0.3/0.7/0.9，每臂 300 步，router lr 1e-3、GFG lr 1e-4、原生损失权重 0.1。

只以未配对训练终态的组分分布监督；原型来自训练目标 KMeans。逐细胞真实命运只用于评价，不用于训练。这是隐藏状态合成系统，不是经过生物学标定的 RNA 反应仿真，也不是整个 ODE/EM/门控系统的验收。

以实际运行目录的 [合成配置](../outputs/veloroute_gfg_synthetic_inputU_recovered_20260914/config.yaml) 为准；不要把早期候选 yaml 中不同系统、比例或预算当作实际实验配置。

### 7.2 原后编码置换：未形成完整优势

原实验先经 GFG 编码 S/U，再重排输出 r/velocity。interaction 系统中 joint MAE 0.063336，shuffled 0.062522，未达到 real 优于 shuffled 的要求；原总状态为 `GFG_SYNTHETIC_ADVANTAGE_NOT_ESTABLISHED`。

这是保留在案的未通过结果：[原合成输出](../outputs/veloroute_gfg_synthetic_20260914/summary.json)。

后续解释：对群体边际任务，重排已经编码好的输出保留其多重集合，可能仍保留总体组分信息；不能普遍要求它导致群体预测退化。表达完全相同时，重排 U 本身也保留群体联合分布。

### 7.3 前编码 U 置换：指定机制检查通过

另开前编码干预：固定 S，在进入 GFG 前打乱 U，破坏 interaction 系统中的 S/U 对应关系。种子、步数和阈值保持一致；不覆盖原后编码结果。

| 系统；三 seed 组分比例 MAE，越低越好 | static | frozen GFG | joint GFG | 前编码 U 置换 |
|---|---:|---:|---:|---:|
| same-S，隐藏组分变化 | 0.300000 | 0.063562 | 0.061584 | 0.058736 |
| S×U 对应关系决定组分 | 0.300000 | 0.071797 | 0.063336 | 0.292213 |
| 静态位置决定组分 | 0.103522 | 0.106853 | 0.102263 | 0.101751 |
| 单峰、无增量 | 0 | 0 | 0 | 0 |

固定检查：interaction 相对 static 与 shuffled 的增益要求 0.1，静态负对照容忍退化 0.05；运行 summary 中四项检查均通过。same-S 的置换不变性是预期，不要求 real 胜过它。

执行中断后保留原目录，恢复时复用 40 个完整 checkpoint，仅补训剩余 8 个单峰臂；最终 48 个模型在 recovered 目录，`training_trace.csv` 只覆盖新补训部分，不能把它当作 48 次完整训练 trace。

积极解读：这个结果比单纯工程阳性更强，说明当前 GFG/router 可以在指定的未配对系统中利用表达之外的 S/U 对应信息。它支持继续寻找真实优势场景。

限制：未加入同容量原始 S+U 编码器；不能证明 GFG velocity 特异性。未证明 joint 稳定胜过 frozen。单峰 K=1 预先指定，不证明自动关闭门控或恢复真实模式数；合成模式成功不等于真实逐细胞命运被识别。

依据：[前编码 U 置换结果](../outputs/veloroute_gfg_synthetic_inputU_recovered_20260914/summary.json)、[全部指标](../outputs/veloroute_gfg_synthetic_inputU_recovered_20260914/metrics.csv)。

## 8. 训练侧多峰筛查

仅使用 14 个训练 TF；local PC1、local PC2、效应方向三轴；10 次随机分裂，每半至少 20 细胞。要求拟合密度有两个峰、BIC 增益至少 10、Ashman 分离至少 2、两组各至少 10%、留出细胞 log-likelihood 改善，并达到 80% 分裂支持。

结果：0/14 TF 达标；标准一维 Gaussian sanity check 0/20 阳性。后者不是与每个条件协方差匹配的零分布，不提供正式 omnibus/FDR 校准。

结论：目前没有可冻结的强多峰优势子集，但功效有限；不据此宣称真实系统全部单峰。guide、编辑逃逸和细胞周期混杂未排除，UMAP 分簇不能替代该检查。

依据：[筛查结果](../outputs/veloroute_gfg_multimodal_screen_20260914/summary.json)。

## 9. 真实 GFG pilot：四臂三 seeds

### 9.1 固定预算和公平性

每臂 A100/B300/C400，batch 8；骨干宽 256、3 residual blocks、K=2，关闭自适应 K/内源/门控/噪声。router/field lr 1e-4，GFG lr 1e-5，原生损失权重 0.1；动态代价 0.1、沿途约束 0.01；E-step 间隔 50，参考更新 200，checkpoint 200。

- static：动态表示置零；GFG 原生计算仍执行但不为 router 提供动态输入。
- frozen GFG：A 阶段适配后冻结，C 阶段不再获得原生更新，因此与 joint 存在预训练更新预算差异。
- joint GFG：任务梯度和原生损失共同更新 GFG。
- joint shuffled：在 condition × S-depth × 局部 S 块中重排 U，S 不变，干预发生在 GFG 前。

此 pilot 的 static 与 joint 同时涉及动态耦合/沿途约束是否启用，不能把全部差异单独归因于 router。未宣称与官方 CellFlow/scDFM 完整公平比较已完成。

### 9.2 全体开发条件平均结果

| 臂 | sampled energy | log 方差误差 | 条件中心化基因均值 MSE |
|---|---:|---:|---:|
| static | 0.338830119 | 0.097763741 | 0.00295654377 |
| frozen GFG | 0.342869443 | 0.098542122 | 0.00295388123 |
| joint GFG | 0.342245164 | 0.098643851 | 0.00295352047 |
| joint U 置换 | 0.342388469 | 0.098693591 | 0.00295382523 |

| joint 相对的对照 | 指标 | gain | 校正区间 |
|---|---|---:|---|
| static | energy | -0.003415045 | [-0.01055062, 0.002427108] |
| U 置换 | energy | 0.000143306 | [-0.000418987, 0.000758292] |
| frozen | energy | 0.000624279 | [-0.000507471, 0.001662023] |
| static | log 方差误差 | -0.000880109 | [-0.002973928, 0.000998665] |
| U 置换 | log 方差误差 | 0.000049740 | [-0.000048511, 0.000233793] |
| frozen | log 方差误差 | -0.000101729 | [-0.000359810, 0.000325609] |
| static | 中心化均值 MSE | 3.023296e-6 | [3.201809e-7, 6.781969e-6] |
| U 置换 | 中心化均值 MSE | 3.047559e-7 | [-3.900689e-7, 9.794409e-7] |
| frozen | 中心化均值 MSE | 3.607614e-7 | [-1.047039e-6, 2.043943e-6] |

判断：分布与方差收益未成立；对 static 的中心化均值误差有微小改善，但未胜过 U 置换，不能用这一项证明 velocity 独立价值。主判据 `all_primary_intervals_positive=false`。

数字以 [机器可读比较表](../outputs/veloroute_gfg_pilot_evaluation_20260914/comparisons.csv) 和 [summary](../outputs/veloroute_gfg_pilot_evaluation_20260914/summary.json) 为准；表内小数为显示舍入。

## 10. 训练源侧审计及唯一一轮 router 修复

### 10.1 为什么安排跟进

在 112 个训练源细胞（每 TF 8 个）的描述审计中：

| 指标 | 实测 |
|---|---:|
| 初始化 / 联训后速度范数均值 | 30.389740 / 2.292530 |
| 初始化与联训方向余弦中位数 | 0.937512 |
| 速度协方差有效秩 | 30.422091 |
| 条件内 U 置换引起的速度相对 RMS 改变 | 0.315524 |
| 同一干预引起的 router 概率 MAE | 0.000237175 |
| 平均模式熵 | 0.692965，接近 ln(2) |
| 概率跨细胞标准差 | 0.007577 |
| 专家场相对离散度 | 0.094211 |

这提示源速度并非全零，但路由教学/使用可能很弱，不是故障原因的因果证明。初始化方向不是独立真值，不能把方向保留或原生损失下降视为速度生物学正确。

依据：[训练源审计](../outputs/veloroute_gfg_source_audit_20260914_seed0/summary.json)。

### 10.2 跟进设计

以每个 seed 的 static pilot checkpoint 提供同一组专家候选；所有三臂共享相同冻结状态编码、条件编码、骨干、专家和 PCA，不按开发分数选择初始化。仅 router/GFG 更新，以未配对训练终态的精确混合 energy 直接监督，不依赖 EM 模式伪标签。

3 臂 × 3 seed × 600 步，source batch 8、target batch 64；router lr 1e-3、GFG lr 1e-5、原生损失权重 0.1。置换仍在条件/深度/局部 S 块中、发生在编码前。所有 9 份预测冻结后统一评价。

这是看过前一轮开发结果与训练侧问题后安排的一次探索性跟进，不是新的独立确认。其 [冻结配置](../configs/veloroute_gfg_direct_router_20260914.yaml) 明确约束：没有新外部锚，不继续同类调参。

### 10.3 结果

| 臂 | exact mixture energy | log 方差误差 | 条件中心化基因均值 MSE |
|---|---:|---:|---:|
| static | 0.337254832 | 0.097561923 | 0.002960172749 |
| joint GFG | 0.337221542 | 0.097541937 | 0.002960132265 |
| joint U 置换 | 0.337223766 | 0.097520281 | 0.002960045706 |

| joint 相对的对照 | 指标 | gain | 校正区间 |
|---|---|---:|---|
| static | exact energy | 3.328930e-5 | [-6.895246e-5, 1.188763e-4] |
| U 置换 | exact energy | 2.223426e-6 | [-7.615247e-5, 6.000603e-5] |
| static | log 方差误差 | 1.998597e-5 | [-6.447875e-5, 8.375535e-5] |
| U 置换 | log 方差误差 | -2.165578e-5 | [-6.820240e-5, 3.448845e-5] |
| static | 中心化均值 MSE | 4.048359e-8 | [-7.279286e-8, 1.804121e-7] |
| U 置换 | 中心化均值 MSE | -8.655908e-8 | [-2.869624e-7, 5.928445e-8] |

所有六项区间跨零；点估计差异也非常小。候选专家逐数组一致性已核验，seed0 task-gradient max 为 0.039805。梯度存在仍未转化为可信增量。

判断：`all_primary_intervals_positive=false`。这说明本轮直接路由修复不足以建立优势，不说明所有可能的动态路由都无效。不得把本轮 exact energy 与上一轮 sampled energy 的行间下降当作公平的改进量。

依据：[完整比较表](../outputs/veloroute_gfg_direct_router_evaluation_20260914/comparisons.csv)、[结果 summary](../outputs/veloroute_gfg_direct_router_evaluation_20260914/summary.json)。

## 11. 中断、恢复与结果选择规则

- 原始中断目录、失败状态与旧 checkpoint 全部保留，不覆盖、不删除。
- pilot 的 static seed1/2 从已保存状态补齐；直接 router 的 joint 三 seeds 亦使用独立恢复目录。
- 每个实验的权威训练/预测路径由其 `selection.json` 指定；规则是完成固定预算，不是选择最好的开发分数。
- 总账审计逐个读取 12 + 9 个 selection，核验完成状态、model.pt 与 predictions.npz 的 SHA256，以及预测 manifest 的源侧输入契约。
- 复用 checkpoint 的合成恢复不虚增独立重复；完整 trace 与补训 trace 明确区分。

权威根目录：[pilot](../outputs/veloroute_gfg_pilot_20260914/)、[直接 router](../outputs/veloroute_gfg_direct_router_20260914/)。完整 21 行路径/哈希表在总账审计包的 `run_index.csv`。

## 12. “velocity 有用”应分成四层声明

| 层级 | 声明 | 当前状态 |
|---|---|---|
| 工程 | GFG/JVP 能被 router 使用并接收任务梯度 | 已验证 |
| 合成机制可行性 | 在指定未配对合成系统中，源侧 S/U 对应信息经 GFG/router 改善组分预测 | 已验证；是实质性积极信号 |
| 真实任务效用 | GFG 动态表示改善真实异质扰动分布预测，相对合理静态和置换对照有收益 | 当前 RENGE 未建立；其他候选未检验 |
| 生物学/特异机制 | 真实 velocity 解释隐藏命运，优于一般 S+U 特征，并正确恢复逐细胞分支 | 未验证；未配对分布本身不足以识别逐细胞命运 |

### 12.1 现在可以写的结论

“在构造的未配对系统中，当 S/U 对应关系包含静态表达之外的组分信息时，GFG 与 router 能利用这一信息改善终态组分预测；在当前 RENGE ER-short 开发任务上，尚未建立稳定的独立增量。”

### 12.2 接受降低声明后的研究目标

候选目标改为：“利用 GFG 提取的源侧 RNA 动态表示，改善未配对扰动后的异质响应分布预测。”

这不要求证明真正的逐细胞命运，也不强求 bifurcation 标题。但该句目前是待验证目标，不是已经完成的论文结果。若未来仅普通 S+U 表示有效，应降为第二模态收益，不宣称 velocity 特异性。

### 12.3 不允许的表述

- “已经证明真实扰动预测中 velocity 有用。”——当前只有限定合成机制阳性。
- “GFG 生物学速度已校准。”——没有独立真值或充分时间锚。
- “多峰数据上必胜静态方法。”——多峰不等于模式归属依赖动态信息。
- “联合训练必然优于冻结 GFG。”——尚未建立。
- “学习门控保证不劣于静态。”——真实非劣未成立。
- “靠改摘要，阴性就能变成阳性。”——降低机制解释不能代替实测收益。

## 13. 其他数据候选：文献与元数据，不是本项目新实验

本节记录 2026-09-14 的公开原论文/GEO 核查及本地元数据统计。候选优先级是研究判断，不是成功率或实际实验效果；未下载这些候选的大规模原始数据，未训练新模型。

### 13.1 Kang / GSE96583：先做低成本结构审查

CellOT 报告 Kang 的部分受扰动 marker 表达边际呈双峰，并做留出供体预测；静态 CellOT 已能较好恢复这些分布，因此不能从双峰直接推出 velocity 独立优势。[CellOT 原论文](https://www.nature.com/articles/s41592-023-01969-x)

GEO 的 batch2 为培养 6 小时对照 `GSM2560248` 和 IFN-beta 刺激 `GSM2560249`，不是逐细胞前后配对，也不是密集时间序列；GEO 标明 raw 在 SRA，S/U 恢复质量尚未实测。[GEO 系列](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583)、[对照](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2560248)、[刺激](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM2560249)

本地仅元数据统计：29,065 行；标注 singlet 24,679、doublet 3,169、ambiguous 1,217；singlet 覆盖 8 供体。CD14+ Monocytes 5,385、CD4 T cells 10,389、FCGR3A+ Monocytes 1,599；这些不是最终 QC 后样本量，也没有检查这些细胞的响应效果。

拟议检查：先隔离开发/留出供体；在开发 donor × cell type 内确认多基因响应模式、跨供体复现，排除零计数/深度/类型混合；随后再投入小规模 raw/S/U 审计。只允许源侧 U/S 进入预测。Kang 的单扰动设置不能独自支持未见扰动泛化。

### 13.2 McFarland / MIX-seq：特定药物的模式候选

作者报告 bortezomib 使 24 个细胞系中的 10 个出现双峰响应，一部分停在 G0/G1，另一部分主要为 S 期。另一个 trametinib 实验有 3–48 小时五个时间点；二者是不同实验，不能拼成同一套“双峰 + 密集时间”数据。[MIX-seq 原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7453022/)

定位：很贴近细胞系内部的异质模式问题，值得补查可用细胞数及 raw/barcode。仍需以静态细胞周期特征为强对照；死亡/选择导致的群体比例变化也不能无条件解释为命运路由。作者报告的模式须经本项目训练侧独立判据检查，不直接等同于 D4 通过。

### 13.3 sci-fate：动力学的外部核验候选

A549 经 DEX 处理 0/2/4/6/8/10 小时，带新生 RNA 代谢标记；适合检验时间信息。新生 RNA 不是 unspliced。若能恢复标准 S/U，可让 GFG 用 S/U，而将新生 RNA 留作独立核验；不能把标记层直接改名成 U。[sci-fate 原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7416490/)

### 13.4 PerturbSci-Kinetics：第二模态审计，不默认动态多峰主战场

228 基因的 CRISPRi 屏，whole/nascent 配套读出；论文明确诱导七天以达到转录稳态，末端标记两小时。因此它适合第二模态信息审计，但不能因为名字有 Kinetics 就假定是转变中的多峰系统。新生 RNA 不能伪装成标准 S/U。[原论文](https://www.nature.com/articles/s41587-023-01948-9)、[GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218566)

### 13.5 其他角色

- RENGE day3→day5：继续复用已投入的数据链，增加时间约束；不保证能扭转阴性。
- IGVF hPSC 分化屏：保留为扩展候选；原计划指定数据包的精确 accession、源侧设计和时间结构仍未完成核验，不凭概述启动百万级重处理。
- Norman/Replogle：仍定位为静态负对照与失效边界；本项目当前不以它们作为追逐阳性的主要投入方向，不把团队历史报告数字冒充本工作区新结果。

## 14. day3 最新状态与后续必要实验

四条 day3 raw 已完成下载、SRA 校验、提取、保留原件压缩和 gzip 检验：

| run | 文库 | 当前状态 |
|---|---|---|
| SRR21518309 | gRNA | raw 完成；布局检查前 100 万对 |
| SRR21518310 | gRNA | raw 完成；布局检查前 100 万对 |
| SRR21518311 | mRNA | raw 完成；布局检查前 100 万对 |
| SRR21518312 | mRNA | raw 完成；布局检查前 100 万对 |

完成时间记录为 2026-09-14 08:44:11 UTC（北京时间 16:44:11），后台 exit code 0、GPU 使用为零。raw 根目录 `/data/yuchang/veloroute_timeanchors_20260914/`；没有移动/删除旧文件。**成品 S/U、guide/QC、训练包和 day3 新模型均未完成。**

依据：[raw 完成记录](../outputs/veloroute_raw_day3_20260914/RESULTS.md)、[逐 run CSV](../outputs/veloroute_raw_day3_20260914/metrics.csv)、[后台状态](../outputs/veloroute_jobs_20260914/day3_raw/status.json)。

拟议下一阶段：

1. 完成 day3 布局验收、两条表达文库联合 USA、guide 对齐/QC。只用训练角色拟合后续变换。
2. Kang 先用已有矩阵做开发侧模式筛查，同时只读核查 McFarland 指定子集的原始可恢复性；不要先训练再挑最好的细胞类型/药物。
3. 在新的比较开始前，记录 donor/TF 切分、源侧信息、模式定义、最小有用效应、预算、seeds、置换方式和终止条件；本文件不是这些尚未运行实验的完整预注册。
4. 对照至少区分静态单场、静态 router、GFG router、合适的前编码 U 置换及等容量 S+U 编码器。置换是否破坏目标信息需按任务证明，不强求保留边际的置换必然变差。
5. 时间锚任务须明确哪些时间边际被训练、哪些真正留出。若 day4 已用于某条件训练，就不能把同一条件 day4 再称独立时间验证；可按条件/时间联合留出。新增时间点也不是独立供体。
6. 只有出现可信的真实分布/结构收益才扩大长时全量、严格基线和多 seed。只改善均值或无法超越合理对照，应按证据降低结论，不无判据追加搜索。

## 15. 资源、环境与复现入口

- 资源硬限制：本项目仅使用物理 GPU 0–3 或已核验的相应完整 UUID；GPU 4–7 不使用。不停止其他项目已有进程，也不承诺整机 4–7 实际空闲。
- CPU `.venv`：PyTorch 2.6.0+cpu、NumPy 1.26.4；GPU `.venv-gpu` 为私有 overlay，继承只读 PyTorch 2.7.1+cu126，不是原计划 CUDA 12.4 认证栈。不升级共享环境。
- GFG 训练/预测/恢复入口有 GPU 白名单检查；旧入口和人工命令也必须显式遵守用户四卡要求，不能假定所有历史路径都自动执行同一守卫。
- `launch_gpu_job.py --gpu 0..3` 可记录 GPU/PID/日志/退出码；`--cpu` 清空 CUDA 可见设备。后台启动不等于任务完成。
- 数据盘专属目录用于新 raw，避免挤占余量有限的系统盘；不清理他人数据或旧实验以获取空间。

只汇编/检查本记录的命令（不训练；输出必须是新目录）：

```bash
.venv/bin/python scripts/phase0/compile_experiment_ledger.py --output outputs/veloroute_experiment_ledger_new
```

训练入口仅作索引：[实际 GFG](../scripts/phase0/run_gfg_joint.py)、[pilot 网格](../scripts/phase0/run_gfg_pilot_grid.py)、[直接 router](../scripts/phase0/run_gfg_direct_router.py)、[GFG 恢复](../scripts/phase0/resume_gfg_joint.py)。当前不存在自动启动 full 的指令；历史配置的工程入口不得替代本轮“暂缓”判决。

当前交接入口：[GFG 执行记录](veloroute_gfg_joint_execution_20260914.md)、[执行计划](veloroute_execution_plan_v1.1.md)、[上一版综合结果](../outputs/veloroute_gfg_joint_progress_20260914/RESULTS.md)。本总账及审计包是补充记录，旧结果与配置原样保留。
