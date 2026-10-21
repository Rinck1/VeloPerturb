# 短时群体预测基线复现实验方案

日期：2026-10-18。对象：VeloRoute 基线实验执行者。

2026-10-21衔接：10月19日推送的[W15-A](../w15a_cellflow_kinetic_plugin.md)为当前CellFlow三线执行规格。
CellFlow主线改为L2群体token插件和L3双流router专家，均不加L1均值锚；仅cf_static可在r4开发集
按上限12配置选型。本文原训练内调参、共享均值扩展和24次拟合预算保留作10月18日基线设计参考，
不能覆盖W15-A当前主线。下列绝对数据路径继续有效；Traxler九份S/U已于10月18日生成，
后文“待生成”是原清点时状态，最新进度见[当前状态](../../docs/current/STATUS.md)。

优先在 sci-FATE2 的 5h、10h 群体预测上复现 CellFlow、CellOT，再迁移到遗传扰动数据。
实验分别回答端到端预测是否领先、共享均值后结构是否更好、动力学路由是否优于直接输入 U 或 velocity。
完整报告 15h 和 24h 边界，不按测试成绩筛选起点或数据子集。

当前状态是实验设计与数据路径已整理，新增基线训练尚未启动。正在运行的 W14 和排队的
RENGE 实验继续按原冻结协议执行；本方案不改写它们的训练、指标和判决。
Traxler 当前仍限于技术处理，模型训练须在后续独立实验协议冻结后开始。

## 1 当前方法与优势场景

当前方法是解析源侧邻域 U−γS 上下文、soft router、专家端点或小 ODE。
没有原版 GFG、JVP 或新的基础模型预训练。现有正向信号来自 sci-FATE2 同系列开发实验：

| 工作模型 | 视野 | Energy | 相对同模型表达路由 | 相对打乱动力学路由 |
|---|---:|---:|---:|---:|
| b1 端点 36k | 5h | 0.595683 | +4.137% | +4.606% |
| 小 ODE | 5h | 0.589504 | +3.723% | +4.620% |
| 小 ODE | 10h | 1.027032 | +6.358% | +7.418% |
| 小 ODE | 15h | 1.007833 | +2.415% | +1.573% |

这些是相对内部对照的提升，不是已经优于 CellFlow 或 CellOT。
内部的 static 只表示局部路由不读 velocity；各臂共享的均值分支仍使用 S/U。
结果来源：[W13 结果](../../docs/current/w13_time_results.md)、
[机器汇总](../../outputs/w13_time_program_v1/final_report/summary.json)。

基线主场为短时、转变已启动后的群体分布预测。CellFlow 和 CellOT 本身也能预测异质分布，
因此不预设它们必然塌峰；检验的是源侧动力学能否进一步改善分布形状和响应成分比例。

## 2 数据集路径与现有产物

仓库绝对根目录为 `/home/yuchang/wangjiaxuan`。以下路径在 2026-10-18 核实；
包含 `{...}` 或 `*` 的写法表示文件族，表中另列尚待生成的产物。
原始矩阵、权重与细胞级预测保留在服务器，不要求从 GitHub 获取这些大文件。

### 2.1 sci-FATE2 主实验数据

GEO：GSE236512。主矩阵已经存在，大小为 4,505,323,176 bytes。

```text
/home/yuchang/wangjiaxuan/data/scifate2/raw/GSE236512_processed_data_splicing.h5ad
```

该文件的冻结 SHA256：

```text
4269f26b5d53d436c692210d965233ed66f2d3d4a6ff0e1c0a6713b108816944
```

训练读取 `rep ∈ {r1,r2,r3}` 与 `timepoint ∈ {D3,D4,D5,D6,D7,D8}`；
小时级评估读取 r4 的 `05h/10h/15h/20h`。训练配对是每个重复内部的五个相邻日级转移，
共 15 个训练群体。基因和计数层以 H5AD 的 `var`、`layers['spliced']`、
`layers['unspliced']` 为准，不把 `X` 自动视作原始 S 计数。

| 用途 | 已有绝对路径 |
|---|---|
| 数据配置与训练时间定义 | `/home/yuchang/wangjiaxuan/configs/w5_joint_kinetics_scifate2_v1.json` |
| 冻结 HVG2000 与 PCA50 变换 | `/home/yuchang/wangjiaxuan/outputs/w5_joint_kinetics_scifate2_pack_v1/source_transform.npz` |
| 逐细胞训练与源侧特征包目录 | `/home/yuchang/wangjiaxuan/outputs/w6_local_router_scifate2_pack_v1/` |
| 逐细胞包清单与 cell ID 来源 | `/home/yuchang/wangjiaxuan/outputs/w6_local_router_scifate2_pack_v1/manifest.json` |
| 逐细胞训练包 | `/home/yuchang/wangjiaxuan/outputs/w6_local_router_scifate2_pack_v1/train_{rep}_D{day}.npz`，rep 为 r1–r3，day 为 3–7 |
| 路由输入标准化 | `/home/yuchang/wangjiaxuan/outputs/w6_local_router_scifate2_pack_v1/normalization.npz` |
| r4 05h 源侧包 | `/home/yuchang/wangjiaxuan/outputs/w6_local_router_scifate2_pack_v1/source_05_to_10.npz` |
| r4 10h 源侧包 | `/home/yuchang/wangjiaxuan/outputs/w6_local_router_scifate2_pack_v1/source_10_to_15.npz` |
| r4 15h 源侧包 | `/home/yuchang/wangjiaxuan/outputs/w6_local_router_scifate2_pack_v1/source_15_to_20.npz` |
| 原评价源细胞抽样清单 | `/home/yuchang/wangjiaxuan/outputs/w13_scifate2_direct_continuation_v1/source_samples.json` |
| 评价目标缓存清单 | `/home/yuchang/wangjiaxuan/outputs/w13_scifate2_direct_continuation_v1/study_manifest.json` 的 `evaluation_caches` |
| 10h 目标评价缓存 | `/home/yuchang/wangjiaxuan/outputs/w7_scifate2_router_scaling_v1/evaluation_5_to_10.npz` |
| 15h 目标评价缓存 | `/home/yuchang/wangjiaxuan/outputs/w7_scifate2_router_scaling_v1/evaluation_10_to_15.npz` |
| 20h 目标评价缓存 | `/home/yuchang/wangjiaxuan/outputs/w7_scifate2_router_scaling_v1/evaluation_15_to_20.npz` |

逐细胞训练包包含 `z/velocity/target/global_rates/cell_ids`；源侧包包含
`z/velocity/cell_ids/source_mean/anchors/duration`。源侧包没有目标细胞，文件名中的后一个时间
不是可读取的未来输入。10h、15h 预测继续使用对应起点的源包，不读取中间时刻的细胞来滚动纠正。

特别注意：`outputs/w5_joint_kinetics_scifate2_pack_v1/training_pack.npz` 是群体层
`base/velocity/delta`，不能代替 CellFlow、CellOT 所需的逐细胞分布。
现有 W6 包没有 raw-U 投影；CellFlow＋U 需按其中 `cell_ids` 回主 H5AD 取 S/U，
复用同一 PCA、S 深度因子和源侧 64-NN 池化，新增 `W log1p(U)` 固定上下文。
这个 U 对照不使用 γ，也不把 velocity 当作 U。

### 2.2 现有 VeloRoute 权重与评价入口

以下文件族已存在，`{arm}` 为 `velocity_router/static_router/shuffled_velocity_router`，
`{seed}` 为 0、1、2。

```text
# b1 端点 36k，76,248 参数
/home/yuchang/wangjiaxuan/outputs/w13_scifate2_direct_continuation_v1/training/b1_direct_distance/{arm}/seed{seed}/model_step36000.pt

# 小 ODE，76,760 参数，保存文件中的 4k 是追加训练步数
/home/yuchang/wangjiaxuan/outputs/w13_time_program_v1/time_training/training/ode/{arm}/seed{seed}/model_step4000.pt

# 冻结解析均值分支
/home/yuchang/wangjiaxuan/outputs/w5_joint_kinetics_scifate2_v1/seed{seed}/analytic_task.pt

# ODE 3h 推理步长与原权重哈希
/home/yuchang/wangjiaxuan/outputs/w13_time_program_v1/ode_refinement_spec.json

# 既有评价程序
/home/yuchang/wangjiaxuan/scripts/phase0/w13_frozen_horizons.py
```

固定 b1 为端点参考、小 ODE 为时间模型；两者都报告，不根据每个窗口选胜者。
ODE 按已保存的 3h 推理规格复算，保持全源群体耦合中心化，不独立切块积分后拼接。
旧评价脚本只接受既有模型类别；扩展外部方法时使用新适配器，不改冻结的 W13/W14 代码。

### 2.3 RENGE 扰动迁移数据

GEO：GSE213069。以下是 W4 冻结清单真实使用的路径，day4 与 day2/day3 不是同一版本目录：

```text
# day2
/data/yuchang/veloroute_w4/renge/processed_release_v2/day2/day2.h5ad

# day3
/data/yuchang/veloroute_w4/renge/processed_release_v2/day3/day3.h5ad

# day4
/home/yuchang/wangjiaxuan/data/renge/processed_release_v1/day4/day4.h5ad

# ESM2 条件嵌入
/home/yuchang/wangjiaxuan/data/renge/conditions/esm2_3b_v1/conditions.npz

# 精确数据哈希与共同 16 TF 名单
/home/yuchang/wangjiaxuan/outputs/w4_protocol_lock_v1/protocol_lock.json

# 原训练 TF 与留出 TF 行号、变换和均值结果
/home/yuchang/wangjiaxuan/outputs/w4_h1_predictions_v1/grid_manifest.json

# 已准备的 32 折逐细胞包与本轮冻结清单
/home/yuchang/wangjiaxuan/outputs/w14b_renge_router_v1/folds/
/home/yuchang/wangjiaxuan/outputs/w14b_renge_router_v1/study_manifest.json
```

任务是 day2→3、day3→4 各留一 TF；源细胞已经受扰动，属于 early-response 预测。
新增 CellFlow 沿用这些行号、共同基因空间和训练折条件嵌入。
基线不能因为全文件都在磁盘上，就使用留出 TF 的目标细胞来训练。

原版 CellOT 为每个扰动学习映射，没有原生未见 TF 条件接口，因此该未见 TF 主表标为不适用，
不计失败，也不训练该 TF 目标后冒充泛化。CellOT 的同条件跨重复扰动验证放到 Traxler。

### 2.4 Traxler 小时级扰动数据

GEO：GSE263761；BioProject：PRJNA1099196。数据根目录：

```text
/data/yuchang/veloroute_w4/traxler2025/
```

| 类型 | 绝对路径与状态 |
|---|---|
| 作者 filtered 矩阵，9份已下载 | `/data/yuchang/veloroute_w4/traxler2025/GSE263761_CROPseq_KO150_{library}_filtered_feature_bc_matrix.h5` |
| 作者 raw 矩阵，9份已下载 | `/data/yuchang/veloroute_w4/traxler2025/GSE263761_CROPseq_KO150_{library}_raw_feature_bc_matrix.h5` |
| feature reference | `/data/yuchang/veloroute_w4/traxler2025/GSE263761_feature_reference.csv.gz`，已下载 |
| GEO 样本元数据 | `/data/yuchang/veloroute_w4/traxler2025/GSE263761_family.soft.gz`，已下载 |
| SRA run 元数据 | `/data/yuchang/veloroute_w4/traxler2025/PRJNA1099196_runs.tsv`，已下载 |
| GEX SRA | `/data/yuchang/veloroute_w4/traxler2025/sra/`，18个文件下载中；`.partial` 不是完整输入 |
| 小鼠参考 | `/data/yuchang/veloroute_w4/traxler2025/reference_ensembl93/`，FASTA/GTF 已下载 |
| USA 索引 | `/data/yuchang/veloroute_w4/traxler2025/reference_ensembl93/splici_r92_v1/`，已构建 |
| USA 定量输出 | `/data/yuchang/veloroute_w4/traxler2025/quant_usa_v1/{library}/`，待各文库两个 run 到齐后生成 |
| S/U H5AD | `/data/yuchang/veloroute_w4/traxler2025/processed_usa_v1/{library}.h5ad`，规划产物；核实时目录存在但尚无 H5AD |
| 下载与定量状态 | `/home/yuchang/wangjiaxuan/outputs/w14b_traxler_raw_v1/status.json`、`/home/yuchang/wangjiaxuan/outputs/w14b_traxler_su_v1/status.json` |

`{library}` 为 A1/A2/A3/B1/B2/B3/C1/C2/C3；A 为感染前 0h，B 为感染后 6h，C 为 24h。
数字 1–3 是文库/芯片号，不能直接当生物重复；需按作者 HTO 规则恢复独立来源和排除 doublet。
作者 h5 不能替代尚未生成的 S/U 文件。后者计划保留 `spliced/unspliced/ambiguous` 三层，
基线表达层须共同显式选择；不得将生成文件的 `X=S+A` 自动等同 sci-FATE2 的 S-only。

优先任务是同 KO、不同来源留出的 0→6h 预测，CellFlow 和 CellOT 均参加；
6→24h 是另外的 18h 任务，不混成 6h 平均成绩。0h 已存在 KO，本任务不是从未敲除细胞
预测新 KO。HTO 关系、细胞覆盖度和技术 QC 完成后，再冻结未来训练/测试名单及训练配置。

## 3 第一轮比较方法与训练预算

| 方法 | 输入和作用 | 第一轮正式拟合数 |
|---|---|---:|
| Identity、mean-shift、仿射均值/协方差匹配 | 简单分布变换参照，仅在训练群体拟合 | 不计神经训练次数 |
| CellFlow | 官方网络/目标，使用 S-PCA 状态与阶段条件 | 3 seeds，共 3 |
| CellFlow＋U | 在相同框架中加入固定源侧 pooled-U 上下文 | 3 seeds，共 3 |
| CellFlow＋velocity | 加入与本方法相同的源侧动力学上下文，不加 router | 3 seeds，共 3 |
| CellOT | 五个日级转移分别训练官方 ICNN 映射，重复间合并同一阶段 | 5 阶段 × 3 seeds，共 15 |
| VeloRoute b1 与小 ODE | 每版均报告 real、表达路由、打乱动力学 | 复用现有冻结权重 |

第一轮共 24 次新的正式神经模型拟合；这不包含适配测试和训练内超参数试验。
CellFlow 的三种输入扩展共享同一训练群体、阶段条件及选择规则。
外部基线允许使用合理的较大容量，不强行压到本方法的约 7.7 万参数；另报实际容量和耗时。

CellFlow＋U/velocity 的上下文固定于出发细胞。训练 OT 配对 `(i,j)` 后，携带的是源细胞 i
的上下文，目标仍为 50维 S 状态；不把源 U 拼成虚构的目标 U。
CellOT 保留 S 空间中的凸势梯度映射，不直接把 100维 S/U 拼入 ICNN 后随意截取50维输出。

## 4 训练切分与小时级时间适配

### 4.1 训练和检查点选择

使用 r1–r3 的 15 个相邻日级训练群体，复用原 HVG2000/PCA50 和共同细胞 ID。
正式超参数在 r1/r2 训练、r3 日级验证中确定，随后按固定配置在 r1–r3 重训。
内验证的变换和标准化只在 r1/r2 拟合；配置确定后的正式 r1–r3 比较才复用上述共同变换。
r4 不参与优化、检查点选择、阶段选择规则调优或时钟拟合。
本轮 r4 仍标记为同系列开发基准，不包装成此前从未访问的独立测试。

每个外层训练折单独拟合其标准化；若扩展为 W14 留一重复任务，必须使用该折自身变换与
训练源行号，不能复用含留出重复的全量 PCA。
训练配对按重复和相邻时间段组织，不把不相邻天的细胞随机配成一个训练转移。

### 4.2 共享时钟

原版24h终点和小时级适配分别命名、分别报告。沿用已有冻结时钟：

```text
h(Δt) = 24 × (1 − exp(−Δt/24)) / (1 − exp(−1))
a(Δt) = h(Δt) / 24
```

Δt 单位为小时，a(0)=0、a(24)=1。时间映射对所有小时级方法相同，不用 r4 未来表达校准。
CellOT-Time 使用 `z + a(Δt) × (T_24h(z) − z)`。
CellFlow-Time 沿学习到的向量场从 flow-time 0 积分到 a(Δt)，不是把路径时间直接宣称为生物时间。
可另列 `a_linear(Δt)=Δt/24` 敏感性结果，但不能在两种时钟间按测试分数选主结果。

CellFlow 五个训练阶段使用固定阶段条件，CellOT 五个阶段使用独立映射。
r4 没有对应的日级阶段标签时，统一仅根据源侧 S 分布选择阶段：对训练阶段的等重复权重
源分布与当前源分布计算 Energy，选择距离最小的阶段；平局选较早阶段。
距离子样本每组至多512细胞、不放回、固定 seed 20261018。选择结果为各起点保存一次，
对 CellFlow 三臂和 CellOT 共用，不读取目标、不在各方法间另挑阶段。

24h 原版比较只用于具有明确同阶段目标的日级留出任务，或训练内复现检查；
不把训练集分数当泛化分数。小时级分数明确标为 CellFlow-Time/CellOT-Time 适配结果。

### 4.3 全部小时级窗口

| 视野 | 固定窗口 | 报告层级 |
|---|---|---|
| 5h | 5→10、10→15、15→20 | 主结果，三个窗口等权 |
| 10h | 5→15、10→20 | 主结果，两个窗口等权 |
| 15h | 5→20 | 时间边界，单窗口 |

同时单列固定05h起点的5/10/15h结果。不同起点窗口的平均值不是同一起点的衰减曲线。
源抽样复用 `source_samples.json`，长视野使用相同起点的源 ID；
目标按目标小时查评价缓存，不因文件名沿用相邻转移就误读源或中间时刻。

## 5 两张主表和异质性分析

第一张是端到端性能表：各方法保留自己的均值与分布预测，比较完整任务竞争力。
基础 CellFlow/CellOT 为表达输入，输入扩展另列，不能把多模态收益全部归为 router。

第二张是共享均值结构表，对所有方法执行相同变换：

```text
x_shared = x_pred − mean_full_prediction + mu_shared
```

`mu_shared` 来自相应 seed 的冻结解析均值分支，只读训练模型与源 S/U。
平移前的均值取完整源群体的加权预测均值，而不是512源评价子样本的偶然均值；
之后再使用相同源 ID 评分。该表所有方法均得到同一 S/U 均值信息，
因此列名标注 shared-mean，不把它们称为原版纯 S 方法。不得以目标真实均值纠正预测。

高异质性子集仅由源侧定义：以冻结 S-PCA 的总方差 `trace(Cov(z_source))` 为分数，
阈值取15个训练源群体分数的中位数。r4 群体按该阈值分层，全体和两层都报告。
这是表达异质性分层，不自动等于多峰或命运分岔。若另做模式比例评价，须先冻结训练定义的
表达成分与支持判据；对预测和目标使用相同成分规则，不直接把网络专家编号当目标真标签。

## 6 指标与统计口径

主指标为精确加权经验 Energy，定义与既有 W13 的
`scripts/phase0/w7_router_scaling.py::blocked_energy` 一致：

```text
ED = 2 E||X−Y|| − E||X−X'|| − E||Y−Y'||
```

固定训练折 PCA 坐标，使用 V-statistic 和 float64 精确距离。
W13 主表不额外除以 sqrt(50)；若另报归一化 Energy，要给所有方法同时除并另命名。
W14 日级脚本有其自身的归一化，不能把两个协议的原始数值直接拼成同一表。

结构指标为总方差比及其绝对 log 误差、协方差差异，必要时加入训练定义的模式比例误差。
辅助指标为均值 MSE、固定训练带宽的 MMD、相同细胞数下的覆盖度；
基因空间指标限于共同可评估基因，使用相同反 PCA 口径。
若报告响应 Δ 指标，所有方法减去同一源侧参考，不能只比较绝对表达相关系数。

每窗口至多512个相同源细胞、相同目标细胞，样本清单冻结。
Energy 对本方法的专家混合直接加权积分；CellOT/确定性 CellFlow 的单点对应权重为1。
PRDC 等依赖样本数量的指标为每方法抽相同数量细胞，不能把8专家展开成8倍样本后比较覆盖度。
同时报告逐 seed 分数和三 seed 等权预测混合，不只给本方法集成而给基线单seed。

r4 窗口重叠，训练 seed 不是生物重复。当前第一轮报告完整逐窗数值和视野等权平均；
新扰动数据的统计按其实际生物来源和 KO 组织，不把细胞数当独立重复数。

## 7 官方代码和训练配方

已下载的官方代码目录存在：

```text
/data/yuchang/veloroute_kang_20260914/external/CellFlow/
/data/yuchang/veloroute_kang_20260914/external/cellot/
/data/yuchang/veloroute_kang_20260914/external/scDFM/
```

这里只复用已获取的源码，不复用 Kang 的训练数字冒充本任务结果。
执行前锁定实际源码 SHA/文件哈希和依赖环境；CellFlow 优先官方 JAX 实现，
已有 PyTorch 移植用于交叉检查，不能将其与官方结果混称。

CellOT 官方起始配置为4层64维、Adam lr=1e-4、betas=(0.5,0.9)、
100,000次外层迭代和每次10次内层更新。保留原目标和凸性约束，
不因本方法训练36k步就把不同算法都截成36k并声称同预算。
CellFlow 正式架构、batch、学习率、训练上限与 solver 容差在训练内验证后写入新增配置并冻结。
各方法给相同的小规模调参机会，最终报告参数量、更新数、墙钟时间和推理时间。
先做训练内1000–2000步测速，再估算整体耗时；本方法几分钟的小模型耗时不适用于这些基线。

相关官方来源：

- [CellFlow 官方仓库](https://github.com/theislab/CellFlow)
- [CellFlow 数据与条件接口](https://cellflow.readthedocs.io/en/latest/user/genapi/cellflow.model.CellFlow.prepare_data.html)
- [CellFlow 网络配置接口](https://cellflow.readthedocs.io/en/latest/user/genapi/cellflow.model.CellFlow.prepare_model.html)
- [CellOT 官方仓库](https://github.com/bunnech/cellot)
- [CellOT 官方训练配置](https://github.com/bunnech/cellot/blob/main/configs/models/cellot.yaml)
- [CellOT 原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10630137/)

## 8 执行顺序与新产物位置

1. 从上述路径建立共同逐细胞数据包，按原 cell ID 补齐 pooled-U，验证与现有 PCA 状态一致。
2. 固定训练内选择方案与正式配置；检查原版24h任务、零时长恒等、源/目标方向和源上下文索引。
3. 训练第一轮24个正式模型。训练、选择和阶段匹配不读取 r4 未来评价缓存。
4. 保存所有方法、seed、六窗口的源侧预测后统一评分，生成原生均值和共享均值两张主表。
5. 扩展日级留一重复比较时，重新使用 W14 各折训练数据和变换；不改写已运行的 W14 判决。
6. CellFlow 接入 RENGE 的既定32折未见 TF 任务，结果作为新增外部基线单独保存。
7. Traxler 的 HTO/S/U 技术验收和后续协议冻结完成后，进行真实6h跨来源比较，
   再单列18h时间边界及未见 KO 任务；CellOT 只参加其支持的同条件泛化任务。

以下是本方案为后续实现预留的路径，不是已存在的模型、配置或成绩：

```text
# 训练配置，待适配和训练内选择完成后生成并冻结
/home/yuchang/wangjiaxuan/configs/short_time_baselines_scifate2_v1.json

# 第一轮 sci-FATE2 基线输出，待创建
/home/yuchang/wangjiaxuan/outputs/short_time_baselines_scifate2_v1/
  data_manifest.json
  source_stage_selection.json
  training/{method}/stage{day_if_CellOT}/seed{seed}/
  predictions/{method}/seed{seed}/{start}_to_{stop}.npz
  evaluation/metrics_native.csv
  evaluation/metrics_shared_mean.csv
  evaluation/horizon_summary.csv
  evaluation/RESULTS.md
```

输出记录源数据/代码/配置哈希、训练与抽样 seed、预测 ID 和训练环境。
发生实现错误时保留旧目录、另开版本，不根据性能不理想反复改测试方案。
本轮仍只使用物理 GPU1–3，等待当前 W14/RENGE 队列释放资源；不抢占其他卡或停止现有任务。

第一轮核心交付是外部基线完整结果表，而不是预填正向结论。
若共享均值下仍领先、优于 CellFlow＋U 和 CellFlow＋velocity，再在独立6h扰动任务上复现，
则分别支持结构预测收益、动力学组织方式的价值和扰动任务可迁移性。
