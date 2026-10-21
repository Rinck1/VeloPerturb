# W15-A / L3 双流专家混合：基础实现

日期：2026-10-21。范围：已完成 Flax/JAX 双流结构、官方 CellFlow hidden 接入、官方 OT 端到端 CPU 工程 sanity；未启动真实数据训练，未占用 GPU。

## 已实现的结构与接口

代码：`scripts/phase0/w15a_cellflow_moe.py`；测试：`tests/test_w15a_moe.py`。

- 共享状态骨干接口 `backbone(x_tau, tau, condition) -> hidden`，只接收状态、插值时间、扰动/生物学时间条件。
- 固定 8 个独立专家头，每个 `hidden -> 128 -> 50`，隐藏层 SiLU。每个头独立随机初始化，避免全部相同参数导致对称锁死；同 seed 各对照臂的全部初始权重完全相同。
- 独立动力学编码器 `E_v(kappa): 50 -> 64 -> 16`，隐藏层 SiLU。编码器不接收原始状态 x、条件、目标细胞或时间。
- router `[source_x(50), E_v(kappa)(16)] -> 64 -> 8 -> softmax`。只由源侧计算一次 q，整个轨迹固定。
- 状态骨干/专家的接口不接收 kappa/m/U。状态条件中若出现 L2 的 `population_token`/`cell_token` 或命名的 U/velocity 字段则直接拒绝。
- 四臂模块与参数形状相同，只切换速度流的输入：零、真实 kappa、同源群体置换 kappa、上游直接 U 特征。
- 训练时先由外部 state-only OT 给出源/目标索引；`align_ot_indices` 同步重排源状态、kappa **和源条件**，支持 OT 重复采样源索引。
- `paired_flow_matching_loss` 用插值状态 `x_tau=(1-tau)x0+tau*x1` 与位移 `u=x1-x0`；固定源路由，计算 `mean_i sum_k q_ik ||v_k-u||² + 0.01 KL(mean_i q_i || uniform)`。没有混合专家速度后再算 MSE，没有均值锚，没有 JVP 或额外 GFG 辅助损失。
- RK4 基础积分器为每个源细胞分别演化 K 个专家状态；`cohort_q`/`uniform_q` 只在源侧替换权重，不改变候选终点。原官方 L2 的求解器与配置仍须待冻结后继承，当前 RK4 仅用于结构测试。
- 提供冻结 U 敏感性审计的 embedding cosine 和 q-TV 汇总接口；真实 U 替换及重新计算 kappa 由上游适配负责，端点 Energy 变化由冻结预测评分负责。

## 官方 CellFlow 集成边界

第一阶段22项结构测试使用 `StubBackbone`；第二阶段已通过 `scripts/phase0/w15a_cellflow_moe_official.py` **实际接入官方隐藏骨干**，不是另写一个普通 MLP 冒充 CellFlow。

官方 `ConditionalVelocityField` 没有公开 hidden 接口。项目局部 `HiddenFeatureField` 继承该类，只覆盖 `_combine_and_decode` 的末尾：保留其条件组合和原 decoder，返回 hidden、不调用最后 `output_layer`。条件编码、状态编码、时间编码、FiLM/ResNet 模块、decoder 均继承官方实现，官方源码没有修改。原最终 Dense 不参与共享骨干初始化，改由 K 个独立专家头输出速度。

包装器只接受 `perturbation, bio_time, bio_delta` 三个外部状态条件；内部补上 L2 `cf_static` 原有的两个同形恒零 token 槽，保留其条件网络容量与参数命名，但绝不从外部接收真实动力学。L2 agent 已确认该做法保留静态条件架构。实际采用的 width/depth 等超参仍等待最终静态开发选择。

测试已证明：hidden 接回原 `output_layer` 可重建 native 速度输出；条件 mean/logvar 一致；同 seed 初始化的全部共享参数逐元素一致；hidden 参数树恰好只比 native 少 `output_layer`；官方 hidden + E_v/router/专家可在 JIT 内联合更新、序列化再预测。

正式 L3 必须等 L2 `cf_static` 骨干、优化器、步数/batch 和求解器配置冻结；`require_l2_freeze` 对缺少冻结标记/提交/配置哈希/官方骨干包装的请求拒绝。该接口要求独立的科学配置文件和官方隐藏包装文件，并实际核验两者 SHA256；同时检查 backbone/optimizer/solver 元数据与文件内容一致。冻结发布环节仍负责核实真实 git 提交归属。尚未做真实网格或 20 正效应 + 20 零效应合成训练，不将结构单测说成完成 CF-2/CF-3。

`arm_input` 是主机侧 cohort 预处理，不放入编译训练步。OT 源重排分成 `align_ot_indices` 的主机范围验证与 `align_ot_indices_jit` 的纯 JAX 核心；编译核心仅检查静态形状/dtype，索引值由官方 state-only OT 采样器保证合法，不能用 JAX 越界钳制替代验证。已用含重复源索引的动态 si/ti 编译配对与完整 paired loss。

L2 当前冻结输出的科学配置实际是 flat JSON，backbone/optimizer/solver 摘要另存于 `freeze_record.json`。已添加只读 `l2_scientific_sections` 映射兼容该真实 schema；不为了门的结构去改写已训练 L2 配置，也不把开发状态的 `freeze_commit=null` 当作冻结完成。

## 专家分化并非目标自动保证

计划规定的损失满足恒等式（vbar = sum_k q_k v_k）：

`sum_k q_k ||v_k-u||² = ||vbar-u||² + sum_k q_k ||v_k-vbar||²`。

因此即便初始化独立，所有专家收敛到相同速度仍是允许的解；批平均路由的均衡项只鼓励使用率均衡，不保证专家分化或多峰。没有擅自替换损失、加入分离正则或使用真实分支标签配对。

后续合成检验必须用与正式实验相同的 **S-only 小批量 OT 配对**。已知的隐藏变量/逐细胞命运只能用于生成和评分，不能暗中用作训练配对来获得阳性；若 OT 抹掉了动力学对应信息，必须把失败如实报告，不能用配对真值替换后仍声称通过该训练方案。

另有损失尺度差异：官方 L2 `_otfm.py` 使用 `mean((v-u)^2)`，同时对细胞和坐标取平均；本计划 L3 的向量范数平方按字面实现为对50个坐标求和、只在细胞上平均。保持原计划的 `lambda_bal=0.01`，没有擅自除以50或调整均衡系数。因此两条线的原始 loss 数值不可直接对比，均衡项相对尺度也不同；正式冻结时应明确这一约定。

## 能量距离的有限样本修正说明

精确加权经验混合分布的 V-statistic Energy 包括同一源细胞不同专家的项：

`within_exact = N^-2 sum_ij sum_kl q_ik q_jl ||x_ik-x_jl||`。

若每个源细胞仅抽一次专家，再用同一批抽样端点计算 V-statistic，所有 i=j 项恒为零，而精确混合的 i=j、k!=l 项一般非零。因此：

`E[Energy_one_draw] - Energy_exact = N^-2 sum_i sum_kl q_ik q_il ||x_ik-x_il|| >= 0`。

原计划“1000次单抽样平均 Energy 与精确加权 Energy 严格相等”的假设不成立，偏差随 N 减小更明显。本实现保留精确加权评估，但 MC 一致性测试使用 **两份独立专家抽样 A、B** 计算 within 的所有 i,j 项；cross 对 A/B 取平均。这对固定源/目标支持上的专家分配期望无偏，单次估计可能为负。不是新的生物学采样置信区间。

单测构造 N=1、专家端点 -1/+1、各权重 1/2、目标0：精确 Energy=1，单抽样 V-statistic 恒为2；1000次独立双抽样均值在 MC 误差内接近1。

## 测试记录

- 初版：14/14，通过；`outputs/w15a_l3_cpu_v1/pytest.xml`（7.27秒；测试骨干类命名曾触发无害 collection warning，后已修正）。
- 扩展版：18/18，通过；`outputs/w15a_l3_cpu_v2/pytest.xml`（9.43秒，无 warning）。
- 第三版：20/20，通过；`outputs/w15a_l3_cpu_v3/pytest.xml`（9.88秒）。
- 第四版：22/22，通过；`outputs/w15a_l3_cpu_v4/pytest.xml`（9.95秒）。新增动态索引 JIT 和实际冻结文件哈希/篡改拒绝检查。
- 最终回归：22/22，通过；`outputs/w15a_l3_cpu_v5/pytest.xml`（9.96秒）。另覆盖 Flax FrozenDict 中的非法动力学条件拒绝；摘要含实现/测试/规格文件哈希，见同目录 `summary.json`。
- 覆盖：注册默认50维/K8、专家非对称初始化、四臂等参同初始化、双流隔离、任务梯度进入 E_v/router/专家/骨干、q源侧一次计算、各专家独立积分、零时长恒等、常速率解析解、OT源属性与条件一致重排、精确期望损失、MC对角偏差、冻结前置。

## 官方 CPU 接入与合成张量工程 sanity

代码：`scripts/phase0/w15a_cellflow_moe_official.py`、`scripts/phase0/w15a_cellflow_moe_sanity.py`；接入测试：`tests/test_w15a_moe_official.py`。

- 首轮官方接入测试 6/6 通过，9.37秒：`/data/yuchang/veloroute_w15a/l3_official_cpu_tests_v1/pytest.xml`。
- 第二轮合并测试29项通过、1项为 pytest 临时目录父路径缺失导致 setup 失败；旧记录保留于 `/data/yuchang/veloroute_w15a/l3_official_cpu_tests_v2/pytest.xml`，不是模型/数学断言失败。新建输出父目录后完整重跑，见 `l3_official_cpu_tests_v3/`。
- 第三轮30/30通过，20.83秒。加入L2实际flat配置schema的正例后，第四轮31/31通过，20.57秒。冻结前最终复验31/31通过，20.77秒，见 `/data/yuchang/veloroute_w15a/l3_official_cpu_tests_v5/pytest.xml`。
- 50维、K8、四臂各20步、官方 state-only Sinkhorn OT 和 `sample_joint` 的小型 CPU 流程已完成：`/data/yuchang/veloroute_w15a/l3_official_cpu_sanity_v1/summary.json`，10.78秒。
- 四臂都从完全同一初始化开始，参数量均 87,576（仅是 width16/depth1 的工程配置，**不是最终 L2/L3 参数规模**）。更新、权重序列化/恢复、每源8候选独立积分和精确加权 Energy 全部有限。
- `moe_kin` 最终 E_v/router 任务梯度范数分别约 0.01134 / 0.05287，证明官方骨干合并后的任务梯度可达动力学路由路径。static 的 E_v 偏置也可以更新；它仍只读恒零输入，不意味着读取 U。
- 该 sanity 使用随机合成张量，不是20正效应+20零效应实验；分数不用于宣称动力学收益。没有按分数调参，没有已知命运配对。
- 运行显式限定 `JAX_PLATFORMS=cpu, CUDA_VISIBLE_DEVICES=''`，实际设备 `TFRT_CPU_0`。CUDA插件发现阶段曾打印无可见设备警告，CPU流程正常完成、未占用GPU。
- 所有新增较大输出位于数据盘；复用 `tools/w15a-cellflow-deps` 指向的数据盘私有依赖，不升级全局环境。

官方速度场源码：`/data/yuchang/veloroute_kang_20260914/external/CellFlow/src/cellflow/networks/_velocity_field.py`；运行前后 SHA256 均 `6105d731c08c7912c653ba44135b74f9dee5355b9f1510bb599f3892cb549d02`。

下一步仍是等待 L2 静态配置冻结后原样继承；完成接入不等于已获准运行真实 L3 网格。最终求解器需继承 L2 的 Tsit5/容差；当前 sanity 的四步 RK4 只是接口检查。

集中提交用小型产物：`outputs/w15a_l3_completed_20261021_v1/`，只含 JSON/XML 摘要和源码哈希，不含权重、数组或原始数据。

未读取 Traxler 表达/扰动效应；未改 src、官方 CellFlow 源码、任何冻结 W14/B3 文件或 PLANS 原文。
