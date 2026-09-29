# Kang 全实验执行协议（2026-09-14）

用户授权：当前以 Kang 为主战场，跑通数据、联合 GFG、router、对比与消融。此授权不等于允许选择性报告阳性。RENGE 历史结果、预注册与封存条件不变。

## 唯一主线

真实源侧 S/U 经实际 GFG decoder-JVP，并与 router 联合训练，应帮助预测同一细胞类型内的多峰 IFN 响应。仅静态混合专家胜出、仅均值改善、仅 RNA 重建下降，均不算主假设成立。没有逐细胞配对，不能声称识别了真实单细胞命运或证明了因果分化。

Kang 对照也是培养 6 小时的群体；ctrl→stim 是未配对反事实任务，不是实测起点到终点轨迹。归一化积分时间仅为运输坐标，不能称作 6 小时真实 RNA 动力学。

## 执行顺序与冻结

1. 在查看表达、多峰或测试效果前冻结 YAML：8 donor leave-one-out、3 seeds、预设 CD14+ Monocytes 主分析与其他常见类型次分析、同容量静态 router/真实 GFG/U 置换/raw-SU 对照。七训练供体拟合全部变换；不调外层测试超参；所有注册预测冻结后统一评估。任何实质修改必须版本化并注明是否已经看过外层结果。
2. 下载 batch2 的两份原始 BAM，核验 ENA 校验码、BAM 完整性、barcode/UMI 标签与发布 barcode 对齐。batch1 混合实验不进入 IFN 任务。不能拿发布 mtx 伪造 U。
3. 从原始分子标签恢复 cDNA/BC/UMI，按实际布局重新 USA 定量（保留 S/U/A），统一 ENSG；不拿历史 BAM 的参考坐标硬套新 GTF。数据盘存所有大文件，原始共享数据只读。
4. 各折仅在训练供体中发现响应轴和响应模式，要求同类型、供体内复现。混合 PBMC 的类型组成不计作响应多峰；单 marker 零膨胀不自动视为多峰。预设重复密度审计不是正式 FDR-D4，不冒用该标签。
5. 无泄漏工程 smoke → 固定预算的静态 K1/static-router/联合 GFG-router/输入 U 置换/frozen-GFG/raw-SU 六臂 → 相同专家几何下隔离 router 增量 → 全量 EM/门控与 corruption/K 消融。GFG 必须有实际 decoder-JVP 和来自任务的梯度；c 不进入 GFG。
6. identity/训练集类型均值平移/训练集类型目标重采样与 CellOT、CellFlow、scDFM 分别报告。CellOT 按用户最新指令仅复用已有复现，不新增训练；已有产物待定位和口径核对，不能标作尚未复现，也不能将未核验结果直接纳入主表。其余未运行的官方基线标记待完成，不能给内部模型换名字。
7. 预测 hash 冻结后，独立评估 held-out donor 的分布、训练定义模式比例和逐模式保真；按 donor 汇总、seed 均值后 bootstrap。禁止把细胞、类型或重复 seed 当独立供体。无多峰/样本不足时报告不可判定，不补造二峰。

## 成功判据

- 模式子集由训练数据独立确定，优先 CD14+ Monocytes；同时报告全部预设类型，不能只挑赢的类型。
- 真实 GFG-router 对静态 router 和 U 置换的分布及模式比例四项比较统一校正（每项 98.75% CI），效果方向一致；全体预设类型按 donor 宏平均、相对 2% 非劣界报告。
- raw-SU 对照分清第二模态增量与特定 velocity 表示增量；即使真实 GFG 优于 U 置换，也不直接证明 RNA velocity 的生物真实性。
- Kang 单扰动，只支持 donor 泛化/机制场景，不支持 unseen-perturbation 广度或分化因果主张。

## 资源与证据

仅物理 GPU 0–3；4–7 保留，不终止其他作业。系统盘约 1.6 GB 空余，所有原始数据、矩阵、模型和新实验产物放 `/data/yuchang/veloroute_kang_20260914`。工作区保存代码、配置、运行入口及索引。每个阶段落 `RESULTS.md`、CSV、provenance（输入/代码 hash、环境、seed、失败原因）。长任务由 detached launcher 记录 PID 与状态。

检索使用 ncbi-entrez-skill 的 GEO→SRA 映射核验：GSE96583/PRJNA381100，ctrl GSM2560248→SRR5398238、stim GSM2560249→SRR5398239。ENA submitted-file 元数据表明两份原始 BAM 为 23,764,981,452 与 22,787,097,572 bytes。尚未完成 BAM 标签审计前，不宣称已获取可用 S/U。

来源：[GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583)、[ctrl ENA](https://www.ebi.ac.uk/ena/browser/view/SRR5398238)、[stim ENA](https://www.ebi.ac.uk/ena/browser/view/SRR5398239)、[CellOT](https://www.nature.com/articles/s41592-023-01969-x)。CellOT 对部分 IFN 响应 marker 的多峰展示是候选场景依据，不是本模型优势的证据。

## 本次落地记录（持续追加，非结果选择）

- 不可变协议：数据盘 `protocol_freeze/`，8 供体、24,679 singlets，冻结时未读取表达值。首批 primary grid 是 8 folds × 3 seeds ×（5 个同几何 router 臂 + 单独训练 static K1 + 3 个简基线），合计 216 份预测；外部基线和全量组件分别报告，不冒充包含在这 216 份里。
- 两组 BAM 各审计 10,000 reads：98 bp cDNA、CR 14 bp、UR 10 bp，原始标签覆盖 100%；这是有界前缀审计，不能替代全文库 MD5/QC。BAM 为 hg19 坐标，因此恢复原始读段并重新映射至统一 GRCh38，绝不把 hg19 比对坐标直接配 GRCh38 注释。
- ENA 8 MB 块耗时 16.99 s、NCBI 同块 4.41 s，SHA256 完全一致；切到公开 `sra-pub-src-1` 镜像，保留 ENA 未完成下载。全文库仍需 ENA MD5 验证。
- 已将项目参考目录 6.1 GB 迁至数据盘 `relocated/gencode32`，原路径保留软链接，未删除文件。索引初次构建因底层排序在当前工作目录写临时文件而失败；失败记录在 `commands/build_index/`，v2 改为整个进程以数据盘为工作目录、64 GiB RAM 上限。
- `smoke_v1/` 使用明确合成的 Kang 形状输入，真实 GFG checkpoint/JVP、5 个 router 臂训练/预测均通过，相同候选几何逐元素一致；真实 GFG-router 任务梯度非零。不是 Kang 阳性结果。
- 全仓库回归测试 130 passed，记录 `tests_all_v1.xml`。预测入口已显式使用配置时间，修正 Kang 可能误用 RENGE day4→day5 默认值的问题。

主评估先封存完整 216 份 primary 预测再读取外层终点；已经固定的全量/外部基线不根据这些结果调参。若后续改变算法或超参，必须另立探索版本，不把旧的外层测试当新验证集。

## 执行变更：CellOT 仅复用（2026-09-14）

- 用户明确 CellOT 已经复现，停止安排任何新的 CellOT 复现、训练或源码获取。扩展配置设为 `reuse_existing_only`，训练入口默认拒绝执行（包括 synthetic smoke），后续扩展 smoke 和下载清单移除 CellOT。
- 本轮尚未启动真实 Kang 的 CellOT 训练，无需终止其他作业。此前完成的 `smoke_extended_v1` 内含 2 步合成 CellOT 工程测试；保留其原始记录，不把它当成真实 Kang 基线，也不覆盖或冒充用户已有复现。
- 待定位用户已有结果，核对 donor 切分、细胞类型队列、预处理/基因空间、指标定义及 provenance。兼容则接入；不兼容则分栏报告或在可行时仅重新评估已有预测，不自动重训。不读取这些外层结果来调整 VeloRoute。
- 已冻结的主 YAML 及其 hash 不变。其中 `CellOT_official` 是比较对象，不再意味着安排新训练。此次变更仅写入可变扩展配置和执行文档；Kang 数据、GFG/router 主实验及四卡上限不变。
- 此变更的定向测试：`test_kang_cellot_policy.py` 与 `test_kang_evaluation.py` 合计 12 passed；JUnit 记录在 `outputs/kang_cellot_reuse_policy_20260914/tests.xml`。确认禁用时不会进入 GPU 策略、数据读取或官方训练模块，也不会创建训练目录。主配置与冻结副本规范化哈希均为 `dc4b89318b2277e1d7d44ee2f1d1f66e6d0890e0bf390b67a368febe88780883`。

## 执行变更：先训练主线，CellOT 由他人负责（2026-09-14）

用户明确要求直接训练主线，CellOT 交给另一位执行者。本执行队列不再等待 CellOT 产物、口径核对或任何外部基线。CellFlow/scDFM 也不进入当前主线训练的前置依赖。

- GPU 0–3 各一条顺序队列：完整 `full_GFG_EM_gate`（实际 GFG decoder-JVP 联合训练、router、专家、EM、门控）→ 同几何 router 对照 → 剩余完整模型消融。先跑主模型全部 8 donor × 3 seeds，不因任何外层指标选择是否继续对照。
- 四卡分工仍为供体列表的步长 4 切片，每卡两个留出供体；单卡内串行训练，不新增到 GPU 4–7。不启动任何外部基线训练。
- 只调整执行顺序，不改变主配置、切分、模型参数、步数或评估冻结规则。主模型先完成预测也不提前打开外层效果；主要评估仍需对应注册预测全部冻结。
- 真实训练的硬依赖只有完成校验的 BAM → S/U/A 定量 → QC → 训练折拟合/供体切分。下载中或只有合成 smoke 时，状态必须写“等待真实 S/U”，不能宣称真实 Kang 已开训。
- 入口 `scripts/phase0/run_kang_mainline_queue.py`；队列日志在工作区 `outputs/veloroute_kang_jobs_20260914/mainline_first_gpu*/`，状态及调度 provenance 在数据盘 `schedules/mainline_first_v1/worker*/`。实际模型输出沿用 `full_experiments/<donor>/seed<seed>/full_GFG_EM_gate/`。
- 21:17（北京时间）已启动四条新队列并验证 GPU UUID 仅对应 0–3；旧的四个 primary 等待进程与旧评估等待进程已停止，未停止任何下载/定量任务或他人作业，历史日志保留。新队列均已进入 `--phase mainline`，独立 primary/full 评估等待任务也已接好。
- 21:18 实测仍在等待真实 S/U：ctrl 下载 15,669,919,744 / 23,764,981,452 bytes（65.9%），stim 15,871,246,336 / 22,787,097,572 bytes（69.7%）。当时 GPU 显存为零，真实 Kang 训练尚未开始；下载完成不等于 S/U 就绪，仍需校验、定量和切分。
- 本次定向测试 19 passed，全仓库 146 passed；记录分别为 `outputs/kang_mainline_first_20260914/tests.xml` 和 `tests_all.xml`。冻结主配置与存档内容、规范化哈希完全一致。

## 执行记录：stim 导入修复与主线开训（2026-09-15）

- 昨晚 22:11 `data_stim_v2` 失败：`import_counts` 重建样本键时硬编码 `-1` 后缀。发布元数据 GSE96583 batch2 中有 313 个 stim singlet 的 barcode 后缀为 `-11`（全部 17 字符 barcode、全部 singlet），其裸 barcode 与 `-1` 后缀的 ctrl 细胞相同，导致 313 个键查不到而抛错。quant 本身无损（12,364 行 = stim singlets 全体），已完成的 ctrl 导入不受影响。
- 修复：`kang_data.py` 新增 `sample_keys`，按该 condition 发布 cohort 的裸 barcode→完整 barcode 映射重建键；同 condition 内裸 barcode 冲突或 quant barcode 未知时显式报错。回归测试见 `test_sample_keys_keep_each_condition_published_suffix` 与 `test_sample_keys_reject_ambiguous_and_unknown_barcodes`；`tests/test_kang_data.py` 等 Kang 四件合计 24 passed, 1 skipped（pysam 在 CPU venv 缺席的既有跳过）。
- 真实数据预验证：ctrl 12,315 行解析结果与旧硬编码零差异；stim 12,364 行全部解析、恰 313 个与旧逻辑不同、全部 singlet。
- 重跑 `run_kang_data.py sample --condition stim`：只补导入，未重跑下载/提取/quant；stim 12,364 细胞、12,301 QC pass、60,668 基因，产物在 `processed/stim/`。随后 8 个供体折全部 `KANG_FOLD_READY`，`folds/ready.json` 已写入，合格类型为 CD14+ Monocytes、CD4 T、B、NK（部分折另含 FCGR3A+ Monocytes）。
- 约 16:05（北京时间）四条主线队列自动退出等待并开训：`full_experiments/<donor>/seed0/full_GFG_EM_gate/`，worker0–3 分别从 101/1015/1016/1039 开始，GPU 0–3 均有负载；进程确认为 `.venv-gpu/bin/python`。日志：`outputs/veloroute_kang_jobs_20260914/stim_reimport_v3.log`、`folds_v2.log`。

## 执行变更：消融中止与增益数据集调研（2026-09-15 21:00）

- 用户指令：停止 Kang 全量消融（`full_ablations`），优先调研 velocity 有实际增益的数据集（分岔多峰其次）。四张卡上 followups 训练约 20:50 被 SIGTERM 终止，队列 `schedules/mainline_first_v1/worker*/status.json` 记为 failed（用户指令中止，非技术故障）；已产生部分消融产物保留，不覆盖、不重跑、不纳入主结果。
- 主模型 24 次、router 对照 216 份预测冻结与 primary 评估（`KANG_PRIMARY_GRID_EVALUATED`）不受影响；`evaluation_full` 继续运行。
- 新调研线只读进行：包装好的 S/U 基准数据（scVelo 生态等）、时间分辨扰动 raw、文献中 velocity 增益证据、分岔+raw 候选；GPU 暂不投入训练。
