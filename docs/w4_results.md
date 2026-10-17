# W4 短时群体预测执行结果

更新：2026-10-17。预注册仍为草案，**尚未冻结**；复制集 day2→3、day3→4 尚未做响应/预测分析。本页区分实际完成结果与后台正在处理的数据。

## 当前判断

在重新建立的 source-only 流程中，RENGE day4→5 的群体速度特征使平均 MSE 降低 **2.30%**，四个开发 TF 均优于静态和 U 交换对照。按预注册采用的“先算各 TF 相对增益再等权平均”，对应 **3.28%**。这是可以继续做复制检验的开发信号。

直接 U 基线的平均 MSE 改善 **2.36%**，TF 等权相对增益 **2.82%**；两种汇总下与速度的排序略有变化，差异很小。现有数据更支持“相对当前 S-PCA 基线，群体 U 表征值得测试”，尚不足以把收益归于速度变换或路由机制。下一阶段保持已写定的预测器和判据，检验新时间段。

## W4-0：原版 GFG 重训机制审计已完成

两数据集各三 seeds、每次 10 epochs 的原生目标域重训均已完成。CPU 复核六个 checkpoint 哈希、保存速度及交换映射，无需再重复训练。

U 置零后，原生 GFG 与真实 U 输入下速度方向的逐细胞中位 cos：RENGE **0.99813–0.99967**，MeRLin **0.99417–0.99830**。解析 moments 参照分别为 **−0.19402**、**−0.07345**。说明当前原生训练配置的输出方向对 U 不敏感；模长仍会变化。这不是 U 没有预测信息的证据。

完整表、变体定义和来源见 [W3 报告](w3_results.md) 与 [W4-0 结果包](../outputs/w4_native_gfg_audit_v1/RESULTS.md)。历史原生 GFG 对照包含 JVP；当前 W4 主预测使用简单解析估计器，不调用该 JVP。

## W4-1：数据落地

| 数据 | 实际进度 | 可支持的下一步 |
|---|---|---|
| RENGE day4 同输入重现 | 完成；S/U/A 全矩阵差异条目均为 0，逐细胞总量相关均为 1 | 原流程口径一致，允许推进 day2/day3 |
| RENGE day3 | 完成；6091 细胞、4153 同时通过技术 QC/guide；S/U 中位数 12163/921 | 保持复制效应封存，等待 day2 共同 TF QC |
| RENGE day2 | 修复后独立恢复作业运行中；首条 SRA 已成功提取 index/barcode/cDNA 三个 FASTQ | 接续后续 raw 获取、提取及定量，无需重跑 day3 |
| sci-fate | 已下载 229.72 MB processed，构建 7404×43167 total/new h5ad | total/new 技术对齐完成，基因级 S/U 尚缺 |
| scNT-seq | 已获取并严格配对 13476×37007 的 S/U/new/old/total，0/15/30/60/120 分钟 | 可准备分钟级机制验证；须先冻结外部分析协议 |
| MIX-seq | 原论文、作者 processed 打包代码已核查 | 公开 raw accession 尚未核实，暂不能训练 velocity |

RENGE 原后台队列：`outputs/w4_renge_day23_job_v1/`；已完成 day3，day2 首次提取因 `fasterq-dump` 被解析为多调用分派器而退出，已修复并在 `outputs/w4_renge_day2_job_v2/` 恢复；首条提取命令实际 exit0，保留失败日志与已有 raw。数据根目录 `/data/yuchang/veloroute_w4/renge/`；day3 H5AD SHA256 为 `3bf623c7dc09cbb03ec76c5cdb42a379b45cb28f9f9b83b8ca26507cf5b718a9`。day2/day3 完成后才填最终共同 TF 集并冻结协议。技术清单工具 `w4_cohort_qc.py` 只读 metadata 生成固定门槛下的 H1/H2 TF 交集，不自动解封响应。sci-fate 完整 raw 约 116.28 GB，超过任务书的 50 GB 阈值，本轮未下载。

具体来源、化学布局、哈希、缺失模态与文件路径见 [RENGE 审计](w4_renge_data_audit.md) 和 [外部数据审计](w4_external_data_audit.md)。

## W4-3：可运行的短时核心与校准

已实现六臂：`static`、`vel_pool`、`vel_mean`、`vel_pool_Uswap`、`pseudo_S`、`direct_U_pool`。HVG/PCA/双尾 γ/k-means 只用训练源细胞拟合；所有计数先以全基因 S 深度归一化再取 HVG。目标仍为细胞等权 mean(log1p S) 的 PCA 坐标。

`vel_pool` 先合并 raw counts 再归一化，属于群体动力学特征；`vel_mean` 是固定源归一因子下的逐细胞导数平均。二者一般不同。直接 U 臂与速度臂额外维数、ridge 搜索预算一致。U 交换只从 outer 训练源条件中选供体并保持受体总 U 深度，完整记录匹配；其含义是压力对照。

全套新增 W4 测试 **48/48 通过**（记录：`outputs/w4_verification_v2/pytest.xml`），其中核心 14 项覆盖解析导数有限差分、线性池化权重、全基因归一化、源拟合白名单、评分前 seed 平均、CSR 目标行隔离、ridge 内层 CV 与独立 sklearn 重算的一致性；其余覆盖技术 QC 名单、提取恢复、双区间完整 H1 设计及正式复制的整网格冻结/锁合同。独立 agent 的 ridge 检查最大误差约 7.1e−15。

最终合成实现 fixture：200 次正对照检出率 **100%**；200 次无 U 增量零情形假阳性率 **0.5%**；200 次严格 U=f(观测 S) 压力测试误报率 **0.5%**。使用独立前后细胞抽样。该固定训练/测试 fixture 不等于 RENGE 完整 H1 功效；正式两区间 LOTO 校准仍待最终 QC 队列。

结果：[合成校准](../outputs/w4_synthetic_calibration_v2/RESULTS.md)、[逐次 CSV](../outputs/w4_synthetic_calibration_v2/replicates.csv)。v1 合成因采样/压力定义问题保留审计，由 v2 替代。

完整 H1 设计校准工具 `w4_h1_calibration.py` 也已实现并通过 [smoke](../outputs/w4_h1_calibration_smoke_v2/RESULTS.md)：两区间、TF 共享随机效应、各 outer 折重新拟合源变换、内层 CV、seed 平均后完整交集判据。8 TF × 2 repeats × 3 rho × positive/null 共 12 个合成重复实际运行 6.24 秒；次数不足以判断功效。正式模式要求真实技术 QC 的 H1 TF 名单、每场景至少 200 repeats、10,000 次 TF bootstrap，等 day2 完成后再执行。当前低维模拟下，18–23 TF 的单次实测约 1.9–2.9 秒，完整 1200 重复预计约 1 小时；这不是 PCA50/HVG2000 几何下的运行预算或实测功效。

学习版 no-JVP 速度头、历史实测位移和簇分层速度仍未接入；不把六臂核心称为全部消融已经完成。

## W4-5：正式复制入口已就绪，尚未解封

`w4_replication.py` 已实现 day2→3/day3→4 六臂、四算法 seed 的完整 leave-one-TF 网格，训练 TF 限 H1 共同 QC 名单。先保存所有折的预测/源变换和哈希，再由独立 `evaluate` 入口读 held-out 目标；全局任一折缺失或任一工件变化，均在读目标计数前拒绝评分。纯合成端到端 smoke 已完成 5 TF×2 时间段的 10 折，未读取实际复制集响应。

正式执行要求 `w4_make_lock.py` 生成的冻结锁，绑定冻结 Git 提交、协议字节、QC、原始数据、嵌入、预处理和统计口径；DRAFT 会在计数访问前被拒绝。当前尚未冻结协议或生成正式锁，也尚未启动生物学复制。

`outputs/w4_h1_calibration_wait_job_v1/` 已实际启动，等待 `outputs/w4_cohort_qc_v1/summary.json`。QC 完成后自动执行 200 repeats×2 情形×3 rho 的纯合成完整设计校准，输出到 `outputs/w4_h1_calibration_formal_v1/`。该队列不冻结协议、不读取生物学表达、不自动启动复制模型。

## 外部数据的新增实质进展

scNT-seq 从 GEO 原始 new/old 矩阵构建 25000×37661 的标记计数，再与 Dynamo 官方历史镜像的原始整数值 S/U 做精确 cellname/time/gene 对齐。最终交集为 **13476×37007**，S/U 细胞匹配率 100%，时间冲突 0；5 层计数均非负、有限、整数值。完整数据为 `/data/yuchang/veloroute_w4/scntseq/processed/scntseq_geo_labeling_dynamo_su_exact.h5ad`，SHA256 `208cc0b266d5a6d5bfbb18834f9f153402a7a3c93aaa9c9e4c27e29c96cefbd1`。

这解决了一个分钟级数据集的配套模态获取问题，不是预测阳性。S/U 与标记 RNA 使用不同定量流程，total 不等于 S+U，new 也不等于净速度。13,476 队列来自作者 Ex 输入，上游完整选样仍待核明；本轮未追加响应或 activity-gene 筛选。3060 细胞的作者缩放子集单独保留，没有用它替代原始计数。数据来源与完整技术结果见外部审计和 `outputs/w4_external_scntseq_pairing_v2/`。

## W4-4：day4→5 开发结果

训练为 14 个训练 TF 加 CTRL/AAVS1，留出 LIN28A、NANOG、POU5F1、ZIC3。严格使用 `spliced` 状态层。开发源不参与 HVG/PCA/γ 拟合，开发目标表达在全部预测冻结后由独立入口读取。训练 TF 的目标合法用于响应 ridge 和内层 CV。

| 臂 | 平均 MSE | 平均 MSE 降幅 | TF 等权相对增益 |
|---|---:|---:|---:|
| static | 0.04931724 | — | — |
| vel_pool | 0.04818515 | 2.30% | 3.28% |
| vel_mean | 0.04887694 | 0.89% | 1.32% |
| vel_pool_Uswap | 0.05024460 | −1.88% | −0.78% |
| pseudo_S | 0.04881766 | 1.01% | 1.54% |
| direct_U_pool | 0.04815217 | 2.36% | 2.82% |

MSE 为 PCA-50 的逐维平方误差、先 TF 等权平均；“平均 MSE 降幅”是均值误差之比；右列是先逐 TF 归一化再平均，后者与预注册主统计口径一致。两列不能混称同一百分比。

四个算法 seed 的预测逐项相同：当前臂不消费 k-means 结果、U 匹配无平局，所以 seeds 没有制造独立重复。当前仅四个反复使用过的开发 TF，不对这张表作显著性或正式 H1 判决。

原始预测与 manifest：`outputs/w4_development_predictions_v1/`；当前评分与逐 TF 表：`outputs/w4_development_evaluation_v3/`。v2/v3 评分使用同一冻结预测，新增的是依赖审计、TF 等权口径与 CSV，不改变模型或调参网格。

## 接下来按顺序做

1. day3/day2 定量、技术 QC 完成，登记 H1/H2 共同 TF 列表；补正式两区间合成设计校准。
2. 冻结预注册，记录 Git 提交和输入数据哈希。
3. 在已实现的锁定入口运行 day2→3、day3→4 outer leave-one-TF；先冻结全部预测，再按 TF 等权统计 H1。
4. 再做固定 day2 起点的跨度分析，以及具备相应模态的外部验证。

本轮数据和线性预测以 CPU 为主；GPU0–3 已获授权，待有需要 GPU 的学习头/消融再调度，GPU4–7 未使用。
