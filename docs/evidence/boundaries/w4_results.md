# W4 短时群体预测执行结果

> 归档状态：保留的阴性、边界或混合结果，不因未获阳性而归为错误。 [当前状态](../../current/STATUS.md) · [撤回与更正清单](../../corrections/CLAIMS.md)

更新：2026-10-17。day2/day3 已处理完成；16 TF 预注册已在 `896e99755faf95ca3544a4dceefe92946098bf78` 冻结，day2→3/day3→4 正式 **32 折已完成并评估**。H1 未通过；H7 辅助比较中直接 U 优于当前静态基线。

## 当前判断

正式复制的群体速度 TF 等权相对增益为 **0.70%**，95% CI **[−0.005%, 1.448%]**，Wilcoxon p=**0.0583**；对 U 交换比较也未通过，故 **H1 不成立于本协议**。直接 U 的辅助增益为 **1.82%**，95% CI **[0.99%, 2.73%]**，p=**0.000504**。当前结果支持 U 特征相对当前 S-only/PCA 基线有增量，但速度构造没有保住同等收益。

之前 day4→5 四个开发 TF 的 2.30% 平均 MSE 改善保留在下方历史开发节，不替代正式结果。新 no-JVP 源稳定性头已经训练并评估，尚未优于解析速度；不把这版当作翻盘证据，也不据开发结果重跑选择 H1 模型。下一步优先做表示/估计归因与更细时间尺度验证，router 暂不启用。

## W4-0：原版 GFG 重训机制审计已完成

两数据集各三 seeds、每次 10 epochs 的原生目标域重训均已完成。CPU 复核六个 checkpoint 哈希、保存速度及交换映射，无需再重复训练。

U 置零后，原生 GFG 与真实 U 输入下速度方向的逐细胞中位 cos：RENGE **0.99813–0.99967**，MeRLin **0.99417–0.99830**。解析 moments 参照分别为 **−0.19402**、**−0.07345**。说明当前原生训练配置的输出方向对 U 不敏感；模长仍会变化。这不是 U 没有预测信息的证据。

完整表、变体定义和来源见 [W3 报告](w3_results.md) 与 [W4-0 结果包](../../../outputs/w4_native_gfg_audit_v1/RESULTS.md)。历史原生 GFG 对照包含 JVP；当前 W4 主预测使用简单解析估计器，不调用该 JVP。

## W4-1：数据落地

| 数据 | 实际进度 | 可支持的下一步 |
|---|---|---|
| RENGE day4 同输入重现 | 完成；S/U/A 全矩阵差异条目均为 0，逐细胞总量相关均为 1 | 原流程口径一致，允许推进 day2/day3 |
| RENGE day3 | 完成；6091 细胞、4153 同时通过技术 QC/guide；S/U 中位数 12163/921 | 保持复制效应封存，等待 day2 共同 TF QC |
| RENGE day2 | 已完成；4710 细胞，2726 同时通过技术 QC/guide；S/U 中位数 13588/968 | 与 day3/day4 共同达标 16 TF；正式复制已启动 |
| sci-fate | 已下载 229.72 MB processed，构建 7404×43167 total/new h5ad | total/new 技术对齐完成，基因级 S/U 尚缺 |
| scNT-seq | 已获取并严格配对 13476×37007 的 S/U/new/old/total，0/15/30/60/120 分钟 | 可准备分钟级机制验证；须先冻结外部分析协议 |
| MIX-seq | 原论文、作者 processed 打包代码已核查 | 公开 raw accession 尚未核实，暂不能训练 velocity |

RENGE 原后台队列：`outputs/w4_renge_day23_job_v1/`；day2 首次提取因 `fasterq-dump` 被解析为多调用分派器而退出，修复后 `outputs/w4_renge_day2_job_v2/` 已完成，保留失败日志与 raw。数据根目录 `/data/yuchang/veloroute_w4/renge/`；day2 H5AD SHA256 为 `44b7cb454cb22d574c7630d4c9a3c865d612c68f407a27f2721d8ce85d666aa5`，day3 为 `3bf623c7dc09cbb03ec76c5cdb42a379b45cb28f9f9b83b8ca26507cf5b718a9`。技术清单工具已生成 H1/H2 各 16 TF 的名单，逐日计数见 `outputs/w4_cohort_qc_v1/condition_counts.csv`，协议已按该名单冻结。sci-fate 完整 raw 约 116.28 GB，超过任务书的 50 GB 阈值，本轮未下载。

具体来源、化学布局、哈希、缺失模态与文件路径见 [RENGE 审计](../data_audits/w4_renge_data_audit.md) 和 [外部数据审计](../data_audits/w4_external_data_audit.md)。

## W4-3：可运行的短时核心与校准

已实现六臂：`static`、`vel_pool`、`vel_mean`、`vel_pool_Uswap`、`pseudo_S`、`direct_U_pool`。HVG/PCA/双尾 γ/k-means 只用训练源细胞拟合；所有计数先以全基因 S 深度归一化再取 HVG。目标仍为细胞等权 mean(log1p S) 的 PCA 坐标。

`vel_pool` 先合并 raw counts 再归一化，属于群体动力学特征；`vel_mean` 是固定源归一因子下的逐细胞导数平均。二者一般不同。直接 U 臂与速度臂额外维数、ridge 搜索预算一致。U 交换只从 outer 训练源条件中选供体并保持受体总 U 深度，完整记录匹配；其含义是压力对照。

全套新增 W4 测试 **57/57 通过**（记录：`outputs/w4_verification_v6/pytest.xml`），覆盖解析导数、群体归一化、源拟合、CSR 目标行隔离、ridge 内层 CV、技术 QC、提取恢复、完整 H1/冻结锁、新增 no-JVP 头及其开发端到端读行检查、H7 只读冻结误差且不修改 H1 判决。独立 agent 的 ridge 检查最大误差约 7.1e−15。

最终合成实现 fixture：200 次正对照检出率 **100%**；200 次无 U 增量零情形假阳性率 **0.5%**；200 次严格 U=f(观测 S) 压力测试误报率 **0.5%**。使用独立前后细胞抽样。该固定训练/测试 fixture 不等于 RENGE 完整 H1 功效；正式两区间 LOTO 校准仍待最终 QC 队列。

结果：[合成校准](../../../outputs/w4_synthetic_calibration_v2/RESULTS.md)、[逐次 CSV](../../../outputs/w4_synthetic_calibration_v2/replicates.csv)。v1 合成因采样/压力定义问题保留审计，由 v2 替代。

完整 H1 设计校准工具 `w4_h1_calibration.py` 也已实现并通过 [smoke](../../../outputs/w4_h1_calibration_smoke_v2/RESULTS.md)：两区间、TF 共享随机效应、各 outer 折重新拟合源变换、内层 CV、seed 平均后完整交集判据。8 TF × 2 repeats × 3 rho × positive/null 共 12 个合成重复实际运行 6.24 秒；次数不足以判断功效。正式模式要求真实技术 QC 的 H1 TF 名单、每场景至少 200 repeats、10,000 次 TF bootstrap，等 day2 完成后再执行。当前低维模拟下，18–23 TF 的单次实测约 1.9–2.9 秒，完整 1200 重复预计约 1 小时；这不是 PCA50/HVG2000 几何下的运行预算或实测功效。

学习版 no-JVP 速度头首版已作为独立 day4→5 开发接入，结果见下节；不进入当前 H1 网格。历史实测位移和簇分层速度仍未接入。

## W4-5：正式复制已完成，H1 未通过

`w4_replication.py` 已实现 day2→3/day3→4 六臂、四算法 seed 的完整 leave-one-TF 网格，训练 TF 限 H1 共同 QC 名单。先保存所有折的预测/源变换和哈希，再由独立 `evaluate` 入口读 held-out 目标；全局任一折缺失或任一工件变化，均在读目标计数前拒绝评分。纯合成端到端 smoke 已完成 5 TF×2 时间段的 10 折，未读取实际复制集响应。

正式执行锁为 `outputs/w4_protocol_lock_v1/protocol_lock.json`，绑定冻结提交 `896e99755faf95ca3544a4dceefe92946098bf78`、协议、QC、数据和统计。H1/H2 均为 16 TF；正式 H1 网格为 16×2=32 折，每折六臂×四算法 seed。`outputs/w4_h1_replication_job_v1/` 已完成预测与自动评估：预测耗时约 342 秒，评估约 12 秒。结果目录为 `outputs/w4_h1_predictions_v1/` 和 `outputs/w4_h1_evaluation_v1/`。

| 方法 | day2→3 增益 | day3→4 增益 | 两区间 TF 等权增益 |
|---|---:|---:|---:|
| vel_pool | 1.23% | 0.17% | 0.70% |
| vel_mean | 1.41% | 0.58% | 0.99% |
| vel_pool_Uswap | 0.49% | 0.27% | 0.38% |
| pseudo_S | 1.79% | 0.99% | 1.39% |
| direct_U_pool | 2.34% | 1.30% | 1.82% |

表中均为相对 static、以 static 误差作分母的逐 TF 增益等权平均，不是均值 MSE 之比。两个时间段起点不同，不能用该表声称固定起点的 H2 跨度衰减。

H1 两比较：

- `vel_pool` vs static：0.70%，95% CI [−0.005%, 1.448%]，单侧 p=0.0583；未通过。
- `vel_pool` vs Uswap：0.32%，95% CI [−0.33%, 0.93%]，p=0.1613；day3→4 点估计 −0.11%，未通过。

H7 使用同一冻结误差、无新模型/子集：直接 U vs static 为 1.82%，95% CI [0.99%, 2.73%]，p=0.000504，两时间段均为正；`vel_pool` vs 直接 U 的增益为 −1.12%，95% CI [−1.63%, −0.57%]，方向上直接 U 更好。H7 是辅助证据，不改 H1 判决，也不证明 U 超过任意更丰富的 S-only 表示。结果将后续问题从“完全没有 U 信号”推进为“当前动力学表示为什么不如直接 U”。

正式结果：[H1](../../../outputs/w4_h1_evaluation_v1/RESULTS.md)、[全部 TF/时间段/臂 CSV](../../../outputs/w4_h1_evaluation_v1/metrics.csv)、[H7](../../../outputs/w4_h7_evaluation_v1/RESULTS.md)。

`outputs/w4_h1_calibration_wait_job_v1/` 已完成 16 TF、200 repeats×2 情形×3 rho 的纯合成完整设计校准，输出 `outputs/w4_h1_calibration_formal_v1/`。rho=0/0.5/0.9 的正例均检出200/200；零情形误报分别2/200、1/200、2/200（1%、0.5%、1%）。这是固定低维模拟假设下的设计行为，不是 PCA50/HVG2000 下实际 RENGE 的功效；不改变 H1 未通过的结果。

## 新增：no-JVP 轻量学习头已完成三 seed 开发

代码为 `w4_kinetic_head.py` 与 `w4_learned_head.py`，固定配置见 `configs/w4_nojvp_head_development_v1.json`。原训练源变换不动；同一训练条件内随机把细胞分成两半，S/U 同步拆分，训练一个 50→16→50、1600 参数的残差校准头，标准化坐标下修正范数至多 25%。头只看动力学特征，不读 c/未来目标；冻结后用相同 ridge/CV 预测。直接 U、pooled S 使用同容量头。该稳定性训练不是原版 GFG，也不是独立真实速度监督。

GPU0 三 seeds（0/1/2）、每头 1000 步已完成；全部预测冻结后独立评估：

| 方法 | day4→5 平均 MSE | 相对 static 降幅 |
|---|---:|---:|
| static | 0.04931724 | — |
| 解析 vel_pool | 0.04818515 | 2.30% |
| learned_velocity | 0.04846625 | 1.73% |
| learned_velocity_Uswap | 0.04949827 | −0.37% |
| 直接 U | 0.04815217 | 2.36% |
| learned_direct_U | 0.04809449 | 2.48% |
| learned_S_only | 0.04882341 | 1.00% |

速度头的源半群体稳定性误差在三 seeds 上下降约 17.8%–19.4%，但未来预测 MSE 比解析版高约 0.58%。说明这一版的降噪目标有优化效果，尚未转成更强的未来预测；当前不替换解析主线，不对四个开发 TF 的细小差异做显著性主张。直接 U 学习头小幅改善，继续把 U 信息与动力学构造分开报告。router 保留、未启用。

落盘：[学习头开发结果](../../../outputs/w4_nojvp_head_evaluation_v1/RESULTS.md)、[逐 TF CSV](../../../outputs/w4_nojvp_head_evaluation_v1/metrics.csv)；训练/权重/源侧诊断在 `outputs/w4_nojvp_head_predictions_v1/`。

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

1. 完整设计校准已收齐；保留低维模拟与实际数据功效的区别，不回改已完成 H1 判据。
2. 写 W5 补充计划，区分动力学变换损失、S 表示容量和 U 本身的测量信息；不事后删 TF 或选择重跑的主模型。
3. 优先补齐小时/分钟级外部数据的独立分析规格，再做速度与短时变化的验证；当前不为 RENGE 加大 router。
4. 固定 day2 起点的跨度分析可作为已定义次要分析继续，但不替代 H1。

后续进展：scNT-seq 分钟级首轮与基线场/γ 支持度诊断已完成，见 [W5 结果](w5_scntseq_results.md)。

本轮数据和线性预测以 CPU 为主；GPU0–3 已获授权，待有需要 GPU 的学习头/消融再调度，GPU4–7 未使用。
