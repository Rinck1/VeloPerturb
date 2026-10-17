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
| RENGE day3 | 两条已缓存 GEX 联合定量运行中 | 完成后仅技术 QC，不读复制效应 |
| RENGE day2 | 约 6.99 GB raw 获取与定量已接入同一串行队列 | day3 后自动接续 |
| sci-fate | 已下载 229.72 MB processed，构建 7404×43167 total/new h5ad | total/new 技术对齐完成，基因级 S/U 尚缺 |
| MIX-seq | 原论文、作者 processed 打包代码已核查 | 公开 raw accession 尚未核实，暂不能训练 velocity |

RENGE 后台队列：`outputs/w4_renge_day23_job_v1/`；数据根目录 `/data/yuchang/veloroute_w4/renge/`。day2/day3 完成后才填最终共同 TF 集并冻结协议。sci-fate 完整 raw 约 116.28 GB，超过任务书的 50 GB 阈值，本轮未下载。

具体来源、化学布局、哈希、缺失模态与文件路径见 [RENGE 审计](w4_renge_data_audit.md) 和 [外部数据审计](w4_external_data_audit.md)。

## W4-3：可运行的短时核心与校准

已实现六臂：`static`、`vel_pool`、`vel_mean`、`vel_pool_Uswap`、`pseudo_S`、`direct_U_pool`。HVG/PCA/双尾 γ/k-means 只用训练源细胞拟合；所有计数先以全基因 S 深度归一化再取 HVG。目标仍为细胞等权 mean(log1p S) 的 PCA 坐标。

`vel_pool` 先合并 raw counts 再归一化，属于群体动力学特征；`vel_mean` 是固定源归一因子下的逐细胞导数平均。二者一般不同。直接 U 臂与速度臂额外维数、ridge 搜索预算一致。U 交换只从 outer 训练源条件中选供体并保持受体总 U 深度，完整记录匹配；其含义是压力对照。

14 项数值/边界测试通过，覆盖解析导数有限差分、线性池化权重、全基因归一化、源拟合白名单、评分前 seed 平均、CSR 目标行隔离、ridge 内层 CV 与独立 sklearn 重算的一致性。独立 agent 的 ridge 检查最大误差约 7.1e−15。

最终合成实现 fixture：200 次正对照检出率 **100%**；200 次无 U 增量零情形假阳性率 **0.5%**；200 次严格 U=f(观测 S) 压力测试误报率 **0.5%**。使用独立前后细胞抽样。该固定训练/测试 fixture 不等于 RENGE 完整 H1 功效；正式两区间 LOTO 校准仍待最终 QC 队列。

结果：[合成校准](../outputs/w4_synthetic_calibration_v2/RESULTS.md)、[逐次 CSV](../outputs/w4_synthetic_calibration_v2/replicates.csv)。v1 合成因采样/压力定义问题保留审计，由 v2 替代。

学习版 no-JVP 速度头、历史实测位移和簇分层速度仍未接入；不把六臂核心称为全部消融已经完成。

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
3. 实现并运行 day2→3、day3→4 outer leave-one-TF；先冻结全部预测，再按 TF 等权统计 H1。
4. 再做固定 day2 起点的跨度分析，以及具备相应模态的外部验证。

本轮数据和线性预测以 CPU 为主；GPU0–3 已获授权，待有需要 GPU 的学习头/消融再调度，GPU4–7 未使用。
