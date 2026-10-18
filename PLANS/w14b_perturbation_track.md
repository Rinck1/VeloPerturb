# W14-B 扰动线：新数据封存、RENGE 探索性迁移、MeRLin 采样核查

日期：2026-10-17　|　规格：Claude　|　负责人：Rinck　|　执行：服务器 agent
与 `PLANS/w14_confirmation_loro.md`（W14 主任务）并行，但 **W14 优先**：GPU 先给 W14，本文件任务以 CPU 为主。

2026-10-18执行登记：W14已启动，B3的32折/576作业已按独立协议准备并排队。
用户已明确批准“直接下载”；KO150的18个GEX SRA共143.1GB，下载上限180GB，不再重复请求50GB批准。
正确BioProject、9个GEX与27个GSM的对应、技术读段布局及进度见
`docs/evidence/data_audits/w14b_traxler_audit.md`。新数据仅做技术处理，不提前训练W15。

---

## 0. 目的

项目的最终目标是扰动预测。目前的扰动证据只有 RENGE 的 H7（直接 U 增益，预注册确认）。sci-FATE2 上的"动力学决定群体模式配比"机制若在 W14 确认，需要在**新的多扰动时间序列**上做正式确认。本文件为此准备数据，并在已用过的数据上做一次探索性检查。

路径：sci-FATE2 确认机制（W14）→ RENGE 探索（W14-B3）→ Traxler 正式确认扰动泛化（W15，W14 出结果后另写预注册）。

---

## 1. 通用规则

- 沿用 W4 任务书第 1 节全部规则：只新增、不改 `src/`、记录来源 URL/文件名/大小/SHA256、原始数据放 `/data/yuchang/veloroute_w4/<dataset>/`、不进仓库。
- 单个数据集下载总量 **超过 50 GB 先问 Rinck**。
- 脚本 `scripts/phase0/w14b_*.py`，输出 `outputs/w14b_<task>_v1/`，结果写 `docs/w14b_results.md`。
- 完成一项提交并推送一项，提交信息带任务编号。

---

## W14-B1 Traxler 2025 巨噬细胞 CROP-seq：核实、下载、封存（最高优先级）

**数据**：Traxler et al., *Cell Systems* 2025，"Integrated time-series analysis and high-content CRISPR screening delineate the dynamics of macrophage immune regulation"。
- GEO SuperSeries GSE263763；KO150 子系列 **GSE263761**；SRA BioProject **PRJNA1099196**（10月18日来源核实更正，原草案写作PRJNA1099184）。
- 已核实（GEO 页面）：135 个转录因子与染色质调控因子，每个 4 条 gRNA，约 10% 非靶向 gRNA；李斯特菌感染，时间点 0、6、24 小时；CROP-seq + CITE-seq；GSE263761 含 9 个样本（CROPseq_KO150 A1–A3、B1–B3、C1–C3）的 filtered/raw h5 矩阵。
- 代码：https://github.com/epigen/macrophage-regulation ；项目页：https://medical-epigenomics.org/papers/traxler2025/

**步骤**

1. **核实（不下载原始读段）**，产出 `docs/w14b_traxler_audit.md`：
   - 10x 化学版本（3′ v3 / 5′ / 其他）。能否得到内含子读段决定 U 是否可用；若是探针法或靶向测序，**停止并汇报**。
   - 9 个样本与时间点、重复、技术批次的对应关系（从 GEO GSM 页面、SRA 元数据、作者代码确认，不靠猜测）。
   - SRA 每个 run 的类型（GEX / HTO / GDO / ADT）与大小；GEX 读段总量。
   - 作者处理流程中的 gRNA 分配、细胞过滤标准。
2. **下载**：先下 GEO 的 h5 计数矩阵（小）；GEX 原始读段总量 ≤ 50 GB 直接下载，超过则先问 Rinck。
3. **构建 S/U**：用 STARsolo（`--soloFeatures Gene Velocyto`）或与 RENGE 处理一致的流程，从 GEX 读段得到 spliced/unspliced/ambiguous 三层；细胞条形码与作者 h5 对齐，报告匹配率。用与作者一致的参考基因组版本（GEO 平台为小鼠）。
4. **gRNA 分配**：用作者的 GDO 数据和流程得到每个细胞的敲除身份；只保留单一 gRNA 目标的细胞。
5. **只做技术 QC**，产出 `outputs/w14b_traxler_qc_v1/`：
   - 每个时间点 × 每个敲除基因的细胞数表（非靶向对照单列）；
   - S、U 文库大小分布，U/S 比按时间点的分布；
   - 与作者 h5 的 S 计数一致性（相关系数、差异条目比例）。

**封存（最重要）**：在 W15 预注册冻结之前，**不得**计算任何敲除相对对照的效应、不得做跨条件比较、不得画按敲除或按时间着色的降维图、不得训练任何预测模型。QC 表只能包含计数类技术量。

**汇报**：化学版本、样本—时间映射、可用敲除数（每个时间点至少 30 个细胞的敲除基因数）、S/U 构建是否成功、下载大小与哈希。

---

## W14-B2 斑马鱼胚胎扰动图谱：只核实，不下载

**数据**：Saunders et al., *Nature* 2023，"Embryo-scale reverse genetics at single-cell resolution"。23 个基因 F0 敲除，18、24、36、48、72 hpf，单细胞核 sci-RNA-seq3，约 2.7M 细胞，对照注射胚胎 159 个。

产出 `docs/w14b_zebrafish_audit.md`：
- GEO / SRA 编号与原始读段总量；
- 作者处理对象是否已含 spliced/unspliced；若无，sci-RNA-seq3 读段能否用 STARsolo 或作者流程拆出内含子读段；
- 每个时间点 × 每个敲除的细胞数（只读作者元数据，不读表达）；
- 单细胞核数据中 U 占比偏高的问题：作者是否讨论，是否有已发表的速度分析先例。

不下载原始读段，不读表达矩阵。

---

## W14-B3 RENGE 探索性迁移（只跑一次，标注"探索性"）

**问题**：sci-FATE2 上的分布路由机制，在扰动数据上有没有苗头？

RENGE 所有时间点都已被评估过，本任务**只能作为探索性分析**，不改变任何已有判决，不作为确认证据。

**先写协议再运行**：`PLANS/w14b_renge_router_protocol.md`，提交后才能运行，内容必须包括：
- 模型：W13 b1 端点配方（与 W14 相同的 K8、结构、学习率、中心化、距离核），**不调参**；训练步数固定为 12,000（RENGE 训练群体少，不沿用 36k；在协议中写明理由）。
- 数据：沿用 W4 的 16 TF 名单、HVG2000、PCA50、正则 γ、源侧拟合规则；均值分支用 W4 `vel_pool` 的冻结流程。
- 任务：留一 TF，day2→3 与 day3→4 两个时间段，预测留出 TF 的 t+1 天分布。
- 臂：`static`、`real`、`shuffled_v`（3 个置换种子）、`direct_U`；各 3 个训练 seed。
- 指标：能量距离（固定 512/512 子样本），TF 等权；相对 static 与相对 shuffled_v 的改善。
- 不做显著性检验结论；只报告方向、逐 TF 与逐时间段数值。

**运行**：协议提交后运行一次。若因 bug 需要重跑，修复内容写入协议的偏离记录，不得因结果不理想重跑。

**汇报**：`docs/w14b_results.md` 中单列一节，标题标"探索性"。

---

## W14-B4 MeRLin 采样时间核查（只看元数据）

产出 `docs/w14b_merlin_sampling.md`：
- 列出 MeRLin 的全部采样时间点和相邻间隔；
- 每个时间点的细胞数与克隆数；
- 是否存在小时级间隔。

只读元数据，不做任何分析。若存在小时级间隔，由 Rinck 决定是否另写群体层面的探索性协议；若全为天级，记录为"时间尺度不支持"，不再投入。

---

## 2. 顺序与资源

| 任务 | 资源 | 预计 | 依赖 |
|---|---|---|---|
| W14（主任务） | GPU 1–3 | 1–2 天 | — |
| W14-B1 核实 | CPU | 0.5 天 | 无，立即开始 |
| W14-B1 下载与 S/U 构建 | CPU + 磁盘 | 1–2 天 | 核实完成；>50 GB 先问 |
| W14-B2 | CPU | 0.5 天 | 无 |
| W14-B3 | GPU（W14 空闲时）| 0.5 天 | 协议提交 |
| W14-B4 | CPU | 1 小时 | 无 |

---

## 3. 本轮不做

- 不在 Traxler 上训练或评估任何模型（留给 W15 预注册）；
- 不下载斑马鱼原始数据；
- 不在 RENGE 上做第二次探索；
- 不使用 Kang 2018、Norman 2019（单时间点设计，不支持动力学预测）、Sheu 2024（靶向测序板，无内含子读段）、人原代 CD4 T 细胞全基因组 Perturb-seq（10x Flex 探针，无 U）。
