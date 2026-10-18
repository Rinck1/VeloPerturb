# W4-1 小时级外部数据：来源与技术落地审计

> 归档状态：有效数据来源与处理审计；后续任务状态按最新状态页更新。 [当前状态](../../current/STATUS.md) · [撤回与更正清单](../../corrections/CLAIMS.md)

日期：2026-10-17。本文主体记录技术 QC 阶段，当时未计算响应/速度/预测。后续 scNT-seq 分钟级协议已在 `659d76e2063a0daae9ce77de4d7005f420dc267d` 冻结并完成首轮科学分析，结果独立记录于 `docs/evidence/boundaries/w5_scntseq_results.md`；不把技术审计当作预测成绩。

## 已完成：sci-fate total/new 原始矩阵落地

原论文的 Data availability 与 GEO 均确认为 **GSE131351**；A549 对应 **GSM3770930 / SRX5848248**。实验为 DEX 0、2、4、6、8、10 小时，收样前标记 2 小时。GEO 实验步骤明确使用 anchored oligo-dT，不能沿用任务书“随机引物加 oligo-dT”的未核实说法。[原论文](https://www.nature.com/articles/s41587-020-0480-9)、[GEO 样本记录](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM3770930)

本次只下载 A549 的六个 GEO processed 文件：**229,715,904 bytes（229.72 MB）**，而非整套原始 reads。精确 URL/大小在 `configs/w4_external_data_scifate_files.csv`，SHA256 在 `outputs/w4_external_scifate_qc_v1/downloads.json`。下载和构建脚本为 `scripts/phase0/w4_external_data_scifate.py`。

已写入：

- `/data/yuchang/veloroute_w4/scifate/processed/scifate_total_new_only.h5ad`，约 268 MiB。
- 7404 个细胞 × 43167 个基因；layers 仅 `total`、`new`。
- total/new 的 barcode、gene ID 集合和顺序完全一致，匹配率 100%。
- 两层均非负；不存在任何 `new > total` 的矩阵条目。
- 总计数矩阵有 50,057,931 个非零项，new 矩阵 23,539,135 个非零项。
- processed SHA256：`c989ff506e97b8e3c3501557de1943078c941d19df9d168716243aca47231660`。

| 时间 | 原始注释细胞数 | exon UMI 中位数 | intron UMI 中位数 | intron/(exon+intron) 中位数 |
|---|---:|---:|---:|---:|
| 0 h | 1151 | 23109 | 3771 | 0.1375 |
| 2 h | 1189 | 24094 | 4135 | 0.1461 |
| 4 h | 1045 | 22250 | 3571 | 0.1399 |
| 6 h | 1404 | 23857 | 4238 | 0.1507 |
| 8 h | 1169 | 23268 | 4164 | 0.1503 |
| 10 h | 1446 | 22270.5 | 3809.5 | 0.1492 |

上表是 GEO `all_exon`、`all_intron` **细胞级注释**的技术汇总，不是从基因级 S/U 矩阵得出。此次保留全部 GEO 细胞，没有把论文过滤后 6680 细胞的数量当成当前文件的实际数量；过滤规则须独立复现后固定。

## 尚缺：sci-fate 基因级 S/U

GEO 六个文件只有 total/new 矩阵与注释。43167 个基因 ID 中没有 `_intron` 行，不能由细胞级 intron 总计分摊得到基因级 U，更不能把 `new` 伪装成 `unspliced`。作者原始处理代码曾独立计数 intron，但这不表示最终 GEO 发布矩阵保留了这一层。[作者原始计数代码](https://github.com/JunyueC/sci-fate_analysis/blob/master/script_read_processing/sciRNAseq_count.py)

原论文确实另外从单细胞 BAM 用 velocyto 生成过 S/U；本轮未找到其已发布、可直接对齐的下载文件。[论文方法 PDF](https://cole-trapnell-lab.github.io/pdfs/papers/Cao_sci-fate_nbt_2020.pdf)

ENA 对 SRX5848248 的完整清单是 **190 个 run、116,281,104,216 bytes（116.28 GB）压缩 FASTQ**。清单、URL 与 MD5 已保存 `outputs/w4_external_scifate_qc_v1/scifate_raw_runs.tsv`。超过任务书 50 GB/数据集阈值，**没有启动 raw 下载**；若走完整重处理，需要负责人批准容量，并另估解压、参考基因组与 BAM 磁盘量。

可在不扩充权限的情况下继续的工作：查找作者另行公开的 S/U；准备同 barcode 的 U/S 适配器及技术验收测试。冻结协议后，已有 total/new 可支持代谢标记的描述性时间预测，但**还不能完成“内含子速度追回比例”比较**。new 仅是前两小时被检测到的标记 RNA；成为净速度还涉及标记检测率与降解模型，不是无误差真值。

## MIX-seq：processed 可定位，raw 入口未核实

原论文 Data availability 指向作者公布的 Figshare 链接，作者 README 指向 article 10298696。作者的打包脚本只输出 `classifications.csv`、`genes.tsv`、`barcodes.tsv`、`matrix.mtx` 与细胞系特征；没有导出 S/U 或 FASTQ。[原论文](https://www.nature.com/articles/s41467-020-17440-w)、[作者仓库](https://github.com/broadinstitute/mix_seq_ms)、[打包脚本](https://github.com/broadinstitute/mix_seq_ms/blob/master/src/build_figshare_repo.R)

作者公布的 [Figshare share link](https://figshare.com/s/139f64b495dea9d88c70) 网页可确认整体约 2.1 GB processed 数据；本机 article API/网页请求返回 403，尚未取得逐文件 README。403 是当前获取障碍，**不等于已证明 dbGaP 受控访问**。原论文和作者 README 中未找到可核验的 MIX-seq raw accession；不能把文中用作比较的 LINCS GEO accession 当作 MIX-seq 本体。

因此本轮状态为 `RAW_ACCESSION_NOT_VERIFIED`，不排队 velocity 重处理，不申请受控访问，不把仅含表达的数据称为 U/S 数据。下一步需取得公开 raw 的确切 run manifest，随后才能按任务书只选择 trametinib 时间序列。

## scNT-seq：已补齐同细胞 S/U + labeling 的技术配对

GSE141851 的 cortical KCl 数据为 0、15、30、60、120 分钟，样本 GSM4512669–GSM4512673。**GEO 的 C 是 new、T 是 old（不是 total），total=C+T。** 十个原始 processed 文件合计 49,779,872 bytes，现已下载并构建 25,000×37,661 的整数计数 h5ad，各样本 C/T 的细胞、基因集合与顺序均完全相同。[GEO 样本说明](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4512669)、[作者计数管线说明](https://github.com/wulabupenn/scNT-seq)

此前 Stanford TLS 失败、Figshare 403 的阻塞已通过 **Dynamo 官方历史代码中的 Dropbox 镜像**解决，不是找到一套不明来源转载：固定版本 `368dc55e63dc5bbd100d16c566cc5206c0307658` 的 `sample_data.py` 给出两个 URL。splicing 文件 474,450,248 bytes、labeling 文件 142,934,894 bytes，总计 617,385,142 bytes，均已下载并校验 HDF5 与 SHA256。[官方历史下载函数](https://github.com/aristoteleo/dynamo-release/blob/368dc55e63dc5bbd100d16c566cc5206c0307658/dynamo/sample_data.py)

技术配对主产物：

- `/data/yuchang/veloroute_w4/scntseq/processed/scntseq_geo_labeling_dynamo_su_exact.h5ad`，206,841,242 bytes。
- **13,476 细胞 × 37,007 基因**；layers 为 `spliced/unspliced/new/old/total`，X 为 GEO total。
- 全部 13,476 个 S/U barcode 精确匹配 GEO 的完整原始 ID，包含样本、时间、run 和 barcode 字符串；未去前缀、未模糊匹配。S/U 覆盖率 100%，GEO 全队列覆盖率 53.904%。
- 时间前缀与 GEO time 字段冲突 0；双方细胞键、基因键重复 0。
- 只用完全相同基因名的交集：GEO-only 654、S/U-only 7014，均有清单，不补零假装已测量，不进行别名猜测。
- 所有 S/U 原始非零值均有限、非负、为整数值（存储 dtype=float64）；GEO 三层为 int32 非负整数。
- SHA256：`208cc0b266d5a6d5bfbb18834f9f153402a7a3c93aaa9c9e4c27e29c96cefbd1`。

| KCl 时间 | GEO 原始细胞 | 精确配对细胞 | GEO 细胞覆盖 |
|---|---:|---:|---:|
| 0 min | 3000 | 1416 | 47.20% |
| 15 min | 5000 | 2661 | 53.22% |
| 30 min | 5000 | 2881 | 57.62% |
| 60 min | 7000 | 4657 | 66.53% |
| 120 min | 5000 | 1861 | 37.22% |

### 两套计数不能强行视为同一种量

GEO labeling 管线先提取外显子和内含子 reads，再按 T→C 事件区分 labeled/unlabeled 并按基因聚合；Dynamo 的 S/U 是另一套剪接分类定量。配对后的 GEO total 与 S+U 有 **10,314,416 个不相同矩阵条目**，因此 `total = new + old` 成立，`total = spliced + unspliced` **不成立**。后续不得在两套定量间直接相减解释为净速度，必须明确各自测量模型和归一化口径。[作者处理代码说明](https://github.com/wulabupenn/scNT-seq)

原实验每个时间点都在收样前标记 2 小时；KCl 暴露时长从 0 到 120 分钟变化。`new` 反映一个标记窗口，不是瞬时转录导数，也不是已测得的净 RNA velocity。[原论文与 Extended Data Fig. 2](https://www.nature.com/articles/s41592-020-0935-4)

### 作者筛选子集独立存档，不作为原始主队列

历史 labeling h5ad 只有 3060×24078。其 `new` 非零值中 1,853,078/1,853,696 为非整数，`total` 为 7,840,021/7,843,397，证明它不是未经变换的整数原始计数；具体缩放步骤未由该文件元数据唯一确定。3060 细胞全部匹配 S/U，基因交集 23918，但本次配对产物**完全没有使用这个文件的表达值或选样名单**。

官方教程明确先取高质量细胞，并聚焦对 KCl 响应较强的兴奋性神经元；教程后续还按 `activity_genes` 筛基因。该标签是响应相关预选，不等价于训练期独立 HVG 选择。我们未读取该字段数值，不使用其基因名单、响应效应、UMAP、拟合速度或预测结果。[官方教程的载入与基因筛选段](https://dynamo-release.readthedocs.io/en/latest/tutorials/notebooks/300_scNT_seq_readthedocs.html)

13,476 个 S/U 细胞的文件仅带 `cellname`，本轮保留其全部细胞，没有新增响应、细胞类型或 HVG 过滤。进一步只读核对原作者 notebook 源代码，确认其由 **`0411_Ex_` 前缀的 Spliced/Unspliced 输入文件**构建，并写为 `neuron_splicing_4_11.h5ad`（零基 cell 4、6）；这提供了 Ex 上游队列的来源证据，不仅是与论文细胞数相似的猜测。源代码随后才另加 Ex 标签过滤和 >1900 detected genes 门槛（cell 13），我们没有复制这些后续筛选；最初 `0411_Ex_` 文件的 Ex 判定与完整 QC 仍未重建。因此当前产物是“作者 Ex 命名 S/U 输入队列与 GEO 的精确交集”，不是已证实无作者选择偏差的全集。[原作者预处理 notebook](https://github.com/wulabupenn/scNT-seq/blob/master/notebook_for_figures/neuron_revision_figures.ipynb)

此来源补充晚于配对文件生成，独立保存为 `outputs/w4_external_scntseq_pairing_v2/upstream_selection_audit.json`；未改写已冻结 h5ad 的计数或哈希。h5ad 内“上游选择尚未知”的简短说明应结合本补充解读：Ex 命名输入已核实，完整 Ex/QC 判据仍未知。

### 技术落地时的科学建议（历史；现已另行冻结并执行首轮）

以下建议形成于技术配对完成时；随后按 `PLANS/w5_prereg_addendum.md` 固定细则，已实际运行 30→60、60→120 分钟预测。首轮 velocity 未胜 identity；后验基线场中心化出现方向线索，不能替代首轮结果。完整数值见 `docs/evidence/boundaries/w5_scntseq_results.md`。

- 只有五个时间点，应优先做机制描述或 rolling-origin 预测；不训练高维转移矩阵，不把大量细胞/基因当独立实验重复。
- 候选方案是在 0 min 基线估计 U/S 的 γ 后冻结；在当前 KCl 响应源时刻重新拟合稳态 γ，可能把群体反应吸收入基线。该选择需在看效应前写入协议。
- 0 min 是未刺激基线，不应要求其瞬时速度预知随后外加 KCl 的全部效应。0→15 min 可作方向描述；优先把 15→30 min 的已启动响应区间用于尺度校准，再滚动预测 30→60、60→120。若用几乎为零的基线场去拟合刺激起始位移，尺度可能被噪声放大；不能据此判定动力学无用。identity、历史 S 漂移、直接 U、new proxy 应同预算比较。只有两个后续预测区间，只作机制描述，不宣称独立大样本显著性。
- 新 RNA 是标记代理，不是无误差净速度真值；上述任何对比目前均未运行。

### 复现与产物

下载配置：`configs/w4_external_data_scntseq_files.csv`、`configs/w4_external_data_dynamo_files.csv`。脚本：`w4_external_data_scntseq.py`、`w4_external_data_dynamo.py`、`w4_external_data_scntseq_align.py`（均位于 `scripts/phase0/`）。

正式技术输出：`outputs/w4_external_scntseq_qc_v1/`、`outputs/w4_external_dynamo_schema_v2/`、`outputs/w4_external_scntseq_pairing_v2/`。包含精确 URL、字节数、输入/脚本 SHA256、barcode/gene 对照表、技术 QC CSV 和 summary JSON。下载 schema v1 因 NumPy 整数的 JSON 序列化中断；配对 v1 因原始 barcode 含字母 N 而停止；两者保留为中断审计，不作为完成结果，修正后未模糊修改任何 ID。

## 其他现有数据在短时主线中的角色

| 数据 | 现在能回答什么 | 不能承担什么 |
|---|---|---|
| MeRLin | Day0→Day21 的克隆关联/长时命运边界；已有源侧计数与作者标签覆盖审计 | 不是小时级连续时间系列，不能充当短时预测复现；作者 clone 分组不是独立测得的克隆内命运比例 |
| Kang | 对照→IFN-β 6 h 的 donor-held-out 响应预测、可靠性/负对照 | 只有对照和 6 h 终点，不能直接估计多预测时距衰减或“已开始响应→下一时刻”的滚动预测 |
| PerturbSci-Kinetics | 跨遗传条件的 whole/nascent 增量、合成/降解机制边界 | 主屏为 CRISPRi 7 天后末尾标记 2 h，不是 2 h 后的未来表达目标；不能用作多未来时点短时主检验 |

MeRLin 的旧“缺 Day0”说法已经过期：最新 `docs/evidence/boundaries/w3_results.md` 记录源侧 QC 后 247 克隆、3785 细胞；作者 Day21 metadata 的获取也已完成。严格源/目标各≥3细胞且标签一致只有 20 克隆，这限制的是当前标签预测样本规模，不能靠合并标签宣称解决。

Kang 的设计依据为 [GSE96583](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96583)；文献中的响应异质性不等于我们已完成的多峰检验。PerturbSci-Kinetics 的主屏明确选择较长 CRISPRi 窗口以接近稳态；论文另有部分单基因 pulse-chase 验证，不能混同为完整 pooled screen 的多时点目标。[PerturbSci-Kinetics 原论文](https://www.nature.com/articles/s41587-023-01948-9)

本次不新增这三套数据的训练或生物学分析。scNT-seq 的模态配对已进入独立分钟级分析；其首轮结果与后验诊断见 W5 报告，不是已建立 velocity 的预测阳性。

## 执行记录

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python \
  scripts/phase0/w4_external_data_scifate.py \
  --output outputs/w4_external_scifate_qc_v1
```

本轮 CPU 不超过 4 个数值线程，未占用 GPU。运行 114.50 秒；git 基线 `58f1954bd3cb54f83e1979a8d896efb28da8d0c3`。脚本、配置、六个下载文件及 raw manifest 哈希均随输出记录。以上是可用数据和技术完整性的结论，不是 velocity 的阳性结果。
