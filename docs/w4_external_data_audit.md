# W4-1 小时级外部数据：来源与技术落地审计

日期：2026-10-17。范围：技术 QC；预注册仍为草案，未计算任何扰动效应、组间表达差异、PCA、速度或预测指标。

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

## 备用 scNT-seq：仅完成元数据/格式核查

GSE141851 包含 cortical KCl 0、15、30、60、120 分钟，样本 GSM4512669–GSM4512673。0 分钟 processed C/T 文件分别为 2,110,605 与 4,525,428 bytes，表头为 3000 个细胞 barcode；本轮只读取压缩流前四行检查格式，没有下载或分析整矩阵。**GEO 明确 C 是 new、T 是 old（不是 total）；total 应为 C+T。** 这些矩阵已把 exonic/intronic 计入同一基因，仍不能仅由 C/T 宣称拥有 S/U。[GEO 0 分钟样本记录](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4512669)

Dynamo 作者代码另给 neuron splicing/labeling h5ad 镜像，但本机 Stanford 链接 TLS 失败、Figshare 下载返回 403；未下载，也未核实两文件的 barcode 是否可一一对应。仅作为后续轻量另源候选，不据此替换预注册数据集。

## 执行记录

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv/bin/python \
  scripts/phase0/w4_external_data_scifate.py \
  --output outputs/w4_external_scifate_qc_v1
```

本轮 CPU 不超过 4 个数值线程，未占用 GPU。运行 114.50 秒；git 基线 `58f1954bd3cb54f83e1979a8d896efb28da8d0c3`。脚本、配置、六个下载文件及 raw manifest 哈希均随输出记录。以上是可用数据和技术完整性的结论，不是 velocity 的阳性结果。
