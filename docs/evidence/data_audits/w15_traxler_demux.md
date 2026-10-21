# Traxler KO150：作者技术注释恢复与 HTO 来源清点

日期：2026-10-21。范围仅为条码、细胞计数、guide 标签、RNA/S/U 文库技术量和 HTO 身份。
没有计算 KO 效应、DE、响应签名或降维，没有训练或评估新数据模型。

## 结果

已从作者 Table S5 恢复 **28,303 个作者 QC 与单 guide 保留细胞**，全部精确匹配现有
76,344 个 USA 细胞的 `library:barcode`。无需重新定量，也不修改原九份 H5AD。

| 时间 | 作者保留细胞 | NonTargeting 细胞 | KO 目标数（不含对照） | 至少 30 细胞的 KO 数 |
|---|---:|---:|---:|---:|
| 0h | 12,176 | 1,289 | 135 | 132 |
| 6h | 11,578 | 1,223 | 135 | 132 |
| 24h | 4,549 | 484 | 135 | 70 |

按总细胞数门槛，0→6h 两端均至少 30 细胞的 KO 为 **132 个**；6→24h、0→24h 为 70 个。
这些只反映计数覆盖，尚未应用严格来源留出、每来源最低细胞数或任何响应强度筛选。

## guide 分配必须采用作者标签

作者处理 notebook 的第 20 个代码单元（0 起始编号）先读取每文库
`crispr_analysis/protospacer_umi_thresholds.csv`，逐 guide 应用 CellRanger 阈值，
并进一步要求 **UMI > 5**。之后才按阈值后非零 guide 数区分单 guide、Negative 和 Multiplet。

旧 USA 导入脚本使用原始 feature 矩阵的非零 guide 数，仅能得到 **9 个**所谓单 guide。
作者保留的 28,303 个细胞全部在原始计数里有不止一个非零 guide；ambient/background guide
不能直接等同多个有效扰动。旧 `guide_call`、`provisional_gene_target` 不可作为正式训练标签。

新派生元数据通过作者条码精确连接，新增 `author_guide_call`、`author_KO_call` 和
`author_post_QC_inclusion`，以作者已经阈值处理的标签和保留集合为准。没有将作者 Mixscape
分类、概率或扰动强度用于筛选，因此不按表达响应挑选“成功 KO”。guide 被分配也不等于编辑被独立验证。

## 作者 QC 可恢复到什么程度

作者处理代码从每文库 `QC_categories.csv` 读取 `hto_demux` 与 `pass_QC`，再保留 HTO 非
Doublet/Negative、RNA QC 合格且 guide 非 Negative/Multiplet 的细胞。
Table S5 的注释源自 `KO150_NORMALIZED_metadata.csv`，可精确恢复这个**最终保留集合**。

目前没有公开定位到原始每文库 `QC_categories.csv` 或 protospacer 阈值 CSV，因此不能声称
已从头复现作者所有原始 RNA QC/HTO 判决。作者 RNA UMI、基因数、线粒体计数比例的原始技术值
已随新元数据保留。S5 头部写作 `Mitochondrial count (fraction)`，但来源为 Seurat
`PercentageFeatureSet` 的 `percent.mt`，新字段明确命名 **`author_mito_percent`**，不再误当 0–1 fraction。

## HTO 与真实来源结构

GEO 实验方案明确说明：细胞来自**三个不同日期解冻的 RAW264.7 独立冻存管**，分别做 hashtag，
随后按 1:1:1 混合到每个处理条件的三个建库管中。不是三个供体；A1–3、B1–3、C1–3 的数字是
tube/chip 的技术批次，不能直接充当独立生物重复。

feature reference 和原矩阵直接给出来源标签：

| 文库 | 处理 | 三个原 HTO feature 的后缀 |
|---|---|---|
| A1–A3 | 0h，untreated | `RAW264.7-a/b/c_untreated` |
| B1–B3 | 6h，Listeria | `RAW264.7-a/b/c_LO28-6h` |
| C1–C3 | 24h，Listeria | `RAW264.7-a/b/c_LO28-24h` |

作者 Table S5 导出代码明确删除了 `hto_demux`、`pass_QC`、HTO 文库统计，因而**作者逐细胞 GMM
来源判决尚未恢复**。为了清点技术可行性，新元数据仅提供固定规则的候选标签：某 HTO 占该细胞
HTO 总 UMI 至少 90% 时记为相应 a/b/c，否则为 `UNRESOLVED`。这是透明的高主导计数标签，
**不是作者 GMM 的复现，也不是已批准的生物重复分样**；`biological_replicate` 仍明确为未解决。

在作者保留细胞内，24,840/28,303（87.8%）得到高主导候选来源，3,463 个未解决：

| 时间 | a | b | c | 未解决 |
|---|---:|---:|---:|---:|
| 0h | 2,988 | 4,131 | 3,683 | 1,374 |
| 6h | 4,567 | 3,253 | 1,852 | 1,906 |
| 24h | 1,447 | 1,359 | 1,560 | 183 |

下一步可获取作者 QC_categories/HTO 原始判决，或在 W15 前单独冻结并技术验证新的 HTO 分样方法。
不能把三个候选标签直接写成“三个独立供体重复”。

## 实际读取范围与来源

- [作者 S5](https://medical-epigenomics.org/papers/traxler2025/results/main_supplement/S05_CROP-seq_large.xlsx)：
  13,201,633 bytes，SHA256 `042c657c23c0bb7a2b67a53f239ae96dd8ebdb76fed5818ce8a20c9a79accc35`。
- 仅解析 `single_cell_annotations` 的 **A:L**：Cell barcode、Time point、Inferred knockout、RNA counts、
  Genes detected、Antibody counts、Antibodies detected、gRNA counts、gRNAs detected、Batch、
  Mitochondrial count (fraction)、Inferred gRNA。M:W 及所有其他工作表数据不读取。
- [作者处理源码](https://github.com/epigen/macrophage-regulation/blob/main/src/KO150_01_processing_unsupervised_analysis_all.R.ipynb)
  SHA256 `48ff4f1eb8e8090bc3b0fdb0f69a8a19291524c6ca2296000283f572968016ea`。
  保存代码单元 3、6、7、13、15、17、20、26、40、44、51，不读取 notebook 输出。
- [作者 S5 导出源码](https://github.com/epigen/macrophage-regulation/blob/main/src/Supplementary_Tables.py.ipynb)
  SHA256 `e7bcd04051d483bcf373ea83d6eac65561b06ac5a7c5a8d10f98cde4c0ef5b6c`；只保留注释导出的单元 58。
- 本地 GEO SOFT SHA256 `d3425444e17ca9647b1739d7896153161587f0c511d50e201b9a2bdeaa63685c`，
  提取实验处理、独立解冻及建库的技术协议。

## 文件位置与可运行入口

原 S/U/A（只读、未修改）：

```text
/data/yuchang/veloroute_w4/traxler2025/processed_usa_v1/{A1,A2,A3,B1,B2,B3,C1,C2,C3}.h5ad
```

作者原附表保留在数据区，不上传仓库：

```text
/data/yuchang/veloroute_w4/traxler2025/author_technical_metadata/S05_CROP-seq_large.xlsx
```

最终派生技术版本（v1 保留为初次恢复记录；v2 补齐源码哈希、线粒体百分数单位和 KO/对照分列）：

```text
/home/yuchang/wangjiaxuan/outputs/w15_traxler_demux_v2/summary.json
/home/yuchang/wangjiaxuan/outputs/w15_traxler_demux_v2/author_processing_evidence.json
/home/yuchang/wangjiaxuan/outputs/w15_traxler_demux_v2/author_annotation_contract.json
/home/yuchang/wangjiaxuan/outputs/w15_traxler_demux_v2/author_technical_annotations.csv.gz
/home/yuchang/wangjiaxuan/outputs/w15_traxler_demux_v2/{A1..C3}_technical_metadata.csv.gz
/home/yuchang/wangjiaxuan/outputs/w15_traxler_demux_v2/time_by_KO_author_QC_cells.csv
/home/yuchang/wangjiaxuan/outputs/w15_traxler_demux_v2/time_source_by_KO_technical_cells.csv
/home/yuchang/wangjiaxuan/outputs/w15_traxler_demux_v2/library_QC_source_counts.csv
```

复现需指定一个新的输出目录，避免覆盖；只用 CPU：

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-gpu/bin/python scripts/phase0/w15_traxler_demux.py \
  --output outputs/w15_traxler_demux_NEW
```

三个合成单元测试通过：条码/文库契约；a/b/c 与技术 chip 区分及未定标签；仅技术工作表 A:L 读取
（即使被屏蔽的 Mixscape 单元含无效 shared-string 索引，也不会解析）。

**当前身份：技术数据与作者保留标签已可用；正式来源分样和 W15 训练协议尚未完成。**
