# W14-B1 Traxler KO150技术审查与下载

2026-10-18。只调查设计、文件与技术流程；未计算新数据扰动效应、PCA或预测分数。

## 已核实身份

- SuperSeries GSE263763、KO150子系列GSE263761；正确BioProject **PRJNA1099196**。
  新推送草案中的PRJNA1099184不是本次KO150下载入口。
- GEO样本共有27个GSM：9个GEX、9个GDO、9个ADT_HTO；GEX对应18个SRA run。
- A1–A3为未感染，B1–B3为Listeria 6h，C1–C3为24h。数字为建库管/芯片号，
  不能直接叫独立生物重复：作者将3个独立解冻来源做hashtag后混合，须按HTO恢复身份。
- 化学为Chromium Next GEM Single Cell **5′ v2** Dual Indexing，非Flex探针法。
  作者参考为小鼠GRCm38/mm10、Ensembl注释v93。
- 这是已有KO背景下的感染响应时间轴；0h不是尚未接受KO的对照起点。
  后续适合定义为跨KO背景的短时感染响应预测，不自动等于de-novo KO预测。

来源：[GEO SOFT](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE263nnn/GSE263761/soft/GSE263761_family.soft.gz)、
[BioProject](https://www.ncbi.nlm.nih.gov/bioproject/1099196)、
[作者数据入口](https://medical-epigenomics.org/papers/traxler2025/)。

## 已下载与正在下载

- 18个filtered/raw feature H5及feature_reference.csv.gz全部落地，来源、字节和SHA256见
  `outputs/w14b_traxler_acquisition_v2/summary.json`。
- H5同时含Gene Expression、CRISPR Guide Capture、Antibody Capture和Custom类型。
  feature reference含600个guide、11个抗体、9个Custom特征，可先用作者发布feature计数研究技术分配，
  不必把GDO/ADT raw也全部下载。
- ENA的18个GEX FASTQ共151,304,308,989 bytes，但每run仅列单个FASTQ，未据此假定barcode完整。
- 用户已批准直接下载。改走保留technical reads的NCBI SRA，共143,055,563,003 bytes，低于180GB上限。
  使用3个并行传输、断点文件、HEAD字节数锁和SHA256；数据在`/data/yuchang/veloroute_w4/traxler2025/`。
- GRCm38主序列和Ensembl93 GTF下载并行，SRA首个文件完成后自动运行READ_LEN/READ_TYPE小样本审查。

首次下载脚本假定GSE*_RAW.tar而收到404，未影响元数据，失败日志保留；v2根据SOFT列出的19个
真实文件下载成功。没有覆盖任何已有数据。

## 尚待完成的组件

1. 验证SRA中的barcode/UMI与cDNA读段位置；不得照抄RENGE的read索引。
2. 依实际读长和方向建立鼠splici索引、USA定量并与作者barcode集合对齐。
3. 重现作者guide/hash技术分配。作者代码使用guide非零计数判断单guide/多guide，
   再结合HTO doublet/negative和RNA QC；完整阈值与hash映射还需记录，不自造“编辑成功”标签。
4. 只输出每时间×KO细胞数、S/U文库统计和定量对齐结果，仍不读取KO效应。

作者处理入口：[KO150处理代码](https://github.com/epigen/macrophage-regulation/blob/main/src/KO150_01_processing_unsupervised_analysis_all.R.ipynb)。
本次只检查技术代码单元，未使用其中分析输出挑选条件。

当前是“原始数据获取与S/U准备”，不是“S/U已成功”、也不是新数据阳性。

## 首个完整SRA的实际布局

SRR28625696的10个spot均为READ_LEN=[26,92,10,10]：第1段为technical barcode/UMI，
第2段为biological cDNA，其余两段是index。前1万个barcode在作者raw whitelist中，
偏移0精确匹配89.69%，偏移1为0.09%、偏移2为0.04%；偏移0额外4.24%可唯一单编辑校正。
已据此固定CB16+UMI10、第1/2读段和92bp索引，而不是照抄RENGE的第2/3读段。

鼠参考已完整下载；CPU自动处理已进入splici索引构建，之后在每个文库两个SRA均到达时，
按文库联合USA定量，保留S/U/A层。guide先保留provisional调用与原HTO计数，
不把芯片号当生物重复、不把guide检测当作已验证编辑。原始SRA保留，派生FASTQ压缩存储。
实时状态：`outputs/w14_jobs_20261018/traxler_su/status.json`，技术步骤见`outputs/w14b_traxler_su_v1/`。
