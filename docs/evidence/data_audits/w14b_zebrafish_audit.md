# W14-B2 ZSCAPE斑马鱼扰动图谱入口

2026-10-18。本轮只读来源、设计和元数据，不下载表达矩阵或原始测序，不计算扰动效应。

- GEO **GSE202639**，BioProject **PRJNA836866**。
- 遗传扰动部分的名义采样为18/24/36/48/72hpf，最短相邻间隔6h；不同基因不一定覆盖全部时点。
- ENA全项目222个run，压缩FASTQ合计1,394,267,790,088 bytes，包含参考与扰动实验，
  不是已单独筛选出的扰动GEX大小。本轮不下载这些原始读段。
- 作者发布Monocle3 cds、raw count RDS和独立cell/gene metadata。仅凭文件名无法断言RDS含S/U；
  当前没有建立可直接使用的S/U入口。
- sci-RNA-seq3为组合条形码流程，不能直接套10x化学；作者给出的bbi-dmux/bbi-sci处理链
  应作为进一步检查内含子分离可行性的入口。单核U富集也不能直接当作净速度质量。
- 作者cell metadata明确含`timepoint`、`gene_target`、`embryo`；已排队仅从这三列统计
  每时间×靶基因细胞数及胚胎数，不读取表达、UMAP、mean_nn_time等列。

来源：[原论文Data/Code availability](https://www.nature.com/articles/s41586-023-06720-2)、
[作者入口](https://cole-trapnell-lab.github.io/zscape/)、
[GEO发布清单](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE202nnn/GSE202639/suppl/)。

元数据与run清单：`outputs/w14b_metadata_audits_v1/zscape_inventory.json`；
技术分组计数作业：`outputs/w14b_zscape_counts_v2/`。计数完成状态以其summary/status为准。
v1在HTTPS握手时EOF失败且未下载矩阵；保留日志，v2加有界重试，不改变计数规则。

定位：规模大、有明确遗传扰动与时间设计的备选，原始数据重处理成本高；优先完成Traxler，不抢占本轮GPU。
