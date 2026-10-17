# sci-FATE2：独立时间系列的 S/U 数据入口

日期：2026-10-17。本轮只做公开来源、下载、层结构与技术元数据核查，不读取生物效应，不训练预测模型。

## 为什么换这个入口

2020 sci-fate 的现有 GEO total/new 不能替代内含子 S/U，已核查的 Dynamo 示例 `scifate()` 仍是占位函数，不构成 S/U 下载入口。[作者原始 sci-fate 仓库](https://github.com/JunyueC/sci-fate_analysis)、[Dynamo 数据加载代码](https://github.com/aristoteleo/dynamo-release/blob/master/dynamo/sample_data.py)

新候选是 Maizels 等的 sci-FATE2，论文为 *Reconstructing developmental trajectories using latent dynamical systems and time-resolved transcriptomics*，作者仓库明确给出总系列 GSE236520。不是同一个实验改名，也不是 scNT-seq 的重复分析。[原论文](https://www.sciencedirect.com/science/article/pii/S2405471224001194)、[作者仓库](https://github.com/rorymaizels/Maizels2023aa)

## 已核实的元数据

- 实验子系列为 **GSE236512**；mESC HM1 神经管分化，D0 FGF、D2 FGF+CHIR、D3–8 RA+SAG。不是大量独立 KO 条件，不能承担未见扰动泛化结论。[GSM7548957](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7548957)
- GSM7548957/958/959 标为三个重复，采样 D3/4/5/6/7/8。重复独立性仍需结合作者元数据确认，不能把时点直接算独立重复。[GEO 总系列](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE236520)
- GSM7548960 标为第四重复，GEO 写 D3.2/3.4/3.6/3.8，名义间隔0.2天约4.8小时；**作者构建脚本实际标签为05h/10h/15h/20h**，因此落地后须核对真实时间单位，不擅自选择4.8或5小时。[GSM7548960](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7548960)、[作者构建脚本](https://github.com/rorymaizels/Maizels2023aa/blob/main/analysis/A0.0_making_adata_sets.ipynb)
- 标记为2小时、500µM 4sU；该窗口不等于未来时间间隔或净速度真值。[GSM7548960](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7548960)
- 作者构建脚本显式输出仅保留 `spliced`/`unspliced` 两层的文件，X用 spliced，不是从 new/old 强行改名。[构建脚本](https://github.com/rorymaizels/Maizels2023aa/blob/main/analysis/A0.0_making_adata_sets.ipynb)

## 官方小型下载可行

官方目录 `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE236nnn/GSE236512/suppl/` 有 counting、estimate、splicing 三个发布文件。本次只获取：

`GSE236512_processed_data_splicing.h5ad.gz`

URL：`https://ftp.ncbi.nlm.nih.gov/geo/series/GSE236nnn/GSE236512/suppl/GSE236512_processed_data_splicing.h5ad.gz`

HEAD核实压缩大小 **1,306,780,035 bytes**，Last-Modified为2023-07-05；无需下载116GB的原 sci-fate raw。下载配置 `configs/w5_scifate2_technical_v1.json` 限1.5GB，解压限8GB，另保留4GB磁盘余量、总时限2小时。目标为本仓库 `data/scifate2/raw/`，不覆盖现有数据。

## 执行状态与剩余检查

**最新实际状态：下载、解压、兼容读取均完成，最终摘要为 `outputs/w5_scifate2_technical_v3/summary.json`。** 原矩阵68,866×24,967，真实S/U两层，细胞和基因键唯一。解压后SHA256为 `4269f26b5d53d436c692210d965233ed66f2d3d4a6ff0e1c0a6713b108816944`。v2随后因HDF5的np.int64 shape无法JSON序列化失败，已修为Python int；无须重复下载。旧v1/v2状态保留作历史。

用户已要求减少追加归因并直接扩大。已启动三重复日级训练→独立小时级系列的模型适配；使用实际05h/10h/15h/20h标签的5小时差，不依赖GEO取整日期。计划和结果见 `PLANS/w5_joint_kinetics_expansion.md`、`docs/w5_joint_kinetics_results.md`。以下原排队说明仅为获取过程记录。

后续已完成真实训练：三版本×三seed，解析速度版在独立r4系列相对static平均降MSE11.615%，各段−0.922%、+34.492%、+7.555%。该科学结果另记 `outputs/w5_joint_kinetics_scifate2_evaluation_v1/`；本文件原技术审查不是科学成绩。

下载已通过 CPU job 启动，记录 `outputs/w5_scifate2_download_job_v1/`。只检查 S/U 层头部、细胞/基因ID、rep/timepoint分组；实际完成以 `outputs/w5_scifate2_technical_v2/summary.json` 为准，不能把已启动下载写成已经可训练。

执行者注意：初版后台进程已加载的 `anndata.io.read_elem` 不兼容当前环境；下载/解压不依赖它，但到元数据阶段会报导入错误。代码已改为项目既有的 `anndata._io.specs.read_elem`，并安排 `outputs/w5_scifate2_inspection_job_v2/` 等旧进程终止后只读验收已下载、CRC解压验证的文件。不要重复下载或覆盖完整文件。若下载本身失败，则后继作业也会明确失败，保留partial而不冒充完成。

下载边界、技术层语义和现有科学链合计 **74项测试通过**，记录 `outputs/w5_U_attribution_verification_v2/pytest.xml`。这不是新数据的科学实验通过。

后续技术验收：原始数值非负/整数支持、S/U/X关系、细胞键唯一性、rep4时间映射、条件/技术批次、源/目标每组细胞量。之后才固定预测协议。先前 scNT-seq 已得分，但 sci-FATE2 的生物效应尚未用于选模型；复制时不挑有利时点或亚群。
