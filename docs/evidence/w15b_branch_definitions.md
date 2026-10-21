# W15-B 分支定义清点：阶段性记录

日期：2026-10-21。状态：BR-0 部分完成；用户优先推进 L1/L2/L3 模型线，BR-1 及之后暂停，未运行分支比例分析。

## sci-FATE2

- 本地文件：`/home/yuchang/wangjiaxuan/data/scifate2/raw/GSE236512_processed_data_splicing.h5ad`。
- 只读 H5AD 的 obs/var/obsm 头与元数据：68,866 个细胞，24,967 个基因。
- obs 有 `X_TC_alpha, total_TC_alpha, unspliced_TC_alpha, spliced_TC_alpha, p_e, p_c_TC, sample, timepoint, rep`，没有作者细胞类型/状态列，obsm 为空。
- r1/r2/r3/r4 分别为 16,874 / 17,305 / 16,878 / 17,809 个细胞。
- 作者 [官方仓库 README](https://github.com/rorymaizels/Maizels2023aa) **另有** `celltyped_adata.h5ad` 下载入口。因此不能仅因本地 splicing 文件没有标签就断言作者没有提供注释。原链接 HEAD 重定向至 Dropbox HTML，转换的下载域名本次返回 404；尚未读取该文件 obs 或完成条码对齐。
- 作者官方代码浅克隆保存在 `/home/yuchang/wangjiaxuan/outputs/w15b_branch_audit_v1/author_repository/`，仅作来源核查，不执行作者分析。尚未决定作者标签或训练源 K8 回退方案，尚未拟合分支定义/分类器。

## LARRY

- 文件：`/data/yuchang/veloroute_larry_20261011/larry_invitro_adata_sub_raw.h5ad`。
- 49,302 个细胞，23,420 个基因；day2/day4/day6 分别 4,638 / 14,985 / 29,679 个细胞。
- 原作者 `state_info` 有 11 类：Undifferentiated 23,770，Neutrophil 8,555，Monocyte 8,165，Baso 5,514，Mast 1,414，Meg 1,035，Erythroid 365，Lymphoid 203，Eos 168，Ccr7_DC 64，pDC 49。
- obs 还包括 `Library`（35个文库标识）、`Well`（0/1/2）、`Starting population`（Lin-Kit+Sca1+ / Lin-Kit+Sca1-）。这些列的实验独立性仍需原论文设计确认，不能直接把文库视为生物重复。
- `obsm['X_clone']` 为 49,302 × 5,864 稀疏矩阵。跨时间共同克隆数尚未计算。

## 暂停原因与数学待修正点

尚未运行 BR-1，不报告 G-BR1/G-BR2 通过与否，也未读取 Traxler 表达或扰动效应。

原计划 BR-3 的“冻结 kNN 类别频率保证可微”不成立：普通硬近邻集合在输入空间中分段常数，其类别频率对候选终点几乎处处梯度为零，在近邻边界不连续。该分支比例损失一般只能经 q 影响 router，不能按原文所述通过分类器训练专家终点。BR-3 暂停，需要明确批准可微分类器/软距离核的规格后再实施；本次没有擅自替换。

任何未来比例统计只是群体成分变化，不是观测到的逐细胞命运；相邻/重叠窗口不算独立生物重复。
