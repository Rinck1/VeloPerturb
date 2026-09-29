# VeloRoute 管线操作与交接（2026-09-12）

> 历史交接文档，以下状态截至 9 月 12 日。全量组件及真实探索现已完成一轮，最新操作与边界见 [9 月 14 日全量手册](veloroute_full_runbook_20260914.md) 和 [总报告](../outputs/veloroute_research_progress_20260914/RESULTS.md)。旧实验目录保持不变。

本轮目标是修订计划的 ER-short：同一扰动的 day4 源 S/U → day5 分布，不是 de novo。数据处理已完成；最小训练/生成/评估接口在合成数据上通过。正式真实数据实验停在质量审计与预注册节点。

## 1. 已完成的输入

| 项目 | 路径 | 已验证内容 |
| --- | --- | --- |
| day4/day5 原始数据 | `data/renge/raw/` | 全部 8 条 run；SRA 校验、technical reads 提取、gzip 完整性与逐 run 布局检查 |
| GENCODE v32 splici | `data/renge/reference/gencode32/splici_r91/index/` | 官方 MD5 校验、r91、k31、5′ rc 定量方向 |
| 同样本联合 USA | `data/renge/quant/day4/`、`day5/` | 两条 GEX run 一起做 UMI 解析，不逐 run 计数相加 |
| 最终计数 | `data/renge/processed_release_v1/day4/day4.h5ad`、`day5/day5.h5ad` | S/U/A 分开；X=S+A；保留原始 guide counts、QC 和角色 |
| 冻结变换及分包 | `outputs/veloroute_real_pipeline_20260912_v2/fold/` | 仅训练 TF 的 day4/day5 拟合；source/target/target_genes 分包 |
| 冻结条件 | `data/renge/conditions/esm2_3b_v1/conditions.npz` | 真正 ESM2-3B 23×2560；公开蛋白输入，不读取细胞表达 |

最终计数附完整 USA 矩阵、barcode、gene、发布 feature 计数及条件清单的输入哈希。新运行附源码快照、环境、seed 和产物哈希；旧失败/中断产物保留，不覆盖为成功。

## 2. 直接复验当前工程闭环

工作目录为 `/home/yuchang/wangjiaxuan`。每次给不同输出目录；已有目录会被拒绝。以下命令不授予正式训练许可。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m pytest tests -q
.venv/bin/python -m pip check
.venv/bin/veloroute pipeline-smoke --output outputs/file_pipeline_repeat
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/veloroute workflow --config configs/veloroute_pipeline.yaml --output outputs/data_workflow_repeat
```

最后一条对已完成的真实计数执行工程级训练折拟合并汇总数据接口，没有协议与 G1-GO 时返回 `ENGINEERING_PIPELINE_READY_FOR_PROTOCOL_REVIEW`。这不是 G1 阳性，不会启动真实 router 训练。

已完成的验收：

- `outputs/veloroute_pipeline_tests_20260912_final/junit.xml`：67 passed。
- `outputs/veloroute_file_pipeline_smoke_20260912_final/`：合成 real/static/shuffled 三臂闭环。
- `outputs/veloroute_real_pipeline_20260912_v2/`：真实数据准备完成，到审核节点停止。
- `outputs/veloroute_pipeline_delivery_20260912/`：交付报告、QC CSV、契约检查、provenance。

## 3. 数据与模型当前的确切含义

- S/U/A 是 USA 定量的三个独立层；表达对象 X=S+A，但最小任务状态、velocity 和基因评估使用 strict S，不把 A 当 U。
- 先按 TF 预留角色，再只对技术 QC 与 guide 筛选通过的训练 TF 拟合。控制类别 AAVS1/CTRL 分开保留，本轮同 KO 的 ER-short 训练包不混入控制细胞。
- 状态：按源 S 总计数作每细胞归一化；训练 S 的方差排序选 2,000 基因，PCA-50。S-only 不通过归一化分母间接读取 U。
- velocity：训练折 upper-tail OLS 的 `U−γS`，经 log1p 链式法则和 PCA 旋转，再按训练尺度标准化。它是简单稳态残差基线，不是 GFG/VeloVI，不具备已校准的天级速度或已验证方向。
- ESM2：GENCODE v32 每 TF 最长蛋白；末层残基均值，不含特殊 token。超过 1022 aa 分窗、长度加权；不是 canonical UniProt 全长上下文表示。模型使用完整 2560 维，固定低成本 G1 探针另使用预声明的随机投影，各臂一致。
- 训练损失不把 velocity 当表达向量场目标，场不读取 velocity；源侧 router 推理时读取 velocity。不能称“velocity 仅训练监督”。
- 模式是统计响应分量，不是有独立真值的逐细胞命运。静态/置换臂移除或置换所有 velocity 消费路径。

## 4. 正式诊断前的剩余步骤

1. 完成 guide 一致性、编辑逃逸/双细胞、S/U 覆盖与动力学相图/稳定性审计，只用允许的训练/开发信息。当前标签为发布 feature UMI 上的保守算法筛选，不是编辑真值。
2. 确定正式细胞清单和估计器；经有记录的批准发布新的数据版本，再用 `prepare-fold --kind development` 生成批准的数据包。不能直接改 `formal_ready` 来代替审计。
3. 在本轮效果产生前，补齐 `configs/veloroute_g1.draft.yaml` 的实际增益、非劣效幅度及独立依据，绑定正式 split/data/velocity/probe 文件。当前 engineering 数据包不能直接冒充 development。
4. 冻结协议后运行四臂 A/B；B0 与 A0 共用，所以实现为 7 臂×3 seeds。所有验证预测先保存并哈希，才读取 day5 验证目标。最终确认 TF 不打开效果。
5. 数值检查通过只返回 `REVIEW_REQUIRED`；结构、扰动特异性、动力学解释与统计功效审核通过后，才能记录 G1-GO。4 个验证 TF 的条件级统计功效有限，增加 seed 不增加条件数。

冻结后的命令形式如下。路径是下一轮已批准文件的占位位置，不是当前可跳过审核直接执行的命令：

```bash
.venv/bin/veloroute freeze-protocol --config configs/veloroute_g1.approved.yaml --output outputs/g1_protocol_approved
.venv/bin/veloroute g1 --help
.venv/bin/veloroute train-unpaired --help
```

正式延续 `workflow` 时，新配置必须同时指定 `frozen_protocol`、`g1_decision`、`frozen_fold`。`frozen_fold` 必须是 G1 使用的同一批准目录；训练验证其源/目标/条件哈希，不允许重拟合后悄悄替换。

## 5. 生成与评估的隔离

```bash
.venv/bin/veloroute predict --checkpoint PATH_TO_MODEL --source PATH_TO_SOURCE_PACK --conditions PATH_TO_CONDITIONS --transform PATH_TO_FROZEN_TRANSFORM --output outputs/prediction_new
.venv/bin/veloroute evaluate --predictions outputs/prediction_new --target PATH_TO_TARGET_PACK --target-genes PATH_TO_TARGET_GENE_PACK --output outputs/evaluation_new
```

预测命令没有 target 参数；它先生成并冻结潜空间、逆 PCA 基因空间和模式选择。独立评估检查预测哈希、角色、时间、条件覆盖及源/目标 ID 不重叠，再读取未来目标。

当前指标为明确实现的 energy V-statistic、均值误差、variance ratio、训练选基因的均值误差/R²及条件中心化均值误差；后者不是声称复现完整 Systema。官方 CellFlow 15 指标尚未接入。

## 6. 不属于本次完成声明

完整 v1.0 的 GFG/VeloVI、复杂 EM/老师、内源场、可靠度门控、自适应 K、官方 15 指标、多数据集、day2/day3、ER-long/de novo 和确认测试均未被本次工程验收替代。当前默认 K=2；无“门关闭无损”或“router 已有真实增益”的结果。

工程环境为 `.venv` 的 NumPy 1.26.4 / torch 2.6.0+cpu；定量使用项目私有 simpleaf 0.30.0、alevin-fry 0.18.2。ESM 编码一次性调用已有 scDFM GPU 环境（torch 2.7.1+cu126）和项目私有 fair-esm，不修改共享环境；这不代表正式训练栈已经认证。

官方依据：[GENCODE v32](https://www.gencodegenes.org/human/release_32.html)、[simpleaf chemistry](https://simpleaf.readthedocs.io/en/latest/chemistry-command.html)、[ESM](https://github.com/facebookresearch/esm)、[NCBI SRA](https://github.com/ncbi/sra-tools)。
