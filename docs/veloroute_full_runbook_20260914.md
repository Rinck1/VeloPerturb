# VeloRoute 全量模型与当前实验交接（2026-09-14）

当前状态：用户要求的全量组件已接通，完成一轮真实 ER-short 全量训练、三 seed velocity/router 增量对照，以及一次固定专家后续验证。**工程可训练不等于科学假设成立：目前没有可信的 velocity/router 正增益，全模型未达到相对自身静态回退分支的非劣。**

结果入口：[本轮总报告](../outputs/veloroute_research_progress_20260914/RESULTS.md)。当前授权以 [2026-09-13 修订](veloroute_full_scope_amendment_20260913.md) 为准：允许真实开发侧探索，不等于正式 G1-GO。最终确认 TF 仍封存。

## 1. 已实现范围

| 组件 | 当前实现与边界 |
| --- | --- |
| 状态编码器 | PCA-50 上 identity 初始化的残差去噪器，masked S 重建 + identity 锚；A 后冻结，不能解释成任意可学习坐标变换 |
| 动力学编码器、不确定度 | 源 S/U → r64、v50、正标量 rho；不接受 c；scratch 训练，参考为训练折稳态残差，不是 GFG/VeloVI |
| 条件编码器 | 冻结 ESM2-3B 2560 维 → set-attention 256 维；组合扰动必须提供显式成分映射，已通过合成组合验收 |
| Router / 专家 | top-2 学生路由，768 宽共享骨干、6 个残差块、8 个低秩专家上界；生成时采样一次模式并锁定 |
| 延迟老师 / EM | 老师读训练目标编码；滞后 3 个 E 轮次监督；同条件三方代价、log-Sinkhorn、学生蒸馏；部署丢弃老师 |
| 参考场 / 沿途约束 | 训练源细胞 kNN-30，冻结查询；沿途方向兼容代价只更新指定场模块；不把参考 velocity 直接当表达速率目标 |
| 内源场 | 源侧可学习尺度 × 初始 velocity 单位方向 + 小修正；只是模型分量记账，非唯一生物学归因 |
| 门控 / corruption | 源侧六特征白名单、独立梯度；clean=1 / corrupt=0；三种 corruption；g=0 精确退回独立冻结静态场和条件编码器 |
| 自适应 K / 随机性 | 渐开、按完整训练条件集合审查使用率并剪枝；永不剪空；模式内随机速率每轨迹只采样一次 |
| 训练 / 生成 / 评估 | A/B/C/D、断点恢复、源码快照及哈希；源侧生成先冻结，之后独立读目标并逆 PCA 基因评估 |

默认部署参数 18,541,718，**包含额外的 9,079,730 个冻结静态回退参数**；训练老师另有 241,416 个参数。不能用这张参数表声称 full 与静态场已严格容量匹配。

真实任务仅为同一 KO 的 **day4 → day5 early-response**。不是对照 → KO de novo，也没有完成 day3 中间时间锚或组合扰动真实验证。

## 2. 环境与验收

- CPU `.venv`：Python 3.11 / NumPy 1.26.4 / PyTorch 2.6.0+cpu。
- GPU `.venv-gpu`：项目私有 overlay，以只读共享环境提供的 PyTorch 2.7.1+cu126 为底层，本地固定 NumPy 1.26.4、anndata 0.10.9、SciPy 1.14.1 等。
- 没有修改原 scDFM/CellFlow 环境。GPU 环境不是计划中的 PyTorch 2.6 + CUDA 12.4 认证栈；继承但未使用的其他包存在依赖冲突，不能声称全局 `pip check` 干净。
- 最新 CPU 回归：`outputs/veloroute_full_tests_20260914_v2/junit.xml`，92 passed。
- 完整配置 CPU/GPU 合成文件级验收：`outputs/veloroute_full_smoke_cpu_20260913/`、`outputs/veloroute_full_smoke_gpu_20260913/`。

重新验收（输出目录必须是新目录）：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m pytest tests -q
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/veloroute full-smoke --config configs/veloroute_full.yaml --device cpu --output outputs/full_smoke_cpu_repeat
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv-gpu/bin/veloroute full-smoke --config configs/veloroute_full.yaml --device cuda --output outputs/full_smoke_gpu_repeat
```

新增验收包括：动力学编码器禁 c、梯度隔离、老师滞后、参考库拟合权限、精确静态回退、模式锁定、噪声/位移记账、组合注意力、断点恢复、最终确认输入拒绝，以及所有候选预测冻结后才允许评估。

## 3. 全量训练、恢复、生成

当前数据包和嵌入无需重新构建。一次新的探索性训练：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .venv-gpu/bin/veloroute train-full \
  --source outputs/veloroute_real_pipeline_20260912_v2/fold/train_source.npz \
  --target outputs/veloroute_real_pipeline_20260912_v2/fold/train_target.npz \
  --conditions data/renge/conditions/esm2_3b_v1/conditions.npz \
  --transform outputs/veloroute_real_pipeline_20260912_v2/fold/transform.npz \
  --config configs/veloroute_full.yaml --exploratory-real \
  --output outputs/full_real_repeat
```

预算 A=1,000、B=2,000、C=12,000 步；每 2,000 个 C 步保存完整恢复目录。读取 `status.json` 和 `training_trace.csv`，不能只看 GPU 利用率判断完成。

`--resume <本轮 checkpoints/step_XXXXXXX>` 恢复模型、优化器、老师历史、缓存、随机数与步数；配置/输入哈希必须完全一致，恢复仍使用新的输出目录。不得把探索性 checkpoint 经 resume 晋升为正式训练。

已完成 checkpoint：`outputs/veloroute_full_real_exploratory_20260913_seed0/model.pt`。开发侧预测：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/veloroute predict-full \
  --checkpoint outputs/veloroute_full_real_exploratory_20260913_seed0/model.pt \
  --source outputs/veloroute_real_pipeline_20260912_v2/fold/validation_source.npz \
  --conditions data/renge/conditions/esm2_3b_v1/conditions.npz \
  --transform outputs/veloroute_real_pipeline_20260912_v2/fold/transform.npz \
  --device cuda --output outputs/full_prediction_repeat
```

预测入口没有未来目标参数。全套对照预测冻结后再调用 `evaluate`；门控干预流程由 `scripts/phase0/evaluate_full_development.py` 一次性执行 learned / g=0 / g=1 三臂。

### 时间支持修复，保留旧产物

9 月 13 日的配置沿用原计划 `[2,5]` 门控时间支持，但实际只训练 `[4,5]`。9 月 14 日将默认值改为 `[4,5]`，并强制训练器检查与实际训练区间一致。**这不改变旧模型在 day4→day5 内的预测或已经保存的指标**，但旧 checkpoint 不能用于声称 day2/day3 已在训练支持内。

旧配置/权重/报告原样保留。严格检查会拒绝直接恢复该旧宽区间配置；若今后需要迁移，必须另存迁移记录，不改写历史 checkpoint。当前完成的模型可继续作 day4→day5 源侧推理。

## 4. 已完成实验及下一步

| 实验 | 产物 | 当前含义 |
| --- | --- | --- |
| A/B 增量，7 臂 × 3 seeds | `outputs/veloroute_velocity_exploratory_20260913/` | 当前稳态残差 velocity 与 U 未显示正增量 |
| 最小 router，5 训练臂 × 3 seeds + 简基线 | `outputs/veloroute_router_increment_20260913/` | real 未优于静态/置换；学到了优于 identity 的变化，但输给训练全局均值位移 |
| 全模型，1 seed | `outputs/veloroute_full_real_exploratory_20260913_seed0/` | 15,000 总优化步完成，557 次老师更新，3,000 个 corruption batch |
| 全模型门控干预 | `outputs/veloroute_full_development_evaluation_20260914_seed0/` | full ED=0.52163，静态回退=0.40255，全动态=1.14350；门控缓解退化但没有无损保证 |
| 训练侧质量审计 | `outputs/veloroute_training_quality_20260913/` | velocity 下采样稳定不等于方向正确；稳态相图拟合弱 |
| 训练侧专家检查 | `outputs/veloroute_expert_separation_20260914/` | 联合训练专家差异小、EM 聚合标签近均匀，提示路由教学信号弱 |
| 固定候选专家后路由，4 臂 × 3 seeds | `outputs/veloroute_frozen_candidates_20260914/` | 专家确实分开，real 仍未稳定胜过静态/置换/均匀；只做了一次预定预算跟进 |

每个区间按条件重采样，先平均 seed；仅 4 个开发 TF，精度和泛化证据有限。不同实验族的预算不同，不能把它们横向排行当作严格消融。多次看过的开发集不再是独立确认集。

以下 VeloVI 建议已被用户“必须用 GFG 且联合训练”的新指令覆盖。当前实际入口、权重、四臂和验收顺序见 [GFG 联训执行记录](veloroute_gfg_joint_execution_20260914.md)。上表仍保留旧残差估计器的历史实验，不能当作 GFG 结果。

历史建议（不再执行）：

1. 保留当前固定预算结果，完成训练侧 velocity 估计器审计。现有共享 `scvi-tools==0.20.3` 未包含 VeloVI，独立 `velovi` 包也未安装；不得为赶进度升级共享环境。
2. 按原计划的替代方案，准备一个独立、固定版本的 VeloVI 环境和训练折适配器。先验证源侧推断、基因/细胞对齐、归一化及 velocity 向量投影，再运行；不能把测试源与未来目标拼起来拟合邻居图或估计器。
3. 只做一次事前记录预算的估计器替代对照，其他切分、PCA、条件表示和探针不变；报告 old / alternative / shuffled，而非选择最好估计器后抹掉失败记录。参考不确定度与未来任务价值分开。
4. 替代估计仍无增量，则停止当前 ER-short 上的 router 调参；下一条 velocity 主线应是预定的时间锚/数据验证，并承认当前任务未成功。扩大数据前先冻结协议和成本。
5. 只有形成可信增量才扩展多 seed full、ER-long、de novo 和其他数据集；guide/编辑逃逸/双细胞审计、正式 G1、官方基线指标与最终确认不能被工程测试替代。
