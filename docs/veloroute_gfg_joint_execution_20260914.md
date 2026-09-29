# GFG 联训执行记录与验收边界

补充详细记录：[实验总账](veloroute_experiment_ledger_20260914.md)，包含合成阳性、两轮真实结果、恢复链、声明边界与候选数据；[审计包](../outputs/veloroute_experiment_ledger_20260914/RESULTS.md) 提供机器可读数值和 21 次训练索引。

## 当前主线

执行用户的新要求：使用真实 GFG RNA 双流与 decoder JVP 估计速度，让 router 的任务损失联合更新 GFG。不是把旧动力学 MLP 改名，不再把 VeloVI 作为下一步主线。源侧 U/S 在推理时仍然需要，因此方法不是“velocity 仅在训练时存在”。

顺序：实际权重/数值验收 → 训练侧多峰筛查 → 未配对优势场景与负对照 → 真实四臂三 seed → 审查增量与数值问题 → 全量 GFG 联训。所有真实任务暂限 RENGE 同 KO day4→day5；day2/day3、de novo 尚未落地。最终 5 个确认 TF 不参与这些步骤。

## 本轮资产与实现

- 执行配置：`configs/veloroute_gfg_joint_20260914.yaml`。此前 `veloroute_gfg_prerequisites_20260914.yaml` 指向的 14.64M corrected pilot 是另一候选架构，并非本轮实际加载的权重；没有按下游分数比较/挑选它们。
- 实际原实现：`/data/yuchang/GFG/model/`；实际初始化：`results/mousebrain_graphbatch_directed_v1_seed0/final.pth`，SHA256 `a9d9fa39063312230e4ba64309e9264ba3c2cb7ac549f04544b366df57c1340b`。固定使用 seed0，理由是可严格加载且能与原 RNA 分支逐数值验收；这是本地 MouseBrain 初始化，不是经过认证的官方 foundation release。
- `gfg.py` 保留双编码器、双软 VQ、共享基因解码器及 JVP。实际速度经 `vS/(1+S)` 和冻结 PCA 旋转进入 router；不以旧稳态残差作为 velocity 目标。真实输入是 2,000 个训练折选定基因的 U/S，不是 PCA-100。
- RNA 输入按源细胞 S 文库归一化、不取 log；均值/尺度仅由训练源细胞拟合。不同于预训练的邻居平滑 moments，本轮不做 S 邻居平滑，以免提前抹掉隐藏状态；这一域迁移仍需实验证据。
- 任务梯度通过 `router ← representation(v) ← decoder JVP ← GFG`。保留原生重构/VQ/RNA ODE 损失。`c` 没有进入 GFG 前向接口；条件相关的任务梯度允许更新 GFG。参考场、E-step、老师标签和门控特征仍 detach。
- 与原训练有明确差异：禁用 dropout，确保重构与 JVP 描述同一确定函数；RNA ODE 线性解使用 float64，残差损失按 RNA 量纲缩放。原逐细胞/逐基因 2 方程、3 速率参数的约束欠定，不能用低残差证明速度正确。
- `rho` 学习源 RNA 重构误差代理，**不是**已校准的真实 velocity 误差或贝叶斯后验不确定度。门控识别 corruption 也不保证下游无损。

## 必须区分的优势检验

1. 表达完全相同、隐藏动力学组分比例变化：静态路由不能知道新群体的组分比例，GFG 有可能提供这个信息。但是群体内部打乱已估计速度保留其多重集合，边际预测原则上可不变；这里不能要求 real 一定胜过 shuffled。
2. S 与 U 的对应关系决定组分比例：先经 GFG 融合 S/U，再打乱输出，仍可能保留被编码的分支比例。检验第二模态来源，需要**进入 GFG 前打乱 U，S 保持不变**。原后编码置换实验保留，不删除；新增前编码 U 置换保持相同 seed、步数和阈值，并有数学不变性/生成机制测试。
3. 静态位置决定分支：静态臂应能完成；单峰 K=1 是输出结构负对照，不可冒充自动关门或自适应 K 的证据。合成实验的目标原型只由训练终态 KMeans 产生，逐细胞真值命运只用于评价。它检验的是 GFG/router 组分学习，不是整个 ODE 专家系统的科学验收。

这些 toy 是 U/S 编码的隐藏状态/群体组成系统，并非经过生物学标定的 RNA 反应仿真。本轮未包含等容量的原始 U 直接输入臂，因此只能证明 GFG 输出可以承载有用的第二模态信息，不能证明 GFG 速度优于任何 S+U 编码器；这仍需后续机制消融。

## 已完成

- 原实现一致性：重构最大绝对差 `4.17e-7`，JVP 速度 `3.58e-7`；见 `outputs/veloroute_gfg_reference_parity_20260914/`。
- GFG 基因输入：训练源 2,733、开发源 1,133；未处理确认输入，见 `outputs/veloroute_gfg_inputs_20260914/`。
- 训练 TF 多峰筛查：14 个条件中 0 个达到固定稳定性门槛，Gaussian sanity check 0/20。它不是正式 D4/FDR，也不能据此断言生物学单峰；但目前没有可预先冻结的强多峰优势子集。guide/编辑逃逸/细胞周期混杂仍待审计。
- 全组件短验收：20,021,704 个部署参数，8 个活跃专家、13 次延迟老师更新、4 个 corruption batch，实际 GFG task gradient 非零，随机预测有限、checkpoint 重载一致。见 `outputs/veloroute_gfg_full_smoke_20260914/`。这些是工程阳性，不是预测增益。
- 首轮未配对合成：same-S 的比例绝对误差 static `0.3000`、joint GFG `0.0616`，三 seed；冻结 GFG 为 `0.0636`。说明这种场景可利用动力学，但尚未证明联训优于冻结。后编码置换未退化，因此当轮完整优势检查未通过。
- 前编码 U 置换补充实验完成：interaction 的比例误差 static `0.3000`、joint `0.06334`、shuffled-U `0.29221`，固定判据通过。输出位于 `outputs/veloroute_gfg_synthetic_inputU_recovered_20260914/`；中断后复用 40 个已保存模型，只补训 8 个单峰负对照，旧中断产物保留。
- 实际恢复验收：从真实 GFG C200 恢复到 C400，与未中断版本的 207 个模型张量和完整 trace 完全一致。见 `outputs/veloroute_gfg_actual_resume_verified_20260914/`。完整回归已达 123 项（`outputs/veloroute_gfg_all_tests_20260914_v5/`）。
- 真实 pilot：static / frozen GFG / joint GFG / joint GFG + 条件内局部 U 置换，各 3 seed；A100+B300+C400，K=2，关闭内源/门控/噪声来隔离基础比较。真实置换在 condition × S-depth × 局部 S 块内进行，与合成全群体置换不是同一干预。
- 所有 12 份预测全部冻结后已统一评价。指标为 energy distance、方差误差、去共享变化的条件中心化基因均值误差；不事后选择有利 TF。冻结 GFG 臂在 C 期不再更新原生损失，这与 joint 有预算差异，比较须单独标注。中断恢复模型以各臂 `selection.json` 为准，不默认原 `train/` 完整。

## 两轮真实结果与当前判决

| 开发实验 | static 分布误差 | joint GFG | U 置换 | 结论 |
|---|---:|---:|---:|---|
| 四臂三 seed pilot，sampled energy | 0.338830 | 0.342245 | 0.342388 | 未建立增量 |
| 固定专家直接 router 监督，exact mixture energy | 0.337255 | 0.337222 | 0.337224 | 差异极小，区间跨零 |

两行是不同预测器和指标口径，不能直接用行间降低证明进步。Pilot 的微小条件中心化均值改善未通过真实对置换的检验，也没有分布/结构收益支持。第二轮使用同 seed 静态 pilot 的同一组冻结专家候选，不按分数挑初始化；3 臂 × 3 seed × 600 步，六项校正比较全部区间跨零。详见 `outputs/veloroute_gfg_joint_progress_20260914/RESULTS.md`。

判决：工程 PASS、指定合成组分检验 PASS、真实增量 NOT_ESTABLISHED；**A300/B2000/C12000 全量暂缓**，不是正式 G1-KILL 或 G1-GO。停止在这四个反复使用的 ER-short 开发 TF 上继续换损失/加 seed 追阳性，原始阴性全部保留。当前联合 GFG 的任务梯度确实存在，但非零梯度、非零 U 敏感性均不等于有用的 velocity 信息。

## 资源与下一步

- 用户要求只用四卡：实际 GFG 入口强制校验 `CUDA_VISIBLE_DEVICES`，仅物理 GPU 0–3 或其完整 UUID 合法；GPU 4–7 禁用。共享 scDFM 等已有进程不属于本项目，不停止它们，不能承诺 4–7 在整机上实际空闲。
- 后台启动器 `scripts/phase0/launch_gpu_job.py` 记录 PID、GPU UUID、日志和退出码；`--cpu` 清空 CUDA 可见设备，原始数据不占 GPU。
- day2/day3 发布表达矩阵已在只读数据目录，但未有成品 S/U。每一天固定两条 mRNA 和两条 gRNA，分别处理，不按文件大小判文库。
- 系统盘余量不足以同时展开两天原始数据；已获工具权限建立专属目录 `/data/yuchang/veloroute_timeanchors_20260914/`，不移动或删除既有文件。day3 四条文库的 CPU 串行下载、SRA 校验、提取和保留原件压缩已全部完成，后台 exit code 0；S/U 尚未定量。状态见 `outputs/veloroute_jobs_20260914/day3_raw/`。
- 原始获取完成后再做逐 run 布局、联合 USA、guide/QC、训练折内时间锚协议；不提前将下载完成称作 S/U 完成。新增时间点是额外时间约束，不是独立供体复现或新的确认集。最终 5 TF 始终封存。

## 全量训练的边界

当前固定全量预算仍是 A300+B2000+C12000，未启动。历史配置的 `final_full_entry` 只代表“代码可跑”，本轮执行判决已明确暂缓，不构成自动启动指令或科学 GO。任何有依据的修复另开结果目录；先前实验、阈值和阴性记录不可改写。

全量启动时另写明确判决记录，区分“探索性联合模型跑通”和“具有真实 velocity/router 增益的科学实验”。正式 G1、官方 CellFlow 15 指标、独立确认和多时间任务仍是后续工作。

## 入口

以下为专用入口参考，不是当前待执行队列；full/resume 须先解除上述暂缓判决。所有 CUDA 命令必须显式设置四卡限制，例如：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
.venv-gpu/bin/python scripts/phase0/run_gfg_joint.py pilot --arm gfg_joint --seed 0 --output outputs/gfg_pilot_new
.venv/bin/python scripts/phase0/evaluate_gfg_pilot.py --output outputs/gfg_evaluation_new
.venv-gpu/bin/python scripts/phase0/run_gfg_joint.py full --seed 0 --output outputs/gfg_full_new
.venv-gpu/bin/python scripts/phase0/resume_gfg_joint.py --checkpoint-directory outputs/gfg_full_new/checkpoints/step_0002000 --output outputs/gfg_full_resumed_new
```

每次使用新目录。`run_gfg_joint.py` 是基因 U/S 的专用入口；旧 `train-full` 默认仍是历史 baseline 后端，不能拿它的记录证明用了 GFG。
