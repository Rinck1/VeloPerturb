# VeloRoute

详细实验过程、正负结果、统计口径与声明降级边界见 [实验总账](docs/veloroute_experiment_ledger_20260914.md)；对应 [审计包](outputs/veloroute_experiment_ledger_20260914/RESULTS.md) 保留 21 次真实训练的权威路径、哈希和原精度 CSV。

RNA velocity 条件化的源侧 router + 共享骨干/低秩专家，用于异质扰动预测。

当前主线（2026-09-14）为 **真实 GFG decoder-JVP 与 router 联合训练**，不是 VeloVI 或旧残差估计器。123 项回归和实际 GFG 全宽、全组件 GPU 短验收通过；未配对群体组成合成对照通过。真实 pilot 的 12 次训练及唯一一次直接 router 监督跟进的 9 次训练均已完成，**未建立真实 velocity/router 增量，暂缓长时全量训练**。见 [GFG 联训执行记录](docs/veloroute_gfg_joint_execution_20260914.md)。工程跑通不是项目成功。

最新数字见 [GFG 研究进度 RESULTS](outputs/veloroute_gfg_joint_progress_20260914/RESULTS.md)，旧估计器的 [历史研究进度](outputs/veloroute_research_progress_20260914/RESULTS.md) 不覆盖。用户已允许正式 G1 前真实开发侧探索，见 [9 月 13 日授权修订](docs/veloroute_full_scope_amendment_20260913.md)；正式 G1 未运行，最终 5 个确认 TF 封存。

资源硬限制：本项目只允许物理 GPU 0–3，GPU 4–7 留给其他任务；已有其他项目进程不由本项目停止。[day3 四条原始文库](outputs/veloroute_raw_day3_20260914/RESULTS.md)已完成 CPU 下载、校验、提取和压缩；下一步为联合 USA 与 guide/QC，S/U 尚未完成。Kang/McFarland 等候选仅完成文献与元数据评估。

## 最小对照结构

```text
冻结潜状态 z₀ ─┬─→ 源侧 router q(k | z₀, v₀, e_c) ─→ 采样一次 k ─┐
冻结源速度 v₀ ─┤                                               │
条件嵌入 e_c ──┘                                               ▼
z(t), t, e_c ─→ 共享骨干 + K 个低秩专家 ─→ 固定使用 F_k ─→ RK4 ─→ z₁
```

- v₀ 只进入 router，以及可选的**训练期**模式兼容代价；专家场不直接读取 velocity。
- 网络输入是已对齐的潜状态/速度和条件嵌入。真实数据已接入训练 TF 内拟合的稳态残差 velocity 基线（不是 GFG/VeloVI）和 23×2560 冻结 ESM2-3B 条件嵌入；前者的动力学有效性尚待诊断。
- 网络类默认状态 50 维、速度 50 维、条件 256 维、2 个专家；文件训练入口按数据包自动配置维度，实际 ESM2 输入为 2560 维。各臂使用同一条件表示与网络预算。
- 低秩下投影随机初始化、上投影零初始化。训练采用逐专家责任度加权误差，不对专家平均场做回归。
- 场损失与路由交叉熵有显式 detach 边界；生成时专家索引全程锁定。
- 此最小对照不含内源场、完整 EM、门控等；这些组件已在独立的全量实现中接通。任何版本均不保证恢复逐细胞真实命运。

## 全量结构

`full_model.py` / `full_training.py` 实现可训练状态编码器、S/U 动力学编码器与不确定度、组合条件 set-attention、top-2 router、共享骨干与 K≤8 专家、训练期延迟老师/EM、训练源侧参考场与沿途约束、内源场、可靠度门控和 corruption、自适应 K、模式内随机性、锁定模式的 RK4、checkpoint/断点恢复。

状态编码器是原 PCA 坐标上的残差去噪器，A 后冻结。历史默认后端是 scratch 动力学编码器/稳态残差；新 `dynamics_backend=gfg, joint_dynamics=true` 后端加载本地 MouseBrain GFG RNA 分支，实际 JVP 速度接 router，并保留原生损失联合更新。推理读取源侧基因 U/S，因此不能称 velocity “仅训练监督”。独立冻结静态场和条件编码器保证 g=0 精确回退，但学习门控在真实数据上的无损性没有成立。

历史默认部署参数 18,541,718；当前 GFG 全宽为 20,021,704，包含冻结回退参数；老师另有 241,416 个训练期参数。真实任务目前只有 ER-short day4→day5，不是 de novo 或四时间点全任务。

## 安装与验收

CPU 独立环境为 `.venv`（NumPy 1.26.4 / PyTorch 2.6.0+cpu）。GPU 为项目私有 `.venv-gpu` overlay，继承只读共享 PyTorch 2.7.1+cu126 并本地固定 NumPy/anndata 等；没有修改已有环境。实际 GPU 训练与全量合成验收通过，但这不是原计划的 PyTorch 2.6 + CUDA 12.4 认证栈；继承的未用软件存在依赖冲突，不宣称全局 `pip check` 干净。

重新创建时：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-framework.txt
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --no-deps -e .
```

常用命令（每次结果使用新目录，禁止覆盖已完成实验）：

```bash
.venv/bin/veloroute status
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m pytest -q
.venv/bin/veloroute core-smoke --output outputs/core_smoke_new
.venv/bin/veloroute synthetic --output outputs/core_synthetic_new
.venv/bin/veloroute pipeline-smoke --output outputs/file_pipeline_new
.venv/bin/veloroute full-smoke --config configs/veloroute_full.yaml --device cpu --output outputs/full_file_pipeline_new
.venv/bin/veloroute workflow --config configs/veloroute_pipeline.yaml --output outputs/real_pipeline_new
```

`core-smoke` 验证实际配置维度上的责任度、反向传播、ODE 和 checkpoint 恢复。`synthetic` 对四类已知真值系统按固定预算训练，输出 CSV、checkpoint、预测、验收检查和 provenance。合成真实模式/配对只用于工程测试，不算 G1-GO。

`pipeline-smoke` 验证最小三臂文件闭环，`full-smoke` 验证全量组件及组合条件。`workflow` 的正式路径仍受 G1 门槛控制；真实探索通过单列的 `train-full --exploratory-real`、`velocity-increment`、`router-increment`、`frozen-candidates` 入口运行，不能自动晋升为 G1-GO。

## 最小网络调用

```python
import torch
from veloroute.model import ModelConfig, VeloRoute

model = VeloRoute(ModelConfig()).eval()
z0 = torch.zeros(8, 50)
v0 = torch.zeros(8, 50)          # 示例占位张量，不是实际 RNA velocity
condition = torch.zeros(8, 256)
prediction = model.predict(z0, v0, condition, t0=4, t1=5)
assert prediction.endpoint.shape == (8, 50)
```

上例网络未训练，只演示接口。实际生成需要经过验证的冻结变换、训练好的 checkpoint，以及任务允许的源侧输入；不接收未来表达或目标标签。

## 代码分工

| 文件 | 已实现 |
| --- | --- |
| `model.py` | Router、物理时间编码、共享场/专家、损失隔离、模式锁定 RK4、安全权重恢复 |
| `coupling.py` | 同条件三方代价、k-dependent 方向兼容、log-Sinkhorn、责任度；非真实命运证据 |
| `contracts.py` | 源侧输入白名单、训练细胞权限、冻结 PCA、分层置换 |
| `metrics.py` | 明确定义的 energy V-statistic、先平均 seed 再按条件 bootstrap |
| `protocol.py` | 预注册完整性/输入哈希检查、真实训练 G1 门槛 |
| `splits.py` | 不读取响应值的 TF 预留切分；拒绝将未判定 guide 当正式条件 |
| `raw.py` | 文库分离的任务清单、带日志的外部命令、FASTQ 配对/布局检查 |
| `synthetic.py` | 合成专家预训练→冻结专家→real/static 路由训练→生成/干预/恢复验收 |
| `preprocess.py` / `download.py` | 可校验分段下载、逐 run 审计、样本联合 USA、S/U/A 导入与保守 QC/guide 筛选 |
| `latent.py` / `protein_conditions.py` | 训练折内 S/U 变换及独立数据包；公开蛋白 ESM2 冻结编码 |
| `experiments.py` | 未配对训练、源侧预测、冻结哈希、独立潜空间/基因空间评价 |
| `probes.py` / `workflow.py` | 固定预算 A/B 探针与正式 G1 门槛、数据到实验总入口 |
| `pipeline_smoke.py` | 原始合成 S/U 到三臂文件级闭环验收 |
| `full_components.py` / `full_model.py` | 全量编码器、参考场、内源场、门控、自适应模式及生成 |
| `full_training.py` / `full_experiments.py` | A/B/C/D、延迟老师/EM、恢复、显式真实探索与独立预测 |
| `full_smoke.py` | 全量配置、合成 S/U 与组合条件的文件级验收 |
| `router_increment.py` / `frozen_candidates.py` | 固定预算匹配对照、固定专家后的独立路由训练与冻结评估 |
| `quality_audit.py` | 训练侧相图、下采样稳定性、guide 一致性描述，不授予正式 QC |
| `gfg.py` / `gfg_experiments.py` | 实际 GFG 双流/VQ/JVP、联合路由梯度、训练源归一化与基因级输入闭环 |
| `gfg_synthetic.py` / `multimodal.py` | 未配对群体组分合成检验、前后编码置换、训练侧多峰候选筛查 |

## 数据处理的重要发现

day4/day5 的全部 8 条 run 已下载、SRA 校验、提取并无损压缩；每条 run 独立检查 `_2/_3` 前 100 万对（不足则全量）布局，gzip 完整性检查覆盖全部文件。`_1` 是 8 bp 样本索引，`_2` 是 26 bp barcode/UMI 读段，`_3` 是 91 bp 文库序列。**不能默认 `_1/_2` 就是所需配对。**

对 `_2/_3` 的完整 2,325,824 对记录扫描，barcode 从第一个碱基开始的发布 barcode 匹配率约 88.26%，右移一位只有约 0.036%。偶见首位 `N` 不是固定要剪掉的前缀。详细输出见 `outputs/veloroute_raw_day4/layout_SRR21518305/`。该匹配不等于已批准 UMI/chemistry 或 S/U 质量。

当前 day4/day5 分别有 7,582/6,910 颗细胞、60,668 个基因；U UMI 中位数分别 675/836。两个时间点均以同一样本的两条 GEX run 联合定量。最终计数位于 `data/renge/processed_release_v1/`，源/目标数据包位于 `outputs/veloroute_real_pipeline_20260912_v2/fold/`。

未完成：gRNA/编辑逃逸/双细胞正式审计、可靠 velocity 的独立有效性证据、正式效应阈值与 G1 冻结、GFG 多 seed 全模型严格匹配消融、day2/day3、其他数据集、官方 CellFlow 15 指标与最终确认。GFG 联训适配和用户要求的全量组件已经实现，不能再列为“未实现”，也不能据此宣称项目成功。

工具依据：[NCBI 官方 SRA 下载](https://github.com/ncbi/sra-tools/wiki/01.-Downloading-SRA-Toolkit)、[technical-read 提取说明](https://github.com/ncbi/sra-tools/blob/master/README.md)、[PyTorch 历史版本安装](https://pytorch.org/get-started/previous-versions/)。
