# GFG v2（动力学版）实现与验收（2026-10-10）

实现方式：**只新增** `scripts/phase0/w2_gfg_v2.py`，未改 `src/`。复用（import）`GFGCodebook`、`GFGBaseDecoder`。

## 相对当前 GFG 的改动（按任务书 section 4 / D1）

| 项 | 当前 GFG | GFG v2 |
|---|---|---|
| 速率 | 无 | **基因级共享 β_g、γ_g**（log 参数化，正） |
| 动力学损失 | 逐细胞逐基因解 θ（rna_ode_projection） | **‖v_s,tan − (β·u − γ·s)‖² 按计数噪声加权 1/(1+u+s)** |
| 对齐损失 | 有 | 删除 |
| 平滑损失 | pretrain moments-smoother（本项目未用） | 权重 ≤1（未用） |
| 重构 | 标准化空间 MSE | **原始计数 NB 似然**（基因级 dispersion） |
| 速度编码器输入 | (u, s) | **(u, s, u−E[u\|s])**（逼它读 U） |
| 码本 | 软分配 | 软分配 + 使用率监控 |
| 保留 | 双编码器、双码本、decoder JVP 切向投影、逐基因 token + 基因间注意力 | 同 |

## D1 单元测试 ✅

| 测试 | 结果 | 判据 |
|---|---|---|
| JVP == 有限差分 | max\|err\| = **5.0e-4** | <1e-3 ✅ |
| 局部打乱 U 后方向改变 | cos 中位 = **0.665** | <0.9 ✅ |

（随机权重下即通过，说明架构确实读 U。）

## D2 训练与验收（pancreas，2000 HVG，3000 步，8 min）

训练收敛：native 25.8 → **2.33**（recon_nb 1.67，dyn_loss 0.024）。

| 验收 | 指标 | 判据 | 结果 |
|---|---|---|---|
| ① 局部打乱 U 方向 cos | **0.687** | ≤0.9 | ✅（当前 GFG 为 0.9998） |
| ② cos(v, v_kin) > cos(v, mean-shift) | v_kin=−0.089 vs mean-shift=−0.005 | > | **❌** |
| ③ count-split 两半方向 cos | **0.980** | ≥0.8 | ✅ |
| ④ CBDir/ICCoh | — | 参考 | 未算 |

**验收 2/4 通过。**

## 结论与待办

- **主要目标达成**：GFG v2 **确实读 U**（U 打乱 cos 0.56–0.69，远低于当前 GFG 的 0.9998；count-split 稳定 0.98）。解决了 A1/T2 暴露的"GFG 不读 U"能力缺陷。
- **未达标**：accept2——v2 的速度方向尚未与稳态 v_kin 对齐（cos≈−0.09）。

### 根因：动力学损失在 free-rate 下不可识别（重要）

进一步诊断（shared β,γ 重训）：

| 量 | 值 |
|---|---:|
| corr(vs, β·u − γ·s)（per-gene） | 0.995 |
| cos(v, fitted_kin=βu−γs) | 0.693 |
| **cos(fitted_kin, v_kin)** | **−0.033** |
| **模型 shared γ/β** | **1.338** |
| **数据稳态 γ_ss（u/s）中位** | **0.008** |

**问题**：`v_s` 是 decoder 沿 velocity-codebook 方向的 JVP 切向，**不是生物的 dS/dt**。损失 `‖v_s − (β·u − γ·s)‖²` 对任意 `v_s` 都能找到 β,γ 拟合（u,s 每基因张成二维），所以 β,γ 反映的是 JVP 方向，不是稳态。shared 情形下拟合出的 γ/β=1.34 与真实 0.008 相差 100+ 倍。**因此 accept2 在当前规格下无法可靠通过——这是设计层面的可识别性缺陷，不是训练不足。**

### 建议的修正（需 Rinck 确认口径）

1. 把 β_g、γ_g **约束到数据稳态**（复用 `FrozenSplicingTransform.gamma` / 尾部分位回归），只学一个全局时间尺度；
2. 或让 velocity codebook 直接以 **U 创新项 u−E[u|s]** 为切向目标（而非自由 JVP）；
3. 或在损失里加 **cos(v_s, u−γ_ss·s)** 对齐项。

在此之前，按 D2 规则，下游仍应用 v_kin；v2 的能力补齐（读 U）已成立，可作为 C 线/动力学先验的前置。

## 命令

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python scripts/phase0/w2_gfg_v2.py --mode unit-test
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python scripts/phase0/w2_gfg_v2.py --mode train --steps 3000 --batch 64 [--shared-rates]
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python scripts/phase0/w2_gfg_v2.py --mode accept [--shared-rates]
```

产物：`outputs/w2_gfg_v2_unit.json`、`outputs/w2_gfg_v2_accept.json`、`outputs/w2_gfg_v2_pancreas.pt`。
