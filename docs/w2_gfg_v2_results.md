# GFG v2（动力学版，no-JVP）实现与验收（2026-10-10）

实现方式：**只新增** `scripts/phase0/w2_gfg_v2.py`，未改 `src/`。复用（import）`GFGCodebook`、`GFGBaseDecoder`。

## 设计（相对当前 GFG）

| 项 | 当前 GFG | GFG v2 |
|---|---|---|
| 速度定义 | decoder **JVP 切向投影** | **直接动力学：v = β_g·u − γ_g·s（不用 JVP）** |
| 速率 | 无 | 基因级 **γ_g**（log 参数化），**β_g 冻结为 1**，γ_g 锚定到数据稳态 u/s |
| 重构 | 标准化空间 MSE | **原始计数 NB 似然**（基因级 dispersion） |
| 对齐/平滑损失 | 有 | 删除 |
| 表示 | 双编码器 | manifold 编码器 + 软 VQ 码本 + decoder（用于 NB 重构与表示） |
| 读 U | ✗（U 置零 cos=1.0） | **✓**（U 打乱 cos 0.28） |

**关键改动（本轮）**：去掉 JVP。上一版用 decoder-JVP 作速度，损失 `‖v_s,tangent − (βu−γs)‖²` 对任意 JVP 都能解出 β,γ（每基因 u,s 张成二维），**不可识别**（模型拟合出的 γ/β=1.34 vs 数据稳态 0.008）。改为**速度直接等于 β_g u − γ_g s**，β=1，γ 由数据稳态锚定 → 速度良定义、读 U、且与 v_kin 对齐。

## D1 单元测试 ✅

| 测试 | 结果 | 判据 |
|---|---|---|
| 速度 == β·u − γ·s | max\|err\| = **0.0** | <1e-4 ✅ |
| 局部打乱 U 后方向改变 | cos 中位 = **0.535** | <0.9 ✅ |

## D2 训练与验收（pancreas，2000 HVG，6000 步，~8 min）

| 验收 | 指标 | 判据 | 结果 |
|---|---|---|---|
| ① 局部打乱 U 方向 cos | **0.278** | ≤0.9 | ✅ |
| ② cos(v, v_kin) > cos(v, mean-shift) | **0.474** vs −0.009 | > | ✅ |
| ③ count-split 两半方向 cos | **0.847** | ≥0.8 | ✅ |
| ④ CBDir / ICCoh | — | 参考 | 未算 |

**验收 3/3 通过**（accept4 仅参考）。

## 结论

- **GFG v2 通过 D2 全部验收**：读 U（0.278，对比当前 GFG 的 0.9998）、方向对齐 v_kin（0.474 ≫ mean-shift）、count-split 稳定（0.847）。
- 代价：速度现在是**解析动力学函数**（不再由网络 JVP 产生），网络主要负责 NB 重构与表示。若需要"网络产生的动力学速度"，应改用 β_g/γ_g **约束到数据稳态 + 可学全局尺度**，并把 U 创新项作为速度头的显式目标（本轮未做）。

## 命令

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python scripts/phase0/w2_gfg_v2.py --mode unit-test
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python scripts/phase0/w2_gfg_v2.py --mode train --steps 6000 --batch 64
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python scripts/phase0/w2_gfg_v2.py --mode accept
```

产物：`outputs/w2_gfg_v2_unit.json`、`outputs/w2_gfg_v2_accept.json`、`outputs/w2_gfg_v2_pancreas.pt`。
