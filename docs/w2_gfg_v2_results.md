# GFG v2（动力学版，no-JVP）实现与验收（2026-10-10）

实现方式：**只新增** `scripts/phase0/w2_gfg_v2.py`，未改 `src/`。复用（import）`GFGCodebook`、`GFGBaseDecoder`。

## 设计（相对当前 GFG）

| 项 | 当前 GFG | GFG v2 |
|---|---|---|
| 速度定义 | decoder **JVP 切向投影** | **网络速度头**（线性，读 u,s,u-innovation） |
| 速率约束 | 无 | 基因级 **γ_g**（log 参数化，锚定数据稳态 u/s），**β_g=1 冻结** |
| 动力学损失 | 逐细胞 θ 求解 | **‖v_head − (βu−γs)‖²**（按 1/(1+s) 加权） |
| 重构 | 标准化 MSE | **原始计数 NB 似然**（基因级 dispersion） |
| 对齐/平滑损失 | 有 | 删除 |
| 读 U | ✗（U 置零 cos=1.0） | **✓**（U 打乱 cos 0.49） |

**演进**：
1. JVP 切向 + `‖v_tan−(βu−γs)‖²` → **不可识别**（自由 JVP 可拟合任意 β,γ；模型 γ/β=1.34 vs 数据 0.008）。
2. 去掉 JVP，速度直接 `βu−γs` → 3/3 通过，但速度是解析函数、非网络产生。
3. **速度改为线性网络头**（读 u,s,innovation），用动力学目标约束 → **3/3 通过，且速度由网络产生**。（MLP 头因过拟合 count-split 稳定性掉到 0.57；线性头稳定 0.92。）

## D1 单元测试 ✅

| 测试 | 结果 | 判据 |
|---|---|---|
| 速度形状/有限 | ok | ✅ |
| 局部打乱 U 后方向改变 | cos 中位 = **0.268** | <0.9 ✅ |

## D2 训练与验收（pancreas，2000 HVG，6000 步，~8 min）

| 验收 | 指标 | 判据 | 结果 |
|---|---|---|---|
| ① 局部打乱 U 方向 cos | **0.489** | ≤0.9 | ✅ |
| ② cos(v, v_kin) > cos(v, mean-shift) | **0.223** vs −0.015 | > | ✅ |
| ③ count-split 两半方向 cos | **0.919** | ≥0.8 | ✅ |
| ④ CBDir / ICCoh | — | 参考 | 未算 |

**验收 3/3 通过。**

## 结论

- **GFG v2 通过 D2 全部验收**：网络速度头**读 U**（0.489，对比当前 GFG 的 0.9998）、方向**对齐 v_kin**（0.223 ≫ mean-shift）、count-split **稳定**（0.919）。
- 速度由**网络头**产生（线性读出 u,s,u-innovation），并受基因级动力学速率约束。
- 未做 MouseBrain 训练（无该数据）；可作下一步。

## 命令

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python scripts/phase0/w2_gfg_v2.py --mode unit-test
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python scripts/phase0/w2_gfg_v2.py --mode train --steps 6000 --batch 64
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv-gpu/bin/python scripts/phase0/w2_gfg_v2.py --mode accept
```

产物：`outputs/w2_gfg_v2_unit.json`、`outputs/w2_gfg_v2_accept.json`、`outputs/w2_gfg_v2_pancreas.pt`。
