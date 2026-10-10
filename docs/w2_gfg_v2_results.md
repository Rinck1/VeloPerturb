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

---

## 下游增量实验：GFG v2 速度在 RENGE 上有没有用

数据：**RENGE day4→day5（留出 TF）**（项目主扰动任务）。在 RENGE 训练源 U/S 上重训 GFG v2（fold 2000 基因，4000 步），三臂端点模型 ridge 预测 day5 位移（static / gfg_v2 / shuffled），另加 v_kin 参照。指标 pseudobulk MSE、energy；条件 bootstrap。

| 臂 | pseudobulk MSE | energy |
|---|---:|---:|
| static（无速度） | 0.2285 | 2.1545 |
| gfg_v2 | 0.2287 | **2.1343** |
| shuffled | 0.2288 | 2.1382 |
| v_kin（解析） | **0.2275** | 2.1422 |

| 比较 | 指标 | 增益 [95% CI] | 跨零？ |
|---|---|---:|---|
| gfg_v2 vs static | pseudobulk | −0.00022 [−0.00035, −0.00010] | 否（**略负**） |
| **gfg_v2 vs shuffled** | pseudobulk | +0.00008 [−0.00026, +0.00042] | **是** |
| gfg_v2 vs static | energy | +0.0202 [+0.0125, +0.0280] | 否（正，但见下） |
| **gfg_v2 vs shuffled** | energy | +0.0039 [−0.0047, +0.0086] | **是** |

**结论**：
- **gfg_v2 vs shuffled 两个指标都跨零 → 没有速度特异性增量**。GFG v2 的速度与打乱速度不可区分。
- gfg_v2 vs static 的 energy 正增益（+0.020）是**分散度效应**（energy 奖励分散度），vs shuffled 后消失 —— 与此前 +0.022 的解读一致。
- v_kin 的 pseudobulk 略优于 gfg_v2（0.2275 vs 0.2287），两者都与 shuffled 无异。

**综合判断**：GFG v2 修好了"读 U"的能力（D2 3/3），**但下游没有正增量**——与 T1（MeRLin 灵敏阴性）、T3（ΔU 信度 0.06）一致。**工具合格 ≠ 假设成立**：U 里没有可用的额外命运信息，换一个能读 U 的提取器也不会凭空产生增益。

## 下游增量实验（MeRLin 310→308，GFG v2 速度）

在 MeRLin Day0(310，71,778 细胞) 上训 GFG v2（2000 HVG，4000 步），取其**基因级速度**做修正版超前-滞后：Day0 克隆均值速度 预测 Day21(308) S 变化，控制 Day0 S；零分布只打乱 Day21；313 个 ≥3 细胞共享克隆。

| 模块 | obs 偏相关 | ε_S 对照 | 边际相关 | null95 | p | 判定 |
|---|---:|---:|---:|---:|---:|---|
| ALL | −0.0151 | −0.0942 | **−0.0542** | +0.0158 | 0.933 | ❌ |
| stress-like | +0.0017 | −0.0698 | −0.0345 | +0.0327 | 0.456 | ❌ |
| neural-crest-like | +0.0232 | −0.0659 | −0.0412 | +0.0408 | 0.176 | ❌ |

**结论**：GFG v2 速度在 MeRLin 上也**预测不了 Day21 命运**（全部不显著，边际相关甚至为负）。与 MeRLin 的 T1（灵敏阴性）、T3（ΔU 信度 0.06）一致。**换用能读 U 的 GFG v2、换到最深的 MeRLin，都没有正信号。**


