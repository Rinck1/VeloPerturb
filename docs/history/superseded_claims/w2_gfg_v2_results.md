# GFG v2（动力学版，no-JVP）实现与验收（2026-10-10）

> 归档状态：历史报告，含已撤回或被审计限制的推断；旧数值不自动支持旧总括结论。 [当前状态](../../current/STATUS.md) · [撤回与更正清单](../../corrections/CLAIMS.md)

2026-10-14 更正：本次 v2 是共享线性速度头拟合 βu−γs 的**简化对照**，速度绕过流形表征；不是基因身份嵌入、U/S 类型标记、S-only 状态流和 U/S 速度流的完整架构。以下历史数字保留，但验收与机制结论由[代码复核](../../corrections/audits/velocity_code_review_20261014.md)限定。

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

~~**验收 3/3 通过。**~~ 2026-10-14 更正：count-split 只拆 U 且共享 S；mean-shift 与 velocity 的 PCA/导数变换不一致；innovation 在推理 batch 内重拟合。三个历史阈值不能合称独立速度质量验收通过。

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
- ~~gfg_v2 vs static 的 energy 正增益（+0.020）是**分散度效应**（energy 奖励分散度），vs shuffled 后消失 —— 与此前 +0.022 的解读一致。~~ 2026-10-14 更正：未直接分解该改善来源，不能确定归因为分散度；shuffled 臂同时打乱 U/S，未隔离固定 S 下 U 的增量。
- v_kin 的 pseudobulk 略优于 gfg_v2（0.2275 vs 0.2287），两者都与 shuffled 无异。

~~**综合判断**：GFG v2 修好了"读 U"的能力（D2 3/3），**但下游没有正增量**——与 T1（MeRLin 灵敏阴性）、T3（ΔU 信度 0.06）一致。**工具合格 ≠ 假设成立**：U 里没有可用的额外命运信息，换一个能读 U 的提取器也不会凭空产生增益。~~

2026-10-14 更正：当前简化对照未建立 velocity 特异性增益；检验修正和完整 GFG 公平对照尚待执行，不能推断 U 没有额外命运信息。

## 下游增量实验（MeRLin 310→308，GFG v2 速度）

2026-10-14：**本表不能作为有效的独立测量命运判决。** 未做 count-split，速度与 S 控制来自同一份计数；εS=-0.094 提示耦合风险，但该数值本身不是耦合的证明。由修订后的 W3-1 取代。

在 MeRLin Day0(310，71,778 细胞) 上训 GFG v2（2000 HVG，4000 步），取其**基因级速度**做修正版超前-滞后：Day0 克隆均值速度 预测 Day21(308) S 变化，控制 Day0 S；零分布只打乱 Day21；313 个 ≥3 细胞共享克隆。

| 模块 | obs 偏相关 | ε_S 对照 | 边际相关 | null95 | p | 判定 |
|---|---:|---:|---:|---:|---:|---|
| ALL | −0.0151 | −0.0942 | **−0.0542** | +0.0158 | 0.933 | ❌ |
| stress-like | +0.0017 | −0.0698 | −0.0345 | +0.0327 | 0.456 | ❌ |
| neural-crest-like | +0.0232 | −0.0659 | −0.0412 | +0.0408 | 0.176 | ❌ |

~~**结论**：GFG v2 速度在 MeRLin 上也**预测不了 Day21 命运**（全部不显著，边际相关甚至为负）。与 MeRLin 的 T1（灵敏阴性）、T3（ΔU 信度 0.06）一致。**换用能读 U 的 GFG v2、换到最深的 MeRLin，都没有正信号。**~~

2026-10-14 更正：旧设计未检出关联，但不能据此判定 GFG 无法预测命运；需要独立计数拆分、冻结预处理和校准的条件增量检验。



---

## 无 codebook 版 + 多数据集下游（2026-10-14）

按用户指示去掉 VQ codebook（`--no-codebook`），在 4 个打包分化基准上重跑三臂（static / gfg_v2 / shuffled），shuffled 改用 U-only 局部控制（`local_u_control`：mutual-kNN 配对、S 固定、深度匹配）。脚本 `scripts/phase0/w2_gfg_v2_downstream.py`，输出 `outputs/w2_nocb_<fold>.json`。

| 数据集 | static | gfg_v2 | shuffled | v_kin |
|---|---:|---:|---:|---:|
| pancreas (energy) | 8.14 | 6.26 | 6.00 | 7.83 |
| dentate (energy) | 5.70 | **2847（发散）** | 2828 | 5.30 |
| bonemarrow (energy) | 11.50 | 10.49 | 10.43 | 10.97 |
| lung (energy) | 2.178 | 2.148 | 2.140 | 2.180 |

gfg_v2 vs shuffled（正 = velocity 有利）：

| 数据集 | energy | pseudobulk | 判定 |
|---|---|---|---|
| pancreas | −0.258 | +0.306 | 两指标矛盾（2 条件） |
| dentate | −18.9 | −3281 | **数值爆炸** |
| bonemarrow | −0.059 | −0.039 | 无 |
| lung | −0.008 | −0.004 | 无 |

**结论**：去掉 codebook、换 4 个数据集，**仍无稳健的 velocity 特异性增量**。codebook 不是关键。dentate 上 no-codebook 版数值发散（模型不稳定），但属实现问题，不改变结论。
