# VeloRoute 项目交接文档（Handoff）

> 用途：把本项目的背景、方法、全部实验数据、当前状态和下一步完整移交给新 agent。
> 2026-09-21 代码审计更正：**真实 velocity 增量尚未建立，不能宣称状态外信息为零。** 分布训练、静态对照、验证折标准化及 R² 定义存在已确认 bug。旧 RENGE +0.022 和“99% 冗余”的机制归因待重算。详见 `outputs/veloroute_bug_audit_20260921/RESULTS.md`；该更正优先于下文历史解读。
> 最后更新：2026-09-21。所有数字来自实际落盘产物，未做外推。

---

## 本轮审计更正（优先阅读）

- 修复了分布训练 no-grad 断路、专家均值替代混合分布、静态臂绕过训练专家、分布分支跳过门控校准等问题。CPU 验收：156 passed / 1 skipped。未重跑真实训练，旧预测和最终确认集保留。
- 旧 `R²=0.989` 实际使用未中心化速度能量作分母，不能解释为 98.9% 的逐细胞方差由状态解释；方向诊断也曾在验证源重拟合标准化。代码已修复，历史数值待重算。
- MeRLin **没有完成跨时间 velocity / router 增量实验**；完成的是 308/315 定量、克隆提取与 Day21 同时间关联。308（in vivo）和315（in vitro）不能直接配对。
- 新可用路径：另一项目已准备 GSM9044662/663 的 in-vitro CTRL→D+T 1周矩阵，报告 958 个跨条件共享克隆。315 定量覆盖全部 11,889 个发布 CTRL barcode，但提取/发布标签仅 1,252/3,172 完全一致。下一步先核实源 RNA 与文库身份、采用经审计的发布克隆标签，再构建统一 fold；目标 S/U 不是该预测任务必须输入。
- 未证明跨时间命运机制；“信息论不可修”“创新信息为零”“唯一稳定作用是场先验”等历史表述撤回。部分负结果仍是有效的特定实验观察，须按 bug 影响范围复核。

---

## 0. 一页速览

| 项 | 内容 |
|---|---|
| 项目名 | VeloRoute |
| 仓库 | `/home/yuchang/wangjiaxuan` |
| 核心问题 | velocity 能否让模型区分"同状态细胞走向不同命运"？ |
| 方法 | 源侧 velocity 条件化 router + 共享骨干 + K≤8 低秩专家 + EM/门控 |
| 主要数据 | RENGE、Kang、打包分化基准（pancreas/dentate/bonemarrow/lung）、MeRLin |
| 核心设计 | 三臂对照：static（无速度）/ gfg_joint（真速度）/ gfg_joint_shuffled（打乱速度） |
| 核心结论 | velocity vs 打乱 ≈ 零；velocity 价值 = 方向场先验，不是逐细胞信号 |
| 根因 | 速度 99% 可由状态 z 确定性推出；独立创新项方向一致性 ≈ 0 |
| 决定性未完成实验 | MeRLin 克隆命运跨时间检验（待 Day0 配对样本） |

---

## 1. Motivation 与核心假设

**Motivation**：RNA velocity 从 spliced/unspliced (S/U) 比例推断单细胞转录方向。领域直觉是：两个当前状态相同的细胞，若 U/S 信号不同，则正在走向不同命运。

**核心假设**：将 velocity 条件化进 router/混合专家，模型可区分"同状态、异命运"的细胞，从而在异质扰动响应预测上获得**超越纯状态模型**的增益。

**形式化**：增益 > 0 ⟺ I(命运; v | z) > 0，即给定状态 z 后速度 v 仍携带关于命运的互信息。

**项目已承认的边界**：工程跑通 ≠ 项目成功；合成机制阳性 ≠ 真实生物学增量。四个声明层级必须分开：
1. 全量组件是否实现（已实现）
2. 合成机制是否可行（可行）
3. 真实任务是否有增量（**未建立**）
4. 是否有 GFG 特异性 / 生物学机制（**未建立**）

---

## 2. 方法 / 架构

### 2.1 模型通路

```
源侧基因 U/S → GFG decoder JVP → vS/(1+S) → 冻结 PCA 旋转
  → 50→64 表示 → router q(k | z₀, v₀, e_c) → 采样一次 k
  → 固定模式专家 F_k → RK4 积分 → 终态分布

冻结潜状态 z₀ ─┐
冻结源速度 v₀ ─┼→ 源侧 router → 采样 k → 共享骨干 + K 低秩专家 → RK4 → z₁
条件嵌入 e_c ──┘
```

- **v₀ 只进入 router**（及可选的训练期模式兼容代价）；专家场不直接读 velocity。
- 条件 c 不进入 GFG 前向接口；条件相关梯度 ≠ c 是前向输入。
- 推理读源侧 U/S，因此**不能称 velocity "仅训练监督"**。

### 2.2 关键实现

- GFG 后端：本地 MouseBrain pretrained checkpoint，decoder-JVP 在线计算，与 router 联合训练（`dynamics_backend=gfg, joint_dynamics=true`）。
- 全宽参数量 20,021,704（含冻结回退）；老师另有 241,416 训练期参数。
- GFG checkpoint SHA256：`a9d9fa39063312230e4ba64309e9264ba3c2cb7ac549f04544b366df57c1340b`
- 路径：`/data/yuchang/GFG/results/mousebrain_graphbatch_directed_v1_seed0/final.pth`
- 低秩下投影随机初始化、上投影零初始化；逐专家责任度加权误差，不回归专家平均场。
- 场损失与路由 CE 有显式 detach 边界；生成时专家索引全程锁定。

### 2.3 训练策略演进（重要）

| 版本 | 训练信号 | 耦合方式 | 速度预处理 |
|---|---|---|---|
| v1 固定四臂 | OT 伪配对回归 | Sinkhorn (ε=0.1) | 稳态残差（原始） |
| v2 router 监督修复 | OT 责任度 CE + teacher | Sinkhorn | 稳态残差 |
| v3 分布匹配 | 能量距离（无配对） | 无配对 | 背景减法（v − E[v\|z]） |
| v4 主仪器 | OT 责任度 + 场回归 | Sinkhorn | GFG decoder-JVP 在线 |

**背景减法 = 把速度减去"状态能预测到的部分"，只保留真正独特的部分**（术语对外可解释为"去除状态可解释成分"）。

---

## 3. 实验协议与统计口径

### 3.1 三臂对照

| 臂 | 输入 | 目的 |
|---|---|---|
| static | z₀ | 基线 |
| gfg_joint | z₀ + v₀ | 待检验 |
| gfg_joint_shuffled | z₀ + v₀(打乱) | 特异性对照 |

- **增益定义** = 对照误差 − joint 误差（**正 = velocity 有利**）。
- 打乱对照（`permute_local`）保留局部场结构（coherence 前后完全一致），只破坏亚邻域尺度的逐细胞微分配 → 隔离"逐细胞速度信息"。

### 3.2 指标

- 主指标：PCA-50 空间 energy V-统计量 `2E‖X−Y‖ − E‖X−X'‖ − E‖Y−Y'‖`（不取平方根）。
- 辅助：方差比、条件中心化基因均值 MSE。
- 统计：先按条件平均 seed，再以**条件**为单元配对 bootstrap 10,000 次（bootstrap seed 20260914）。**不把细胞数或 seed 当生物学重复**。
- 多重比较：pilot 9 项 → 置信水平 1−0.05/9；直接 router 6 项 → 1−0.05/6。

### 3.3 无泄漏契约

- 供体/时间留出：训练条件不含验证条件细胞。
- 预测先冻结+哈希核验，再统一读取开发目标。
- GFG 的 HVG/PCA/邻居/模式/标准化只由训练源拟合。
- 最终确认 TF 封存，不参与任何拟合/调参/早停。
- `validate_gene_source` 要求 task=='Kang_ctrl_to_IFNB_donor_holdout' 且 day==4，除非 metadata/gene_metadata 的 `kind=='exploratory_gain_probe'`（探索豁免）。

### 3.4 资源硬约束

- **只允许物理 GPU 0–3**（`gpu_policy.py` 强制）；GPU 4–7 留给其他任务。
- 不停止其他项目已有进程。
- CPU 环境 `.venv`（torch 2.6.0+cpu）；GPU 私有 `.venv-gpu`（继承只读共享 torch 2.7.1+cu126）。

---

## 4. 数据集

| # | 数据集 | 物种/场景 | 扰动 | U 中位数 | 细胞数 | 多峰筛查 | 特点 |
|---|---|---|---|---:|---:|---|---|
| D1 | RENGE day4→5 | 人 hiPSC | 14 TF 诱导 | day4 675 / day5 836 | 7,582/6,910 | ❌ 0/14 | 同 KO、ER-short；U 中等；方向迁移最好 |
| D2 | Kang ctrl→IFN-β | 人 PBMC | IFN-β 6h | 194 | 24,679 singlets (8 donors) | ❌ 1/35 边缘 | U 极浅；单扰动；8 供体留出 |
| D3 | Pancreas E15.5 | 小鼠胰腺 | 无（发育） | 1221 | 2,531 | ✅ 程序轴通过 | 经典速度展示；零膨胀控制后不可区分 |
| D4 | Dentate Gyrus | 小鼠海马 | 无（发育） | 313 | 2,930 | — | 打包 S/U |
| D5 | Bonemarrow CD34 | 人骨髓 | 无（正常） | 1781 | 5,780 | — | 打包 S/U |
| D6 | Lung regeneration | 小鼠肺 | 博来霉素损伤 | 115 | 24,882 | — | U 最浅 |
| D7 | MeRLin Day21 | 人黑色素瘤 PDX | BRAFi+MEKi | 3120 | 67,356 | ❌ 深度分层覆盖 | DTP 四程序 + 克隆条形码；U 最深 |
| D8 | MeRLin in-vitro naïve | 人黑色素瘤 | 无 | — | 65,868 | — | 克隆配对用 |

### 4.1 RENGE 细节

- 任务：同 KO 条件下 day4 源侧 → day5 终态的 ER-short 预测（**不是** de novo，**不是** 2→5 外推）。
- TF 切分：14 train / 4 development（LIN28A、NANOG、POU5F1、ZIC3）/ 5 confirmation（封存）。
- 训练源/目标：2,733 / 2,401；开发源/目标：1,133 / 944。
- 工作空间：训练折选 2,000 基因，PCA-50；条件表示 23×2,560 冻结 ESM2-3B，模型内部条件维 256。
- 原始文库纠错：每天两条 mRNA/GEX + 两条 gRNA；`_1`=样本索引、`_2`=barcode/UMI、`_3`=文库序列。

### 4.2 Kang 细节

- 不可变协议冻结在 `protocol_freeze/`；8 供体、24,679 singlets，冻结时未读表达值。
- 首批 primary grid = 8 folds × 3 seeds ×（5 同几何 router 臂 + 单独 static K1 + 3 简基线）= 216 份预测。
- 供体：101、1015、1016、1039、107、1244、1256、1488。

### 4.3 MeRLin 细节（重点数据集）

**生物学背景**：人黑色素瘤 PDX（WM4237-1），BRAF+MEK 双靶点抑制剂（dabrafenib+trametinib）治疗。耐药休眠经典体系：治疗初期大部分死亡，一小群"药物耐受持久细胞"（DTP）潜伏存活→耐药复发。

**时间线**（每时间点 in-vitro 与 in-vivo 两套、各 2 重复）：

| 时间点 | 含义 |
|---|---|
| Day0 | 治疗前（naïve） |
| Day21 | 治疗中（残余 DTP） |
| Day92 | 耐药复发 |

**三个独特优势**（本项目唯一同时具备）：
1. **克隆条形码**：每细胞携带慢病毒条形码标记来源克隆 → 命运的 ground truth，不依赖 velocity 推断。
2. **U 最深**：U/(S+U) 池化 27.8%（Day21），逐细胞中位 9.6%——约为 Kang/RENGE 的 5 倍。
3. **已发表命运程序**：5 个 DTP 签名（stress-like 21 基因、neural-crest-like 32、lipid 16、PI3K 17、ECM 24）。

**它要回答的问题**：前五数据集说"velocity 逐细胞信息≈0"，有两种可能：(a) 命运信息在 RNA 层不存在；(b) 存在但 velocity 测不到。克隆条形码可区分二者。

**样本映射**：

| Run | 样本 |
|---|---|
| SRR33960314/315 | in-vitro naïve (+rep) |
| SRR33960312/313 | in-vitro D+T treated (+rep) |
| SRR33960310/311 | in-vivo Day0 naïve (+rep) |
| SRR33960308/309 | in-vivo Day21 treated (+rep) |
| SRR33960306/307 | in-vivo Day92 (+rep) |
| SRR33960300-305 | WM4007 系列（无克隆） |

**条形码配置**：flank `CAGATCTTAGCCACTTTTTAAAAGAAAAGGGGG`，规则见 `merlin_barcode_config.json`。

---

## 5. 全部实验与详细数据

### 5.1 合成正对照（机制验证）

指标：三 seed 终态组分比例 MAE（越低越好）。训练只看未配对群体分布，逐细胞真值命运仅评价。

| 系统 | 静态 | 冻结 GFG | 联合 GFG | 前编码 U 置换 |
|---|---:|---:|---:|---:|
| same_S_velocity_branch | 0.3000 | 0.0636 | **0.0616** | 0.0587 |
| state_velocity_interaction | 0.3000 | 0.0718 | **0.0633** | 0.2922 |
| position_branch | 0.1035 | 0.1069 | **0.1023** | 0.1018 |
| no_increment | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**结论**：S×U 对应关系决定比例时，联合 GFG 胜过静态和前编码 U 置换。**机制可行**（非生物学标定，是隐藏状态群体组成 toy）。不足以证明联训本身优于冻结 GFG。

### 5.2 RENGE

#### 5.2.1 固定四臂 × 3 seed（OT 耦合，最小模型）

12 份源侧预测冻结后统一评价。增益 = 对照误差 − joint GFG 误差：

| 比较：joint GFG 相对 | 指标 | 增益 | 校正后 CI |
|---|---|---:|---|
| static | energy_distance | −0.003415 | [−0.010551, +0.002427] 跨零 |
| gfg_joint_shuffled | energy_distance | +0.000143 | [−0.000419, +0.000758] 跨零 |
| gfg_frozen | energy_distance | +0.000624 | [−0.000507, +0.001662] 跨零 |
| static | log_variance_error | −0.000880 | [−0.002974, +0.000999] 跨零 |
| static | **condition_centered_gene_mean_mse** | **+3.023e-06** | **[+3.202e-07, +6.782e-06] 显著正** |
| gfg_joint_shuffled | condition_centered_gene_mean_mse | +3.048e-07 | [−3.901e-07, +9.794e-07] 跨零 |

**→ 主指标 energy 全部跨零；仅"条件中心化基因均值 MSE"对 static 显著正（均值层微弱信号）。**

#### 5.2.2 直接 router 监督修复轮（唯一一轮，3 臂 × 3 seed × 600 步）

固定专家候选，仅 router/GFG 更新，用精确混合分布指标（排除模式抽样噪声）：

| 比较 | 指标 | 增益 | 校正后 CI |
|---|---|---:|---|
| static | exact_mixture_energy | +3.329e-05 | [−6.895e-05, +1.189e-04] 跨零 |
| gfg_joint_shuffled | exact_mixture_energy | +2.223e-06 | [−7.615e-05, +6.001e-05] 跨零 |
| static | log_variance_error | +1.999e-05 | [−6.448e-05, +8.376e-05] 跨零 |
| gfg_joint_shuffled | log_variance_error | −2.166e-05 | [−6.820e-05, +3.449e-05] 跨零 |
| static | condition_centered_gene_mean_mse | +4.048e-08 | [−7.279e-08, +1.804e-07] 跨零 |
| gfg_joint_shuffled | condition_centered_gene_mean_mse | −8.656e-08 | [−2.870e-07, +5.928e-08] 跨零 |

**→ 六项主比较全部跨零。**

#### 5.2.3 传导断裂诊断（关键）

- GFG 对条件内 U 置换的**速度相对 RMS 改变 = 0.3155**（速度通道是活的）。
- 但 **router 概率 MAE 仅 0.000237**（路由通道几乎没收到）。
- 唯一旧估计器探索轮：vs static_router 增益 **−1.6e-05**（显著为负）。

**→ 传导链在 router 处断裂。**

#### 5.2.4 GFG 速度方向诊断（4 留出 TF）

| 留出 TF | n | cos(v, 未来) | cos(打乱, 未来) | cos(E[v\|z], 未来) | cos(innovation, 未来) | z 冗余 R² |
|---|---:|---:|---:|---:|---:|---:|
| LIN28A | 356 | +0.332 | +0.182 | +0.335 | −0.049 | 0.989 |
| NANOG | 285 | +0.338 | +0.182 | +0.341 | −0.030 | 0.989 |
| POU5F1 | 399 | +0.305 | +0.182 | +0.310 | −0.075 | 0.989 |
| ZIC3 | 93 | +0.330 | +0.183 | +0.337 | −0.096 | 0.989 |

训练 TF：ETS2 cos 0.343、FOXH1 0.345（shuffled ~0.16-0.17）。

**→ 速度方向正确（cos 0.31-0.34，高于打乱 0.18）；但 z 冗余 R²=0.989；创新项方向 ≈ 0/负。方向信息 100% 来自状态场成分。**

#### 5.2.5 残差估计器方向诊断

| 留出 TF | cos(v, 未来) | cos(innovation, 未来) | z 冗余 R² |
|---|---:|---:|---:|
| LIN28A | +0.150 | +0.000 | 0.542 |
| NANOG | +0.171 | +0.007 | 0.542 |
| POU5F1 | +0.148 | −0.008 | 0.542 |
| ZIC3 | +0.119 | +0.011 | 0.542 |

#### 5.2.6 创新尾部检验

| 数据/条件 | cos_v_mean | innov_mean | q90 | p_mean | p_tail |
|---|---:|---:|---:|---:|---:|
| RENGE ETS2 (train) | 0.343 | −0.078 | 0.164 | 1.0 | 0.2345 |
| RENGE FOXH1 (train) | 0.345 | −0.083 | 0.171 | 1.0 | — |
| Pancreas Alpha (train) | 0.468 | −0.041 | 0.276 | 1.0 | 0.001 |
| Pancreas Beta (train) | 0.411 | −0.067 | 0.275 | 1.0 | — |

**→ innov_mean 全部 ≈0 或负；p_mean 全为 1.0。**（pancreas Alpha 尾部 p=0.001 但方向为负，是反信号。）

#### 5.2.7 背景减法下游实验（分布匹配，无 OT）

| 数据集 | static | joint | shuffled | joint vs static | joint vs shuffled |
|---|---:|---:|---|---|---|
| RENGE | 0.4064 | **0.3842** | 0.3843 | **+0.0222** [+0.0148,+0.0296] ✅ | +1.42e-05 [−0.0004,+0.0004] 精确零 |
| Pancreas | 12.4584 | 12.4025 | 11.8156 | +0.0559 [−0.0151,+0.1269] 跨零 | −0.5868 [−1.122,−0.052] 显著负 |

RENGE 逐条件增益：[0.0340, 0.0145, 0.0151, 0.0252] 全正。

**→ 去掉状态可解释成分后，velocity 的"状态方向场先验"产生显著正增益（+0.022）；但它 vs 打乱 = 精确零 → 增益全部来自状态场成分。**

#### 5.2.8 多峰筛查

14 个训练 TF 的训练侧密度多峰筛查（BIC≥10 / Ashman≥2 / split 支持≥0.8 / 高斯零模拟）：**0/14 通过**。

### 5.3 Kang（PBMC ctrl→IFN-β，8 供体，216 冻结预测）

primary scope `all_primary`，增益 = 对照误差 − gfg_joint_router 误差：

| 对照 | 增益 | 校正后 CI | 判定 |
|---|---:|---|---|
| static_K1 | −0.003239 | [−0.015006, +0.003146] | 跨零 |
| static_router | −1.818e-05 | [−0.001050, +0.001100] | 跨零 |
| **gfg_joint_shuffled_U** | **+6.976e-05** | [−0.000861, +0.001111] | **跨零（打乱=真实）** |
| gfg_frozen_router | +0.000297 | [−0.000452, +0.000974] | 跨零 |
| raw_SU_router | +0.000380 | [−0.000574, +0.001265] | 跨零 |
| training_celltype_mean_shift | **−0.111882** | [−0.240047, −0.005323] | **显著负（均值平移更好）** |
| training_celltype_target_resample | +0.091625 | [−0.395934, +0.588418] | 跨零 |
| identity | +23.8899 | [+21.7676, +25.6783] | 显著（平凡） |

其他：
- `velocity_multimodal_increment`: **NOT_ESTABLISHED**
- `celltype_noninferiority`: noninferior=true（margin 0.02，mean_excess −0.0199）
- 训练侧模式筛查：支持类型极少（多数 heldout donor supported_types=[]），gaussian_null_positive_fraction=0.0
- 六臂间 energy 差 <0.001（塌缩）

**→ velocity 与打乱完全不可区分；显著差于简单均值平移基线。**

### 5.4 打包分化基准（速度增益探针 A1 vs A2）

增益 = error(shuffled) − error(velocity)，正 = velocity 有利：

| Run | 数据集设计 | 验证集 | 增益 | 95% CI | 判定 |
|---|---|---|---:|---|---|
| probe_lung_v1 | lung | 11,12,13 | −0.008 | [−0.087,+0.092] | null |
| probe_pancreas_v1 | pancreas main | Delta,Epsilon | +2.848 | [+1.191,+4.505] | positive |
| probe_pancreas_seeds2 | pancreas main | Delta,Epsilon | −0.017 | [−0.783,+0.750] | null |
| probe_pancreas_main_s20 | pancreas main | Delta,Epsilon | +0.046 | [−0.050,+0.141] | null |
| probe_pancreas_main_s30 | pancreas main | Delta,Epsilon | −0.195 | [−0.452,+0.062] | null |
| probe_pancreas_main_s40 | pancreas main | Delta,Epsilon | −0.842 | [−1.008,−0.676] | negative |
| probe_pancreas_swap_v1 | pancreas swap | Alpha,Beta | +1.209 | [+1.092,+1.326] | positive |
| probe_pancreas_swap_seeds2 | pancreas swap | Alpha,Beta | +1.867 | [+0.009,+3.725] | positive(mat) |
| probe_pancreas_swap_s20 | pancreas swap | Alpha,Beta | −0.416 | [−0.505,−0.328] | negative |
| probe_pancreas_swap_s30 | pancreas swap | Alpha,Beta | −0.334 | [−0.508,−0.160] | negative |
| probe_pancreas_swap_s40 | pancreas swap | Alpha,Beta | −0.871 | [−1.080,−0.663] | negative |
| probe_bonemarrow_v1 | bonemarrow main | Dendritic,Mega | +1.821 | [+1.041,+2.601] | positive |
| probe_bonemarrow_main_s20 | bonemarrow main | Dendritic,Mega | −0.632 | [−1.506,+0.242] | null |
| probe_bonemarrow_swap_v1 | bonemarrow swap | Ery,Mono | −0.596 | [−0.630,−0.561] | negative |
| probe_bonemarrow_swap_s20 | bonemarrow swap | Ery,Mono | +1.268 | [−0.024,+2.560] | null |
| probe_dentate_v1 | dentate | gi_gm_12,nb_gi_12 | +1.585 | [+1.552,+1.619] | positive |
| probe_dentate_main_s20 | dentate | gi_gm_12,nb_gi_12 | +0.229 | [+0.091,+0.366] | positive |
| probe_dentate_main_s30 | dentate | gi_gm_12,nb_gi_12 | −0.263 | [−0.350,−0.177] | negative |
| probe_dentate_swap_s20 | dentate | gi_gm_35,nb_gi_35 | −0.682 | [−1.003,−0.362] | negative |
| probe_dentate_swap_s30 | dentate | gi_gm_35,nb_gi_35 | +0.412 | [+0.408,+0.415] | positive |

**→ 主要是"种子彩票"：pancereas 3 阳/7 非阳、dentate 3 阳/2 阴、bonemarrow 翻转、lung null。跨 seed 组复验后衰退。**

额外：分布匹配 + kNN 平滑 velocity（k=10）探针仍 seed 不稳定（pancreas 2 阳 1 阴；dentate 2 阳 1 边缘）。

### 5.5 主仪器（全预算，OT 耦合，seed 0，全组件）

| 数据集 | static | gfg_joint | shuffled | joint vs static | joint vs shuffled |
|---|---:|---:|---|---|---|
| pancreas | 12.4584 | 11.7204 | 11.1717 | +0.7380 [+0.583,+0.893] 正 | −0.5487 [−1.024,−0.074] **负** |
| dentate_pairs | 13.5141 | 13.5733 | 13.7796 | −0.0592 [−0.112,−0.0067] 负 | +0.2063 [+0.091,+0.322] 正 |
| bonemarrow | 10.6487 | 11.4285 | 11.1445 | −0.7799 [−1.412,−0.148] **负** | −0.2840 [−0.557,−0.012] **负** |
| lung_pairs | 2.0204 | 1.9574 | 1.7021 | +0.0630 [−0.311,+0.437] 跨零 | −0.2553 [−0.321,−0.189] **负** |

**→ 3/4 数据集真实速度显著差于打乱；唯一 joint 优于 static 的 pancreas，也输给打乱。**

### 5.6 机制诊断

| # | 实验 | 关键结果 |
|---|---|---|
| 21 | GFG 速度创新分解 | z 冗余 R²=0.989；创新方向 cos −0.03~−0.10 |
| 22 | 残差速度创新分解 | z 冗余 R²=0.542；创新 cos ≈ 0 |
| 23 | 非线性场检验（线性 vs MLP） | 残差方向 cos 不因非线性拟合而升高 |
| 24 | 条件自适应场信号 | trainfield 残差 cos +0.09~+0.21（p<0.001）——**场先验有信号** |

### 5.7 多峰筛查（各数据集汇总）

| 数据集 | 方法 | 结果 |
|---|---|---|
| RENGE | 14 训练 TF 密度多峰 | **0/14 通过** |
| Kang | 训练侧模式筛查 | 仅 1/35 边缘通过；多数 donor 无支持类型 |
| Pancreas | 程序轴 + 配对随机基因集对照 | 程序轴通过但**对照通过率 ~0.72-0.89** → 不可区分 |
| MeRLin Day21 | 三层校准（每基因打乱 / 表达匹配基因集 / 深度分层打乱） | 程序轴"通过"但对照通过 72-89%；**深度分层打乱零假设覆盖真实信号** → 不可确证 |

**→ 所有"多峰"证据均被表达匹配/深度匹配零假设复现，判为稀疏性/零膨胀伪影。**

### 5.8 MeRLin（Day21 样本）

**克隆×程序×状态检验**（2,571 有克隆细胞，164 克隆 ≥3 细胞，置换检验）：

| 程序 | 状态解释度 R² | 克隆效应 observed vs null_mean | p 值 |
|---|---:|---|---:|
| stress-like | 0.6603 | 0.001006 vs 0.000848 | 0.049 |
| neural-crest-like | 0.5763 | 0.000348 vs 0.000247 | **0.0005** |
| lipid metabolism | 0.4209 | 0.000859 vs 0.000645 | **0.0005** |
| PI3K signaling | 0.3830 | 0.000470 vs 0.000358 | **0.004** |
| ECM remodeling | 0.6334 | （见 `merlin308_clone_program.json`） | **0.009** |

**→ 5/5 程序全部显著：同状态细胞因克隆不同而走向不同命运，状态外命运结构存在，幅度小（克隆解释残差方差 ~1-2%）。**

**多峰筛查（Day21）**：五条 PC 轴 per_axis_support 全 0.0，approved=false；程序轴密度在 10/10 splits"通过"但随机基因集对照通过 72-89%。

**MeRLin 决定性实验（未完成）**：克隆命运跨时间检验——需同一批克隆在 Day0（治疗前）与 Day21（治疗中）的配对数据。核心问题：**Day21 不同克隆的不同命运程序，能否从 Day0 的 velocity 创新项预测？**

---

## 6. 根因分析（为什么 velocity 无效）

### 6.1 速度的"独特信息"为零

模型已有状态 z。速度要有用，必须提供 z 之外的命运信息。实测：

- **速度 99% 可从状态推导**（GFG R²=0.989；残差 R²=0.542）。
- 剩余"独立成分"（innovation）与未来方向一致性 ≈ 0（甚至轻微负）。

### 6.2 三层原因

| 层面 | 原因 | 证据 |
|---|---|---|
| **测量层** | U 计数太浅（Kang 194、RENGE 675），泊松噪声淹没逐细胞方向 | Kang 打乱 U = 真实 U（+7e-05）；U 越浅效果越差 |
| **估计层** | 速度计算是状态的平滑函数（GFG/残差都一样），同状态细胞速度≈相同 | 速度 99% 可状态推出；同状态速度差异≈0 |
| **生物层** | 命运可能不由转录层决定（染色质/蛋白/微环境）；RNA 测量看不到 | 打乱 U 后结果不变 |

### 6.3 通俗类比

Velocity 像**地图上每个位置画的风向标**——告诉你"这个位置的风通常往哪吹"，不是每辆车自己的 GPS。同一个路口的车看到同样的风向，无法区分它们各自要去哪。router 需要的是"你和旁边那辆车有什么不同"，而地图上没有这个信息。

### 6.4 传导断裂的历史观察（不可修结论已撤回）

- 速度对 U 置换响应 RMS 0.3155（速度通道活），router 概率 MAE 仅 0.000237（路由通道死）。
- 此处不能从特定估计器/损失和 kNN 方向代理的阴性推导信息论不可修；本轮已发现可修复的训练断路与错误的 R² 定义。
- OT 耦合额外偏差：OT 编码"最小位移耦合"，训练信号惩罚速度的远跳预测。

---

## 7. 当前状态

### 已完成
- 合成、RENGE、Kang、打包基准、主仪器、全部诊断、多峰筛查全部完成。
- MeRLin Day21 + in-vitro naïve 的定量（各 ~66-67k 细胞）+ 克隆提取完成。
- MeRLin 克隆×程序×状态检验完成（5/5 显著）。
- 背景减法主仪器（RENGE + pancreas）完成。

### 进行中（截至 2026-09-21）
- GPU 0–3 空闲（0 MiB 占用）。
- 一个陈旧进程 `evaluate_kang_full.py`（PID 2976507，Sep14 启动）仍挂着，可忽略/排查。
- MeRLin 下载：`SRR33960314` 部分下载（~313GB parts 保留），其余大样本 5 轮 pass 全部失败（网络间歇）。
  已转换 fastq：308、315（4 files / ~106.6GB）。转换数 converted=4。
- `mainline_bg/pancreas` 与 `mainline_bg/renge` 的 summary.json 已存在（见 5.2.7 / 5.5）。

### 阻塞
- MeRLin Day0（310/311）或 in-vitro treated（312/313）未下载 → 克隆命运跨时间检验无法进行。
- Papalexi 2021 U 快检待做（5' 化学，U 深度风险）。
- RegVelo 斑马鱼需自建斑马鱼 index（未建）。

---

## 8. 代码地图与复现命令

### 8.1 核心脚本（`scripts/phase0/`）

| 脚本 | 作用 |
|---|---|
| `prepare_gain_probe.py` | 从打包 S/U 构建增益探针 fold（含 gene pack） |
| `run_gain_probe.py` | 增益探针（含 `_fit_dist_torch` 分布匹配、`_smooth_velocity` kNN 平滑 k=10） |
| `run_gain_mainline.py` | 全仪器运行器（三臂 × 全预算；`--coupling`、`--background`、`--gene-dir`、`--conditions`） |
| `verify_velocity_direction.py` | 速度方向正确性（vs kNN 未来方向） |
| `innovation_tail_test.py` | 逐细胞创新尾部检验（置换零假设） |
| `nonlinear_field_test.py` | 非线性场残差检验 |
| `velocity_innovation_diagnostic.py` | 速度 z 冗余 + 创新分解 |
| `gfg_innovation_diagnostic.py` | GFG 版本创新诊断 |
| `precompute_velocity_background.py` | 预计算 E[v\|z] 背景 |
| `parse_merlin_tables.py` | 解析 MeRLin 补充表 |
| `merlin_extract_barcodes.py` | 克隆条形码提取（多进程） |
| `prepare_merlin_fold.py` | MeRLin fold 构建 |
| `analyze_clone_programs.py` | 克隆×程序×状态分析（kNN 状态回归 + 置换克隆检验） |
| `screen_merlin308.py` | MeRLin DTP 程序筛查 |
| `run_kang_experiment_grid.py` / `run_kang_full_grid.py` / `run_kang_mainline_queue.py` | Kang 主仪器 |

### 8.2 源码（`src/veloroute/`）

| 文件 | 作用 |
|---|---|
| `full_model.py` | `make_context` + `predict`（含 `background=` 逐细胞 r/v 减法） |
| `full_training.py` | `FullTrainConfig`（`velocity_background`、`coupling_mode`）；`_distribution_step` 软端点能量距离 |
| `gfg_inputs.py` | `validate_gene_source`（探索豁免 `kind=='exploratory_gain_probe'`） |
| `gfg_experiments.py` | `train_gfg`（读 gene pack；`gfg_task_gradient_median` NaN-guard） |
| `gfg.py` / `gfg_direct_router.py` / `full_components.py` | GFG 后端与组件 |
| `gpu_policy.py` | GPU 0–3 强制策略 |
| `metrics.py` | energy V-统计量等 |

### 8.3 关键数据路径

| 路径 | 内容 |
|---|---|
| `/data/yuchang/veloroute_gainprobe_20260915/` | 增益探针数据根（prepared folds、mainline 结果、背景） |
| `/data/yuchang/veloroute_ucheck_20260915/` | MeRLin U 快检数据根（下载、克隆、定量、Papalexi 表） |
| `outputs/veloroute_real_pipeline_20260912_v2/fold` | RENGE fold |
| `outputs/veloroute_experiment_ledger_20260914/` | 实验总账审计包 |
| `outputs/veloroute_kang_jobs_20260914/` | Kang 全部 jobs |

### 8.4 复现命令示例

```bash
# RENGE 背景减法主仪器（已完成，供参考）
.venv-gpu/bin/python scripts/phase0/run_gain_mainline.py \
  --fold outputs/veloroute_real_pipeline_20260912_v2/fold \
  --tag renge_bg --output <out> --coupling distribution_matching \
  --background <bg.npz> --gene-dir <gene_pack_dir> --conditions <conditions.npz>

# 胰腺背景减法
.venv-gpu/bin/python scripts/phase0/run_gain_mainline.py \
  --fold /data/yuchang/veloroute_gainprobe_20260915/prepared_pancreas_pathways \
  --tag pancreas_bg --output <out> --coupling distribution_matching \
  --background /data/yuchang/veloroute_gainprobe_20260915/pancreas_velocity_background.npz

# MeRLin 克隆×程序分析
.venv-gpu/bin/python scripts/phase0/analyze_clone_programs.py ...
```

---

## 9. 环境与资源

- CPU 环境：`.venv`（NumPy 1.26.4 / PyTorch 2.6.0+cpu）。
- GPU 环境：`.venv-gpu`（继承只读共享 torch 2.7.1+cu126，本地固定 NumPy/anndata）。
- GPU：8×A100-80GB（`nvidia-smi` 97000 MiB/卡）；**本项目只用 0–3**。
- 磁盘：`/data` 3.5T，当前 1.7T 用、1.6T 可用（53%）。
- 网络：代理 `127.0.0.1:7890` 间歇；AWS SRA 镜像最快（单连接 0.4-0.6MB/s，20 并发峰值 12-24MB/s）。

---

## 10. 下一步

### 10.1 立即可做（GPU 空闲，无需新数据）

| 任务 | 预计 | 产出 |
|---|---|---|
| pancreas/dentate/bonemarrow/lung 多峰三层校准筛选 | 数分钟 CPU | 判定程序轴多峰是否为稀疏伪影 |
| 背景减法 RENGE 实验多 seed 复验 | ~6h GPU | 确认 +0.022 稳健性 |
| 排查 `evaluate_kang_full.py` 陈旧进程 | 分钟 | 清理 |

### 10.2 需数据

| 任务 | 依赖 | 预计 |
|---|---|---|
| MeRLin Day0（310/311）或 treated（312/313）下载+转换 | 网络稳定窗口 | 1-2 天 |
| MeRLin Day0→Day21 fold 构建 | 上述完成 | ~2h CPU |
| **MeRLin 克隆命运跨时间检验（决定性）** | Day0+Day21 定量 | 数小时 CPU |
| 若 clone 效应跨时间可预测 → 标记引导路由新方法 | 上一步阳性 | 数周 |
| 若不可预测 → 边界刻画论文（阴性+机制诊断） | 上一步阴性 | 数周 |

### 10.3 长期备选

- 代谢标记（4sU/scEU-seq/sci-fate）：直接测"此刻合成了什么"，绕开 U/S 反推——需新管线。
- Smart-seq3 深测扰动数据（RegVelo 斑马鱼）：深 U + 扰动闭环——需斑马鱼 index。

---

## 11. 注意事项 / 不要误读

1. **不要把"工程通过/合成阳性"写成"真实 velocity 增益"。** 两轮真实增量未成立。
2. **能量指标区间跨零时写"未建立增量"，不写"已证明等效"。**
3. **只有 4 个反复查看的开发 TF**，不能当独立确认；最终 5 TF 封存。
4. **增益正负号**：本文档统一 `增益 = 对照误差 − joint GFG 误差`，正 = velocity 有利。
5. **打乱对照保留局部场结构**（coherence 完全一致），所以"velocity = 打乱"意味着没有逐细胞信息，而非"打乱破坏了全部结构"。
6. **RENGE 的 +0.022 来自状态方向场先验**，不是逐细胞路由；它与打乱精确相等证明这一点。
7. **MeRLin 克隆效应 5/5 显著**说明状态外命运信息**存在**，只是 velocity 测不到——不要写成"命运完全由状态决定"。
8. **声明层级**：全量组件已实现 ≠ 有收益；合成机制可行 ≠ 生物学增量；开发集信号 ≠ 确认集结论。
9. **`--speed-limit 1` 下载策略**在网络差时会丢弃数据并重试，5 轮 pass 全失败即由此；恢复下载需网络稳定窗口。
10. 本项目**未**运行正式 G1，**未**打开最终确认集。

---

## 12. 一句话总结

> 当前结论是：真实 velocity 增量未建立，且一部分关键训练/诊断证据受已确认实现问题影响。下一步使用修复版本，并先核实 MeRLin in-vitro 发布样本与源侧 S/U 的身份和标签对应；跨时间三臂实验尚未进行，不应提前宣告成功或信息论不可能。
