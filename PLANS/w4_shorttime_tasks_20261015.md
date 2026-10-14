# W4 执行任务书：短时线（交给服务器 agent）

日期：2026-10-15 起　|　基线提交：`7d1db23`　|　规格：Claude　|　执行：服务器 agent　|　审阅：Rinck

配套文件：**预注册** `PLANS/w4_shorttime_prereg.md`。本任务书说明"做什么、怎么做"；预注册规定"检验什么、怎样算成功"。两者冲突时以预注册为准。

## 0. 背景与主张

项目主线改为短时线（Rinck 2026-10-14 决定）。要检验的主张：

> 扰动后的短时间内，群体层面的 RNA 速度（由 U、S 计算）能提前预示 S 的下一步变化，比只用当前 S 更准；而且增益随预测跨度变长而衰减。

和原命题的区别：
- 不再讲逐细胞或逐克隆的命运；
- 只讲扰动群体、几小时到一天的预测。

理论上，这正好是非配对数据下可以识别、准稳态假设不成立的区域。

架构采用**双流**：
- 状态流只吃 S；
- 速度流吃 U+S，使用简单动力学速度 v = u − γ_g·s；
- 速度通过状态映射的雅可比投影到状态坐标。

不使用 GFG 解码器 JVP；不使用 MouseBrain seed0 权重。

## 1. 通用规则（违反任一条即停止汇报）

- 只新增，不改 `src/`。脚本放 `scripts/phase0/w4_*.py`，测试放 `tests/test_w4_*.py`，输出放 `outputs/w4_<task>_<version>/`，结果写进 `docs/w4_results.md`。
- **GPU**：按 2026-10-14 授权，可同时使用物理 GPU 0–3；GPU 4–7 不用。启动前检查占用，不 kill 别人的进程。本任务以 CPU 为主。
- **下载**：单个数据集总量超过 50 GB 前先问 Rinck。所有下载记录来源 URL、文件名、大小和 SHA256。原始数据放 `/data/yuchang/veloroute_w4/`，不放仓库。
- **封存**（最重要）：
  - **复制集**是 RENGE 的 day2→3 和 day3→4，以及 MIX-seq、sci-fate 的全部时间段。在预注册冻结（W4-2）之前，只允许做技术 QC：细胞数、U/S 计数分布、guide 分配率。**不允许**计算任何扰动效应、做任何跨条件比较、画按条件着色的降维图。
  - **开发集**是 RENGE day4→5，只用于调试代码，结果只能写"开发"。
- 每个 JSON 记录 `git_commit`、seed、命令行参数、数据哈希和耗时。
- 每完成一项提交一项并推送到 `origin/master`，提交信息带任务编号。
- 只做任务书写的事；有想法写进汇报的"建议"一栏。

## 2. 顺序与资源

| 任务 | 内容 | 资源 | 预计 | 依赖 |
|---|---|---|---|---|
| W4-0 | 收尾 W3-7a（原版 GFG 在目标数据上重训的读 U 敏感性） | 1–4 GPU | 0.5–1 天 | 无 |
| W4-1 | 数据下载与处理 | CPU + 磁盘 | 2–3 天，可并行 | 无 |
| W4-2 | 冻结预注册 | — | 0.5 小时 | W4-1 的 QC 完成 |
| W4-3 | 实现双流短时框架和单元测试 | CPU | 1–2 天 | 无，可与 W4-1 并行 |
| W4-4 | 开发集运行（RENGE day4→5） | CPU | 0.5 天 | W4-3 |
| W4-5 | 复制集运行（RENGE day2→3、day3→4） | CPU | 0.5 天 | W4-2、W4-4 |
| W4-6 | 时间跨度衰减 | CPU | 0.5 天 | W4-5 |
| W4-7 | 外部复制（MIX-seq）和机制验证（sci-fate） | CPU | 1–2 天 | W4-1、W4-2 |
| W4-8 | 消融（foundation 状态流、扰动嵌入、直接靶基因） | CPU/GPU | 1 天 | W4-5 |

---

## W4-0 收尾 W3-7a

W3 里原版 GFG 在 RENGE、MeRLin 上的重训已经准备好，但读 U 敏感性的结果还没推上来。按 W3 任务书 W3-7a 跑完：V0、V3、V4、V7 四种变体，三个 seed，报告 GFG 速度的 cos、模长比，以及同变体下解析速度的参照。结果写进 `docs/w3_results.md`。

这是给导师"GFG 是否被不公平对待"的交代，**优先完成**。下游三臂比较不再做（短时线另有协议）。

---

## W4-1 数据下载与处理

先做 1a；1b 到 1d 可以并行。每个数据集最后都要交一张 QC 表：细胞数、每个单位（TF / 细胞系 / 时间点）的细胞数、每个细胞 U 和 S 计数的中位数、U/(U+S) 的中位数，以及 guide 或单位分配率。

### W4-1a RENGE day2、day3（主数据，必须完成）

- **来源**：GEO **GSE213069**（RENGE，Ishikawa 等，Commun Biol 2023）。仓库里 day4、day5 已用同一来源处理（`data/renge/processed_release_v1/`）。
- **找 run**：从 GSE213069 的 SRA Run Selector 导出 run 表，挑出 day2、day3 的 GEX run 和 guide/feature run。和服务器上已有的 `outputs/veloroute_phase0/renge_structural_20260912/run_manifest.csv` 对照，确认 day4/day5 的 run 编号（如 SRR21518305、SRR21518308）和已处理的一致。把 day2/day3 的 run 列表写进 `configs/w4_renge_day23_runs.csv`。
- **下载**：用 `tools/sratoolkit.3.4.1-ubuntu64/bin` 的 `prefetch` 加 `fasterq-dump --include-technical --split-files`，和 day4/day5 一样。下载前先估总大小，超过 50 GB 先问。
- **定量**：**完全复用** day4/day5 的流程和参数（`configs/renge_preprocess.yaml`）：GENCODE v32 splici 索引 r91、几何 `1{b[16]u[10]x:}2{r:}`、方向 rc、同一样本的多条 GEX run 联合做 UMI 解析、USA 三层分开、相同的 QC 阈值和 guide 判定规则。新建一个配置 `configs/w4_renge_day23_preprocess.yaml`，只改 `days` 和输出路径。
- **输出**：`data/renge/processed_release_v2/day2/day2.h5ad`、`day3/day3.h5ad`，附输入哈希和流程版本。
- **一致性检查**：用新配置**重新处理 day4 的一条 run**，和已发布的 day4 计数逐细胞比较 S、U 总量（相关 ≥ 0.999）。不一致就停止。

### W4-1b MIX-seq 曲美替尼时间序列（外部复制，先核实再下载）

- **来源**：McFarland 等，Nat Commun 2020（bioRxiv 10.1101/868752）。处理好的 UMI 矩阵在 figshare（article 10298696），**里面没有 U**。
- **第一步核实**：读论文的 Data availability 和 figshare README，找原始 reads 的位置（SRA / dbGaP / 其他）。
  - **受控访问（如 dbGaP）或找不到**：停止此项并汇报，不要申请。
  - **公开**：只下载**曲美替尼时间序列**那一组：24 个细胞系混合培养，3–48 小时共 5 个时间点，加每个时间点的 DMSO 和未处理对照，约 13,713 个细胞。先列出对应 run 和总大小。
- **定量**：simpleaf/alevin-fry splici（GENCODE v32，与 RENGE 相同的参考），化学版本按论文写的 10x 版本设定，并用 `simpleaf` 的 chemistry 检查确认方向。
- **细胞系归属**：直接用 figshare 处理数据里作者给的细胞系分配，按 barcode 匹配；不要自己重新做 SNP 分离。报告匹配率。
- **输出**：`/data/yuchang/veloroute_w4/mixseq/processed/mixseq_trametinib.h5ad`，每个细胞带细胞系、时间、处理三个标签。

### W4-1c sci-fate（机制验证）

- **来源**：Cao 等，Nat Biotechnol 2020（bioRxiv 10.1101/666081）。A549，地塞米松 0、2、4、6、8、10 小时；每个时间点收样前用 4sU 标记 2 小时；6,680 个细胞。
- **核实 GEO 编号**：从论文 Data availability 读。候选是 GSE131351，**必须核实，不能直接用**。
- **先看处理好的文件**：是否已有每个细胞的"新合成"和"总"计数矩阵。有就直接用作实测速度的来源。
- **U/S**：sci-RNA-seq 用随机引物加 oligo-dT，内含子读段多，可以算 U。处理好的文件里没有 U 时，用作者的 sci-fate 处理流程（Cao 实验室 GitHub）重新比对，并加内含子/外显子分开计数。先估大小，超过 50 GB 先问。
- **输出**：`/data/yuchang/veloroute_w4/scifate/processed/scifate.h5ad`，包含 layers `spliced`、`unspliced`、`new`、`total` 和时间标签。

### W4-1d scNT-seq（sci-fate 的备选，低优先级）

- **来源**：GEO **GSE141851**（Qiu 等，Nat Methods 2020）。小鼠神经元 KCl 刺激，0–120 分钟，代谢标记。
- 只在 sci-fate 拿不到 U 时才做。先只下载处理好的矩阵，查有没有新旧 RNA 和 U/S。

### W4-1 汇报表

```
数据集 | 编号 | 下载文件 | 大小 | 细胞数 | 单位数 × 时间点 | 每单位细胞数中位 | U 中位 | S 中位 | U/(U+S) | 状态
```

---

## W4-2 冻结预注册

W4-1a 的 QC 完成后（MIX-seq、sci-fate 可以晚一点，它们在预注册里有各自的条款）：

1. 读 `PLANS/w4_shorttime_prereg.md`，只填写标有"〔数据确定后填〕"的字段（细胞数、TF 列表、实际可用的时间段）。**不改任何假设、指标、对照或成功标准。**
2. 提交，提交信息写 `prereg: freeze W4 short-time protocol`。把这个提交号写进 `docs/w4_results.md` 开头。
3. 冻结之后才允许 W4-5、W4-7 读取复制集的扰动效应。

---

## W4-3 双流短时框架

**脚本**：`scripts/phase0/w4_shorttime.py`（主程序）、`w4_streams.py`（两条流）、`w4_arms.py`（对照臂）。可以复用 `w3_validation.py` 里已校准的组件（U-only 打乱、统一 log-PCA 导数），但要在新脚本里显式调用，不改原文件。

### 数据流（每个时间段 t→t+Δ，每个留一 TF 折）

1. **拟合（只用时间 t 的源细胞，排除留出 TF）**：基因面板（2000 个 HVG）、S 文库归一化因子、log1p 后的 PCA-50、每个基因的 γ_g（上尾稳态 OLS，和 `a1_gfg_direction_source.py` 的 `fit_gamma_tail` 一致）。拟合后冻结。时间 t+Δ 的任何数据都不参与拟合。
2. **状态流（只吃 S）**：每个细胞 z_i = PCA(log1p(s_i / 文库因子 × 1e4))。
3. **速度流（吃 U+S）**：每个细胞每个基因 v_ig = u_ig − γ_g·s_ig，U 用同一个 S 文库因子归一化。
4. **投影**：ż_i = W · (v_i / (1 + s_i))。W 是 PCA 载荷，s_i 是归一化后的 S，链式法则对应 log1p 的导数。
5. **群体聚合**（每个条件 c、时间 t）：
   - z̄_c：状态均值；
   - **主**：v̄_c^pool，先合并条件 c 所有细胞的 U、S 计数，再按第 3、4 步算（在合并后的 s 上求导数）；
   - 对照：v̄_c^mean，逐细胞 ż_i 的平均；
   - 次要：按状态簇（在训练源上用 k-means 拟合 k=8，冻结）分层的群体速度。
6. **预测器**：ẑ_c,t+Δ = z̄_c,t + A·v̄_c,t + B·[z̄_c,t, e_c] + b。A、B 用岭回归拟合，训练样本是训练 TF 和对照条件；α 从 {0.1, 1, 10, 100, 1000} 中按内层"留一个训练 TF"选择。e_c 默认是现有 ESM2 嵌入，按训练 TF 做 StandardScaler。
7. **目标和指标**：时间 t+Δ 留出 TF 的真实状态均值 z̄_c,t+Δ。坐标用第 1 步冻结的 PCA，对时间 t+Δ 的 S 做同样的归一化和 log1p。指标是 TF 等权的平方误差。

### 对照臂（每个臂的超参数选择预算相同）

| 臂 | 预测器的速度输入 |
|---|---|
| `static` | 无（去掉 A 项） |
| `vel_pool` | v̄_c^pool（**主臂**） |
| `vel_mean` | v̄_c^mean |
| `vel_pool_Uswap` | 把条件 c 的 U 计数换成"时间 t 的源 S 均值最近的另一个条件 c′"的 U（按受体的 U 总量缩放深度），S 不动，再算 v̄^pool |
| `pseudo_S` | 只用 S 推出的伪速度：每个细胞在训练源 S 空间里 30 近邻的均值减自身，再按条件平均。维度和 v̄ 相同 |
| `learned_head` | W3 的 no-JVP 学习版速度头（冻结 innovation），作为次要比较 |
| `empirical_disp` | z̄_c,t − z̄_c,t−1（只在存在 t−1 时）：实测位移，作上限参照 |

### 单元测试（必须全部通过才能运行 W4-4）

1. 投影的雅可比和有限差分一致（相对误差 < 1e-4）。
2. 线性情形下，没有 log 变换时，先合并再算和逐细胞平均结果一致。
3. 泄漏检查：在 t+Δ 数据上加随机扰动，所有拟合产物的哈希不变。
4. `vel_pool_Uswap` 前后 S 完全不变，U 的总量匹配受体。
5. 合成数据正对照：构造"U 领先 S 一步"的模拟数据，`vel_pool` 必须明显优于 `static` 和 `vel_pool_Uswap`；构造"U 由 S 决定"的零情形，`vel_pool` 与 `vel_pool_Uswap` 无差异（各重复 200 次，报告检出率和假阳性率）。

---

## W4-4 开发集运行（RENGE day4→5）

- 留出 4 个开发 TF（LIN28A、NANOG、POU5F1、ZIC3），训练用其余训练 TF 和对照条件。
- 跑全部对照臂，seed 0–3（seed 只影响 k-means 和打乱，不是生物学重复）。
- 只用于检查代码、数值稳定性和运行时间。**结果写进 `docs/w4_results.md`，标"开发集，不作结论"**。
- 发现 bug 可以修，但修完之后**不能**为了让开发集结果变好去改指标、对照或预测器结构，这些都已写进预注册。

---

## W4-5 复制集运行（预注册冻结后）

按预注册第 3、4 节执行：RENGE day2→3、day3→4，所有可用 TF 做留一 TF。

- 先冻结所有臂的预测（保存并记录哈希），再读目标计算指标。
- 结果原样汇报，不论正负。

---

## W4-6 时间跨度衰减

按预注册 H2：同样的流程跑 day2→4、day2→5、day3→5，和相邻跨度（day2→3、day3→4、day4→5）一起，画"相对增益 vs 跨度"曲线（按 TF 做 bootstrap 区间）。

---

## W4-7 外部复制和机制验证

**MIX-seq**（W4-1b 拿到数据才做）：
- 单位是 24 个细胞系；扰动固定为曲美替尼，DMSO 作为对照。
- 每个相邻时间段（3→6、6→12、12→24、24→48 小时）做留一细胞系预测，流程同 W4-3。e_c 换成细胞系在**时间 t 的 DMSO 对照**状态均值，不用外部嵌入。
- 检验预注册 H1（推广）和 H2（衰减）。

**sci-fate**：
- 只有一个扰动，单位是时间点，所以不做留一预测。做"下一时间点群体均值"的预测比较：内含子 U 算出的群体速度，对比代谢标记的实测速度（新合成 RNA 计数减"总量 × 降解项"，具体按论文的速度定义），加上 static 和实测位移。
- 报告两种速度的方向一致性（cos）。
- 回答预注册的 H5：内含子 U 能追回实测速度多少增益。

---

## W4-8 消融

1. **foundation 状态流**（预注册 H4）：状态流换成冻结的 foundation 去噪编码器（`/home/yuchang/GFG_velocity_foundation_mousebrain_v1/experiments/foundation_pilot_20260830/gpu6_denoise_pretrain/pretrain_final.pt`）。
   - 先核实它的输入格式（原始计数还是 moments）和 3,000 基因词表，在 RENGE 上报告基因交集。
   - 投影用编码器对 S 的 JVP。
   - 输入格式无法匹配时，停止此项并汇报原因。
2. **扰动嵌入**：e_c ∈ {无, ESM2（默认）, one-hot, LLM 基因文本嵌入}。LLM 嵌入用公开的 GenePT 类基因嵌入，记录来源和版本；不在线调用 LLM 生成新内容。
3. **直接靶基因子集**：从 ChIP-Atlas 取每个 TF 的直接靶基因（固定阈值，记录版本）。比较速度增益在"直接靶基因"和"其余基因"上的大小。只在复制集主结果出来后做，属于描述性分析。

---

## 3. 汇报格式

```
W4-0: GFG 原版重训 V3/V4/V7 cos（RENGE）=…；（MeRLin）=…
W4-1: 见汇报表；RENGE day4 复处理一致性相关=…
W4-2: 预注册冻结提交号=…
W4-3: 单元测试 n/n 通过；合成正对照检出率=…，零情形假阳性率=…
W4-4（开发）: 各臂 MSE=…（不作结论）
W4-5: H1 主统计量=… [95% CI]；逐时间段=…；逐 TF 胜负=…
W4-6: 相对增益随跨度=…
W4-7: MIX-seq H1/H2=…；sci-fate 内含子速度 vs 实测速度 cos=…，追回比例=…
W4-8: foundation 状态流下增益=…；e_c 消融=…；直接靶基因 vs 其余=…
异常/偏离：…
建议（不执行）：…
```
