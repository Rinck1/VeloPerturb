# L3 群体组成可学性：20 正效应 + 20 零效应合成计划

日期：2026-10-21。独立 runner：`scripts/phase0/w15a_cellflow_moe_synthetic.py`；最小 CPU 测试：`tests/test_w15a_moe_synthetic.py`。不修改已提交的 L3 core 或损失。

## 这项实验回答什么

源侧 S 分布相同，而源动力学 κ 的群体组成不同时，官方 CellFlow hidden + E_v + router + K8 专家能否学会对应的目标群体模式比例？

这是**群体组成可学性**检验，不是逐细胞命运识别，也不是真实 RNA velocity 估计器有效性的证据。κ 是已知机制的合成输入，任何生成/评分真值都不进入配对、模型条件或训练损失。

## 运行前固定的构造

- 每个重复的所有训练群体共用完全相同的一份源 S-PCA50 经验分布，测试群体另用一份相同分布生成的源细胞。群体ID不进入模型。
- 源状态每坐标标准差0.1。κ 的首坐标有 ±1 两成分，其余为0.01标准差噪声；成分分配独立于源 S。
- 训练群体 κ 正成分比例为0.1/0.3/0.7/0.9；测试为0.2/0.4/0.6/0.8。使用整数细胞数最近取整。
- positive：目标首坐标为±2双模式，噪声标准差0.1，目标正模式比例等于该源群体 κ 正成分比例。
- null：目标正模式比例固定0.5，与源 κ 组成无关。
- 训练始终使用官方 **S-only Sinkhorn OT + sample_joint**；κ 和源条件随源索引同步重排。源和目标独立抽样，不用已知命运配对。
- 三臂为 `moe_static / moe_kin / moe_kin_shuf`，同seed相同初始权重、群体/细胞采样和优化预算；shuffle仅在对应源群体内置换。
- **组内shuffle保留κ组成**，且构造中 κ 与源 S 独立，因此它在总体上与真实κ等价。不要求真实臂胜过shuffle；完整报告其结果，作为等价性检查。这里不硬加没有独立物理意义的 direct-U 合成臂，四臂等参已由独立工程测试覆盖。

## 无逐细胞配对的可学性局限

同一群体内部，κ与被OT挑出的目标细胞没有观测对应；可利用的是**跨群体 κ 组成与目标模式比例的关联**。即使positive构造有这种群体规律，当前期望专家MSE仍允许所有专家相同、路由仅形式上均衡。故positive也可能失败。不会因此添加分离正则、使用真值配对、换初始化或反复调整损失。

## 固定基座、预算与判据

基座科学配置：`configs/w15a_cellflow_selected_20261021_v1.json`；L2冻结提交 `a0cde23036737c06a5d2dc7747bc1f1379083573`，配置 SHA256 `6ba3dc12bf2bb200a205c4181a8a4bf3b31a510aba4cfa10ace7a76da870845b`。独立冻结记录：`configs/w15a_l2_freeze_20261021_v1.json`。

- 官方骨干128×2，K=8，E_v 50→64→16，专家宽128，router宽64。当前完整L3为 **307,496参数**。
- 每臂5000更新，Adam lr=3e-4，batch128；官方OT ε=0.1、最多200迭代；EMA按冻结配置继承。
- 20个positive重复 + 20个null重复，seed0–19，每个重复3臂，共120拟合、600,000更新。
- 每群体512源/目标细胞；预测每专家独立 Tsit5，rtol=atol=1e-5，τ_end=1。q只在源端计算一次。
- 主评分是精确加权经验混合V-statistic Energy，各测试群体等权；另报模式比例TV、均值MSE、有效K、群体平均q以及E_v/router任务梯度。精确Energy用分块距离，避免构建 `[4096,4096,50]` 临时数组。
- 工程可学性门：positive中 real 相对static的重复级相对Energy改善中位数>0；null中同一改善均值的1000次重复级bootstrap95%区间下界≤0。shuffle不作为优越性门。不是生物学显著性或FPR校准。
- 全部重复、全部臂、全部测试群体原样保存；不挑成功seed。

运行前计划已落盘并绑定源码/配置哈希：

```text
/data/yuchang/veloroute_w15a/l3_population_frozen_plan_v1/plan.json
/data/yuchang/veloroute_w15a/l3_population_frozen_plan_v1/plan_manifest.json
```

## 已完成的小型验证与耗时估计

- 最小CPU测试7/7通过，37.15秒，记录 `/data/yuchang/veloroute_w15a/l3_population_runner_tests_v1/pytest.xml`。其中端到端smoke仅1个positive + 1个null重复、各臂1步、16细胞群体，明确不能通过完整20+20门。
- CPU25步 profile 没有目标评分：首步含编译1.34秒，暖态每步6.23毫秒。完整600,000更新净训练约1.04小时，尚不含Python采样、480个测试群体的K8/Tsit5预测、精确Energy和I/O。完整CPU估计预留 **2–4小时**，不是GPU计时。
- profile：`/data/yuchang/veloroute_w15a/l3_population_cpu_profile_v1/summary.json`。
- root已批准CPU后台完整固定网格；不占GPU0/1–3，不进行L3真实数据训练。未完成全部120拟合前不报告工程门通过。

命令（完整网格明确确认，CPU限定和依赖/cache在数据盘）：

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
TMPDIR=/data/yuchang/veloroute_w15a/runtime/tmp \
JAX_COMPILATION_CACHE_DIR=/data/yuchang/veloroute_w15a/runtime/jax_compilation_cache_l3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src \
.venv-gpu/bin/python scripts/phase0/launch_gpu_job.py --cpu \
  --script scripts/phase0/w15a_cellflow_moe_synthetic.py \
  --output outputs/w15a_jobs_20261021/l3_population_cpu_v1 -- \
  run --output /data/yuchang/veloroute_w15a/l3_population_frozen_plan_v1 \
  --acknowledge-full-run
```

所有模型和数组留在数据盘；仓库小记录见 `outputs/w15a_l3_population_preflight_20261021_v1/`。失败停下并保留状态，不调参重抽求阳性。
