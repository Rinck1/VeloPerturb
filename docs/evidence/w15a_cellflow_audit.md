# W15-A CellFlow：CF-0 / CF-1 审计与实现差异

日期：2026-10-21。对应执行规格：`PLANS/w15a_cellflow_kinetic_plugin.md`。
本页是新增实现差异记录，不改写原设计草案或冻结的 W14 / RENGE 训练代码。

## 官方来源与运行环境

官方源码路径：

```text
/data/yuchang/veloroute_kang_20260914/external/CellFlow
```

该目录来自下载归档，不含 `.git`，不能声称已绑定某个上游 commit。
原下载归档 SHA256 为
`2c8f7320295048b04a930f693d2918a53f15e1a22e000879c5436eae307abc25`。
审计另逐个记录 42 个 Python 源文件哈希，未修改官方源码。
发行包 `cellflow-tools` 未全局安装，故不将打包回退版本 0.1.0 当已验证发行版本。

主环境 JAX/JAXlib 0.6.2、Flax 0.11.0、ott-jax 0.5.1、Optax 0.2.5。
旧依赖覆盖层提供 Diffrax 0.6.2，但缺 Optimistix；原 JAX 没有 CUDA 插件。
项目局部新增 Optimistix 0.0.10、jax-cuda12-plugin / pjrt 0.6.2：

```text
/home/yuchang/wangjiaxuan/tools/w15a-cellflow-deps
# above is a symlink to:
/data/yuchang/veloroute_w15a/runtime/python_deps
```

JAX GPU 插件需要 cuDNN ≥9.8，主训练环境为9.5，因此另装私有cuDNN9.8，
不升级全局 PyTorch/JAX 环境：

```text
/data/yuchang/veloroute_w15a/runtime/cuda_deps/nvidia/cudnn/lib
```

运行必须显式设置 `CUDA_VISIBLE_DEVICES`、`XLA_PYTHON_CLIENT_PREALLOCATE=false`，
并把上面的私有库目录放进 `LD_LIBRARY_PATH`。物理 GPU0 的实际张量运算通过，
不是只检查 `jax.devices()`。授权范围本轮扩大到最多六卡，但 L2/L3 合计仍先只用一张。

机器证据：`outputs/w15a_cellflow_audit_v1/summary.json`；完整时间原文副本同目录。

## API 判定

| 能力 | 判定 | 实现 |
|---|---|---|
| 任意源细胞集合 | 原生支持其数值输入；高层 source/control 是布尔角色 | 使用显式源—目标群体包，不把已经受扰动的源强行说成未扰动对照 |
| 额外非池化群体协变量 | 原生支持 | `ConditionEncoder.covariates_not_pooled` |
| 各输入独立 50→64→条件维 MLP | 原生支持 | `layers_before_pool` 的分输入配置；六臂都保留两种插件插槽 |
| 逐源细胞条件 | 速度场原生支持，训练/预测需要包装 | OT `src_ixs` 重采样后同步重排条件；推理按粒子 vmap |
| 指定部分积分终点 | 需要包装 | 原生 OTFM 硬编码 `t1=1`，包装同一个原生 `_base_velocity`，只改变终点 |
| ODE 求解器 | 原生 Tsit5 自适应积分 | 默认 PID rtol=atol=1e-5；没有固定步数 |

最重要的适配问题是：原生 `OTFlowMatching.step_fn` 在重采样源细胞后不重排条件。
对于群体常量条件没有影响，但直接添加逐细胞动力学会使标签错配。
本地 `source_labeled_step` 保留官方 OT、概率路径、MSE、优化和 EMA，只同步重排源标签。
不 patch 官方库。群体条件下与官方单步更新要求逐元素一致。

部分终点包装在 tau=1 时已与原生完整积分逐元素一致；tau=0 返回原始源状态。
逐粒子条件在全部条件相同的情况下也与原生积分一致。

## 生物学时间原点：训练前固定的实现决定

[GEO GSM7548960](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7548960)
将 r4 标为 D3.2、D3.4、D3.6、D3.8；同页培养说明 D3–8 为 RA+SAG。
[作者 A0.0 构建脚本](https://github.com/rorymaizels/Maizels2023aa/blob/main/analysis/A0.0_making_adata_sets.ipynb)
的代码 cell 4 同时定义日级 `D3..D8` 和小时级 `05h/10h/15h/20h`，并把后者读为 r4。
这些原始证据已下载保存并写哈希。

训练前采用：RA+SAG 开始的 D3 为 0h；日级 `Dn → 24(n−3)h`；r4 采用作者
标称 5、10、15、20 小时。后一个绝对原点关系是合并作者元数据后的实现推断，
不是声称作者 notebook 已给出同一数值映射。此决定已在读取新模型成绩前确认。
GEO 小数日对应的4.8小时间隔不替代标称5小时，不按成绩挑选两套单位。

原规格没有写出傅里叶具体频率，现统一固定为 `log1p(hours/1h)` 上的弧度频率
1、2、4、8；八维 sin/cos，对所有臂一致。生物时间只构造条件，不复用 flow tau。

## 数据/token 实现

- 复用 L1 已冻结的训练 D3 源侧 HVG2000/PCA50 及 γ，无新 PCA 或新 GFG/JVP。
- 各群体先对原始 S/U 求和，二者均使用全基因 S 总深度缩放。
- 群体动力学、S摘要、直接U各减去同重复最早源群体的相应投影。
- 局部动力学按源 S-PCA 的64邻居聚合计数，减相同群体参考速度。
- 训练群体 token 采用群体等权标准化；局部 token 仅合并训练源细胞拟合标准化。
- 最早时点参考相减后的原始群体 token 为0，经过训练标准化后是一个常量，
  不一定仍为数值0；其局部细胞 token 不会被强制置0。
- sci-FATE2 单条件，不运行跨条件群体打乱；保留局部条件打乱实现。
- 两个插件槽在 static 中都是常数零，网络结构与其他臂相同；不加 L1 均值锚。
- 单扰动集合大小1，初始开发配置固定官方支持的 mean pooling；这是网络配置，
  不是声称复制论文默认 attention pooling。后续所有臂继承同一 static 选型。

新源侧数据包位置：

```text
/data/yuchang/veloroute_w15a/cellflow_source_pack_v1
```

开发配置：`configs/w15a_cellflow_development_v1.json`。初始列出3个 static
网络配置；在任何真实static训练开始前，经负责人确认增加一个官方默认宽度的
1024×3 hidden / decoder、1024×3 time encoder配置（time_freqs=1024），共4个，
仍低于原规格12配置上限。该配置保留本方案固定的mean pooling及插件槽，不冒充
逐项照搬论文超参数。128×2首先仅作管线pilot，不预先当作充分的强基座。
选型只按r4三个5h窗口的均值Energy选择，所有候选同seed、同512样本。
5/10/15h全部窗口、decay/linear两时钟都报告，不给插件单独调参。
此文件还不是 CF-3 冻结配置，不能据此启动 Traxler 检验。

## L3 数学问题：只记录，不静默改设计

K候选按q加权的Energy是混合经验测度的精确Energy。但是每源只抽一个专家后
计算有限N的V-statistic，再对专家抽样取平均，与它不严格相等。
后者还多出 `N^(-2) Σ_i E_{k,l} ||candidate_ik−candidate_il||` 的源内对角修正。
因此原规格 §11.6.2 的字面等价测试不可能在任意N下成立。
保留加权混合为主要评估对象，并把这一差异报告给负责人；不靠放大蒙特卡洛误差
或偷偷改变主指标掩盖差异。

## 验证进度

基础适配与官方积分测试：11/11通过，`outputs/w15a_adapter_tests_v2.xml`。
随后追加相对时间原点、评分、官方单步训练等价，以及玩具完整训练→序列化→预测冻结→
目标评分和篡改拒绝测试，15/15通过：`outputs/w15a_adapter_tests_v5.xml`。
完整链路测试曾定位到本地状态日志重复计时键，已改为独立fit计时字段；v4失败记录保留。
20正效应+20零效应、三臂的120次官方CellFlow合成拟合在GPU0执行；
v1在第19次正效应后的状态写入被根盘瞬时满盘打断，未进入零效应；并非模型训练异常。
旧记录保留。隔离依赖和本轮自己的下载缓存已可恢复迁往数据盘；
从头重跑全部120拟合的结果路径改为：

```text
/data/yuchang/veloroute_w15a/cellflow_synthetic_v2
```

在任何null分数被查看之前，另写入
`configs/w15a_cellflow_engineering_gate_v1.json`：正效应的两对照配对改善中位数均>0；
若null相对两对照的配对均值改善95% bootstrap区间下界同时>0，视为可疑系统假增益暂停。
这是工程异常验收，不是FPR校准、不是生物学显著检出率。

源侧数据包已完整生成，耗时67.9秒，目标未读。正式 CF-2 需全部前置验证完成并由负责人
审核，随后 `w15a_cellflow_program.py` 只启动一个static候选跑通，不自动扩大调参预算。

### CF-1 完整工程结果与CF-2启动

v2全部120拟合完成。相对两个对照的正效应平均改善为92.645%、中位数93.314%，
20次方向均正；这里的S摘要为相同理论源分布的常量，故Ssum与static分数逐元素相同，
不是两个独立实验。零效应平均改善−3.050%、中位数−0.069%，配对均值改善95%
bootstrap区间分别为static的[−8.222%,1.258%]与Ssum的[−8.101%,1.186%]。
保留全部偶然正向、负向分数，未发生事先定义的系统假增益异常。
这只是已知真值构造下的工程验收，不是实际扰动上的效果，也不是FPR校准。

负责人审阅完整结果后批准CF-2。GPU0已启动四候选顺序队列：

| static候选 | 实测可训练参数 |
|---|---:|
| w128d2，首个管线pilot | 121,138 |
| w256d3 | 433,970 |
| w512d3 | 1,485,106 |
| native_width1024d3，原生默认hidden/time宽度 | 9,594,930 |

每个候选都完整训练、冻结全部六窗×两时钟预测，然后评价。
全部使用seed0和相同512细胞子样本；选择仅看static的三个5h窗口decay Energy均值，
没有读取kin插件成绩。此单seed选型不是正式三seed方法对比。

```text
job: /home/yuchang/wangjiaxuan/outputs/w15a_cellflow_jobs_20261021/static_grid
results: /data/yuchang/veloroute_w15a/cellflow_static_grid_v1
gate: /data/yuchang/veloroute_w15a/cellflow_synthetic_gate_v1/summary.json
```

四候选结束后输出`selection/frozen_l2_configuration.json`和`freeze_record.json`。
后者在负责人提交配置前`freeze_commit=null`，因此不把开发选择完成冒称已经完成CF-3。
L3必须继承被选配置，绑定其官方hidden包装代码哈希，再启动真实数据训练。

首个w128d2 pilot已完整跑通，含5000步训练、全部12份窗口/时钟预测冻结与评价，
总耗时102.7秒。三个5h窗口decay原始Energy分别为0.806957、0.640006、0.484130，
等权均值0.643698；这只是static开发pilot，不是velocity增益或最终基线选型。
队列已自动进入w256d3。CF-1可提交的小记录汇总位于
`outputs/w15a_cf1_completed_20261021_v1/`，仅52KB，不含模型、矩阵、细胞级数组。

## 三seed正式比较前的剩余接线

当前四候选为单seed开发网格。现有`Artifacts`快照禁止多个输入共用basename，
因此直接把三个`prediction_manifest.json`同时传给当前评价入口会触发日志快照冲突。
正式三seed比较前需以唯一的冻结网格索引记录所有manifest及其哈希，并补多seed链路测试；
不需要改变距离公式、预测或训练。为保证当前四候选使用同一评价代码，本轮运行中不改该文件。
