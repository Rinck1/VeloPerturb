# 当前状态：短时群体预测与动力学模式路由

更新：2026-10-21。此页为最新状态入口。

## 最新结果与发布

Traxler单细胞velocity三套KO内打乱已全部完成：相对三套打乱均值Energy改善0.486%，
相对S+U改善0.401%，相对U程序改善0.027%。群体版三套打乱后为−0.079%，没有稳定优势。
[结果与机制汇总](w15k_traxler_results_20261021.md)。独立L2仍在训练，不能提前报告L2/L3胜负。
最近新增实验/队列/GPU策略40项集中回归通过。用户要求大规模CV，当前正在接入，尚非已启动/已完成。

## 最新执行：Traxler独立L2与单细胞三打乱已开训

用户确认开始后，GPU0已启动原生CellFlow L2四臂×三seed，共12拟合；GPU7补训
单细胞版within-KO permutation1/2各三seed，共6拟合。每拟合5000步，自动冻结和评分。
两条任务已实际进入训练更新，不是仅排队。15项相关测试通过。

L2给定与L3相同的源S50、U50、5维单细胞程序、ESM和真实生物时间；不含router/专家。
L2宽度只按参数数目选择为256×2，共384,594参数；L3为389,704，差1.31%，未用预测分数选宽度。
L2最终自动输出同口径`L2_L3_comparisons.csv`；GPU7复用真实velocity/S+U/U程序父模型，只新增打乱臂。

- GPU0启动器PID753617，日志`outputs/w15k_traxler_jobs_20261021/l2_gpu0/`；
  输出`/data/yuchang/veloroute_w15k_traxler_l2_20261021/matched_l2_v1/`。
- GPU7启动器PID753879，日志`outputs/w15k_traxler_jobs_20261021/cell_shuffle_gpu7/`；
  输出`/data/yuchang/veloroute_w15k_cell_shuffle_20261021/extra_shuffles_v1/`。

本轮为Traxler 6→24h的18h early-response开发比较，7测试KO目标不读取；仍为解析S/U速度，未启动GFG重训。
RENGE GPU2/3与共享冻结源码不变，GPU6保留。
[固定配置、数学口径与完整路径](../../PLANS/w15k_traxler_l2_cell_20261021.md)。
下方“Traxler独立L2尚未开跑/仅群体打乱”描述已由本段新执行覆盖；本轮尚无最终性能结果。

## 最新资源更新：GPU7已授权并启动独立对照

用户明确要求使用GPU7，白名单现允许物理0–5和7；GPU6仍保留，单进程最多可见6卡。
GPU0原补充队列已实际运行，继续其短时/单细胞实验，不拆已开始的模型。
GPU7独立补训6→24h的两套额外固定打乱（permutation1/2，各seed0/1/2，共6新拟合），
复用父实验真实velocity/S+U/U程序模型，最终与原permutation0等权汇总，不挑选最佳打乱。
这是在0.025%小差距下补充随机对照稳健性，不改变既有模型和实验预算。

GPU7启动器PID509543，日志`outputs/w15j_gpu7_jobs_20261021/shuffle_controls/`；
输出`/data/yuchang/veloroute_w15j_gpu7_20261021/shuffle_controls_v1/`。
配置`configs/w15j_gpu7_shuffle_controls_20261021_v1.json`；GPU白名单9项测试通过。
共享`launch_gpu_job.py`受RENGE哈希锁约束，未修改；新GPU7任务独立记录最新资源授权。
下文GPU6/7均保留的描述为历史状态，现仅GPU6保留。

## 最新：必要补充实验已进入后台队列

父实验Traxler 6→24h四臂12/12完成，84份预测及评分已落盘。
velocity程序版Energy=0.156835，S+U=0.157121，打乱=0.156879，U程序=0.156778。
等KO/seed配对相对增益分别为+0.149%（vs S+U）、+0.025%（vs打乱）、−0.102%（vs U程序），增益很小。
不扩参数；按用户“需要跑的直接排队”补充三个缺口：

- 冻结父模型的velocity关闭/反向/跨KO打乱、群体/均匀q及U特征关闭；0训练、0额外ODE积分。
- 0→6h同口径四臂×三seed，12个新训练；不是重复此前cos先验pilot。
- 6→24h单细胞程序velocity/KO内部细胞打乱/U程序，三臂×三seed，9个新训练；复用父实验3个S+U基线。

共21新拟合，保持K8、5000步；只用开发集，不读取测试目标。
14项队列/特征/基线复用/实际OT与ODE相关测试通过。
GPU0已被其他项目占用，CPU等待器PID284893已启动，GPU0空闲后自动接续，不抢占其他任务。
等待器：`outputs/w15i_router_followup_jobs_20261021/controller/`；
实时状态：`/data/yuchang/veloroute_w15i_router_followups_20261021/dispatcher_v1/status.json`。
正式训练输出：`/data/yuchang/veloroute_w15i_router_followups_20261021/queue_v1/`。
GPU2/3的RENGE继续，GPU6/7保留。[补充队列与数据完整路径](../../PLANS/w15i_router_followups_20261021.md)。

## 最新执行：Traxler S+U 完整程序 router 已排队

用户要求直接排队训练，GPU0已启动四臂×三seed，共12模型，每模型5000步。
S+U为共同输入，额外5槽分别为零、群体velocity响应程序方向、跨KO打乱方向、U响应程序方向。
K=8、官方CellFlow隐藏骨干、固定源侧q、各专家独立Tsit5 ODE；不使用硬方向cos约束或均值锚。
这是6h→24h的18h early-response完整分布开发实验，不是重复ridge均值probe。
56训练KO、7验证KO；7测试KO目标保留。全84份验证预测冻结后自动评分Energy、均值、方差和路由统计。
10项相关CPU测试通过，含实际OT训练更新、相同初始参数、保存恢复及独立专家ODE。

训练作业PID34799，日志`outputs/w15h_program_router_jobs_20261021/router_gpu0_retry/`；
输出`/data/yuchang/veloroute_w15h_traxler_program_router_20261021/full_distribution_v1_retry/`。
首启在训练前因系统cuDNN版本冲突失败，现复用RENGE兼容库重启；失败目录保留，模型配置不变。
GPU2/3的RENGE L2/L3继续，其他任务不停止，GPU6/7不使用。
[固定协议与完整路径](../../PLANS/w15h_traxler_program_router_20261021.md)。目前尚无本轮性能结果。

## 最新：响应程序方向小实验已完成

Traxler 6→24h的群体均值probe：56训练KO定义5条响应轴，7验证KO，7测试KO保留。
三个固定正则下，加入velocity方向相对S+U改善3.91%/4.49%/1.82%，但均输U程序方向。
追加56训练KO的5折开发CV后，增益为6.04%/1.37%/−0.81%；整体误差最低的配置方向略差。
物理正方向修正几乎无收益，尚未建立稳定突破口，不据此扩大方向约束模型。
见[完整结果、方法和路径](w15g_response_program_results.md)。

此前Traxler 0→6h五臂×三seed已15/15完成：方向版相对static平均Energy改善0.867%，
相对原router+0.086%、方向打乱+0.109%、direct-U+0.179%，均为小幅描述性变化。
完整结果位于`/data/yuchang/veloroute_w15f_traxler_direction_20261021/pilot_v1/summary.json`。
下方“已启动/训练中”是历史状态，以上最新完成记录为准。

## 最新：方向小实验与 Traxler 开发接入

RENGE ETS2两个24h区间的方向小实验16/16完成：方向版相对原router平均+1.121%，
相对方向打乱−0.005%，相对direct-U−5.763%；正确方向增量尚未建立。
组内速度方向非常一致（随机配对cos约0.99），冻结反向偏置检查也无跨区间一致优势。

Traxler132/132真实小鼠ESM2-3B条件已生成，已启动0→6h五臂×三seed的15个短训练：
106训练KO、13验证KO，13测试KO保留。生物时间增量6h；所有验证预测冻结后自动评分。
Traxler训练目标已经用于拟合，现属开发数据，较早“未解封全部效应”的描述已过时。
完整数值、配置及路径见[方向与扰动接入报告](w15e_direction_results.md)。

## 当前执行：RENGE L2/L3

原RENGE L1的576次拟合及评分已完成，见[RENGE结果](w14b_renge_results.md)。
按用户要求，L2官方CellFlow群体插件与L3官方CellFlow双流router专家已接入相同32折：
16个留出TF、day2→3及day3→4、各三seed。共享源数据包32/32完成，约85MB。
数据/适配器/队列31项测试通过，含实际官方OT更新、模型恢复和独立ODE预测。

2026-10-21启动GPU2上的L2共672拟合、GPU3上的L3共576拟合；固定5000步，继承已选128×2配置。
每条线完成后自动冻结全网格、按相同源ID与目标抽样计算Energy。当前没有本轮性能结果。
物理GPU0/1/4/5由其他任务占用，6/7保留。

首个正式拟合均已完成训练、ODE预测和哈希核验，并自动进入seed1。
真实2560维ESM条件下，L2参数203,026、首fit约62秒；L3参数389,384、首fit约63秒。
按首fit速率粗估训练约10–12小时，尚不含最终评分且后续编译时间会变化。

[执行协议与完整路径](../../PLANS/w15c_renge_l2_l3_20261021.md)。
控制器PID3121968；小型日志`outputs/w15c_renge_jobs_20261021/`，
实时状态`/data/yuchang/veloroute_w15c_renge_20261021/l2_l3_transfer_v1/{L2,L3}/status.json`。
下文的旧恢复/等待记录保留历史背景，以本段最新执行状态为准。

## 项目定位

利用源侧RNA动力学上下文校准混合响应模式配比，预测短时群体状态分布。
主数据是sci-FATE2，训练r1–r3的日级转移，r4小时序列用于同系列开发。
当前不是独立复制结果，也不是未见遗传扰动泛化。

## 工作模型

解析邻域U−γS及log-PCA投影提供上下文，soft router给出固定源侧混合权重；
8个专家沿独立轨迹积分。群体中心化保持冻结均值，ODE为群体耦合而非独立分批积分。
端点参考76,248参数，小ODE76,760参数。训练最大6h RK4步长，推理建议最大3h。
当前没有原版GFG、JVP、EM老师、门控或可训练状态编码器的新增主线实验。

## 已完成结果

W13按用户要求完成1/2/4、跳过3，共27次正式训练；相关模型测试39项通过。

| 模型与视野 | Energy ↓ | vs 同类static | vs shuffled kinetics |
|---|---:|---:|---:|
| b1端点36k，5h | 0.595683 | +4.137% | +4.606% |
| ODE，5h | 0.589504 | +3.723% | +4.620% |
| ODE，10h | 1.027032 | +6.358% | +7.418% |
| ODE，15h | 1.007833 | +2.415% | +1.573% |

ODE相对匹配端点时间版，在5/10/15h再改善1.226%/2.821%/0.027%。
ODE也改善static，所以未进一步扩大动力学相对static的差距。15h没有实质积分增益。
各视野含3/2/1个重叠窗口，不是六个独立生物学重复。“static”只指局部路由，均值分支各臂共享。

此前W5解析均值MSE相对其static均值模型改善11.615%；它与上述分布energy增益是不同指标，不能相加。

[完整结果](w13_time_results.md) · [机器汇总](../../outputs/w13_time_program_v1/final_report/RESULTS.md)

## 机制证据属于哪一版

W10–W12端点模型的冻结干预显示，群体平均q保留大部分收益；S-only蒸馏丢失的收益可由源侧
老师群体模式质量恢复一大部分。支持“群体配比上下文”的定位，不证明单细胞真实命运。
10月21日已完成最终ODE同组干预：全42份预测先冻结，原18项Energy复现差为0。
群体平均q后重新积分，5/10/15h保留原收益100.33%/101.43%/64.96%；均匀q后
5h只保留29.38%，10/15h相对static变差。支持5–10h主要是群体配比校准，而非必须逐细胞路由。
[最终ODE路由机制结果](w15_l1_routing_results.md)。

## W14 已完成与后续恢复

W14的81个36k训练、405份逐窗口预测和主评分已于10月18日完成；冻结配置和模型没有改动。
24h内部跨重复的平均相对static增益为2.496%，P1通过、P2未通过，判为部分确认。
真实动力学没有稳定胜过打乱，direct-U的全局Energy为0.15097，优于real的0.15655。
这是24h内部验证，不是独立确认，也不能直接否定5–15h开发结果。

原控制器在Git忽略outputs时发布失败，误将依赖置为失败，RENGE因此未启动；
10月21日以新控制器核验完整预测后恢复。原失败记录保留，科学完成状态与发布状态分离。
W14主结果、补充统计及Traxler技术结果已补推；不重训、不改变原P1/P2。

RENGE原32折、576个12k作业已在物理GPU1–3实际训练，结束后仍先冻结全网格再评分。
根盘满打断worker1/2，恢复前审计确认51份完整模型/预测哈希有效，worker0仍运行；
worker1/2已用独立恢复包装在GPU2/3重启，跳过成功作业，两个无检查点半成品逐文件哈希后移到data盘保留。
恢复只改变调度，不改变原模型、条件留出、源特征、均值锚、训练预算和评价协议。

Traxler18个GEX SRA共143.1GB已完整下载，9个文库的S/U/A矩阵已生成，共76,344细胞。
作者文库总量与S计数总量的相关性约0.9985–0.9992，是技术对齐而非生物效应结果。
原guide调用只是provisional，已从作者附表技术列精确恢复28,303个作者QC/单guide保留细胞。
0h/6h各至少30细胞的KO有132个；HTO逐细胞作者判决尚未恢复，a/b/c是独立解冻来源而非供体。
不在新数据上读取KO效应或训练预测模型。
[作者技术标签与来源审计](../evidence/data_audits/w15_traxler_demux.md)。

[W14结果](w14_results.md) · [补充统计](w14_supplementary_results.md) ·
[Traxler定量](w14b_su_results.md) ·
[恢复控制器实时状态](../../outputs/w15_renge_partial_recovery_20261021/controller/status.json) ·
[RENGE队列](../../outputs/w14b_renge_router_v1/queue_status.json)

## 三条模型线并行

用户已确认三条线全部保留；新推送计划为[W15-A](../../PLANS/w15a_cellflow_kinetic_plugin.md)。

| 路线 | 结构和问题 | 当前执行 |
|---|---|---|
| L1 小模型 | 解析动力学上下文＋router＋专家／小ODE，检验最小实现与群体机制 | 最终ODE干预完成；RENGE迁移网格续排 |
| L2 CellFlow插件 | 动力学群体token进入CellFlow条件编码器，与S摘要、direct-U、打乱等同结构对照 | 固定18模型/216窗口已完成并统一评分；群体kin胜static，但未胜S摘要，模式比例误差未改善 |
| L3 合并模型 | CellFlow状态骨干＋K8专家＋独立动力学编码器＋固定源侧router | 120次固定合成仍在CPU运行；真实训练/预测/统一评分入口21项CPU测试通过，等待合成判决 |

L2/L3没有L1的解析均值锚，分别报告本线的static对照。L3不是预测结果的后期加权集成；
状态流只读S，解析动力学经可学习编码器进入router，q在轨迹上固定，没有原版GFG/JVP。
L2主配置只用cf_static选型，L3不另调主干；合成和结构测试完成后才启动对应正式开发网格。

多agent分别推进CellFlow冻结干预、L3真实管线和合成机制分析，主agent负责L1、源侧特征准备、恢复调度与交叉审核。
[群体分支可行性计划](../../PLANS/w15b_population_branching.md)保留，尚未越过门槛训练BR-3。

[CellFlow实现与合成证据](../evidence/w15a_cellflow_audit.md) ·
[合并架构与数学检查](../evidence/w15a_l3_implementation.md)

用户于10月21日将资源上限扩大为六卡：允许物理GPU0–5，GPU6–7保留。
RENGE继续使用GPU1–3；CellFlow固定对照网格已释放GPU0，L3合成在CPU并行；其余卡不干预其他项目作业。
六卡是总上限，不表示所有卡或三条正式网格已经同时开训。

CellFlow合成工程检验20正效应+20零效应、每次三臂全部完成。正效应平均改善92.65%，
零效应平均−3.05%、95%区间跨0，工程验收通过。这只是已知真值构造的接口验证，不是生物数据效果。
四static候选为12.1万/43.4万/148.5万/959.5万参数，最后一个含官方默认宽度；
均已完成，按static开发Energy选择128×2；四候选均值依次0.643698/0.673767/0.656273/0.685502。
插件不另调参；固定CellFlow开发比较已完成，尚非独立外部确认。[选型记录](w15a_static_selection.md)。
跨线比较已实现共用W13原源细胞ID与目标缓存的独立评价入口，8项CPU测试通过；
后续网格使用该入口，不直接横比不同抽样的历史分数。注册网格连同共样本入口13项测试通过。
L2网格为static/kin/Ssum/directU/kin_cell/kin_cell_shuf，各训练seed0/1/2；
复用已完成static0，只新增17次训练，全216窗口已冻结并完成统一评分。
局部shuffle固定一个perm_seed0；三个训练seed不是三个置换，不声称完成本单条件数据不支持的群体CF-H3。

L3完整开发基座为307,496参数；固定20正+20零、每重复三臂、每臂5000更新，合计60万更新。
合成仍在运行，未完成前不判通过。真实L3的源特征附包已完成：15个训练源群体44,222细胞、
3个开发源群体13,636细胞，原S状态、动力学与细胞ID逐位保留，补充局部directU及当前源U干预。
真实网格为static/kin/directU/三种kin-shuffle各三seed，共18训练；加上冻结干预共30份预测清单、360个窗口。
完整合成通过后才启动，不因L2结果改损失或选参。[真实管线](../evidence/w15a_l3_real_pipeline.md)。

### CellFlow完整开发结果

三seed等权混合、同W13源ID和目标缓存下，群体kin相对本线static的5/10/15h Energy改善为
15.87%/32.53%/45.47%；但相对S摘要为−3.76%/−7.94%/−2.57%，相对直接U为−4.33%/−6.81%/+1.54%。
局部kin三seed均输固定的单置换shuffle；模式比例TV也未改善，5h为0.08898，对照static为0.06359。
因此目前支持群体上下文有用，不支持velocity特异性或多峰比例优势。

05h起点的群体特征按定义与自身参考相减为0，不能把该起点的改善解读为部署时当前U的新增信息。
冻结权重token遮蔽现已完成：72份新推理与36份原预测先冻结，原Energy复现误差0。
替换为训练均值使5/10/15h误差增加26.24%/62.66%/123.69%；但换成固定05h参考常量，
5/10h误差反而降低0.27%/4.83%，15h不变。10→15h单窗确实变差11.35%，其它非参考起点窗反而改善。
推理输入路径有效，但当前实际动力学token相对常量参考没有稳定优势，不能把主收益归给实时动力学信息。
[冻结输入干预](w15a_l2_token_intervention.md)。
[L2完整结果与L1对齐](w15a_l2_results.md)。L1原18项用共用评分复算最大差2.92×10⁻¹⁰。

L3自动控制器已启动（PID195386，最长24h），等待完整120-fit合成判决；通过后才在GPU0执行18训练和360窗口。
另有CPU等待器在合成完整结束后分析专家几何，即使合成判据失败也会保存机制描述。
新增接线联合CPU回归46项通过；token遮蔽另6项CPU测试通过。
[启动记录](../../outputs/w15a_execution_20261021_v1/RESULTS.md)。
[L2运行状态](../../outputs/w15a_cellflow_jobs_20261021/registered_l2_grid/status.json) ·
[L3合成设计和命令](../evidence/w15a_l3_population_synthetic_design.md)

## 后续未完成

1. L3固定20+20合成完成后自动判断是否启动真实数据训练，并分析专家几何是否分化；不自动调参重试。
2. 方法/现有结果初稿与机制图并行。旧Velocity Horizon提纲已移入历史，不再作为当前提纲。
3. CellFlow已完成本数据固定开发比较；其它外部强基线尚无正式结果。直接U/非线性S+U对照已完成W14网格。
4. 固定方案的独立动态系统或新生物重复复现尚未完成。

2026-10-17目录整理本身没有训练；上述W14/W14-B任务是10月18日新的授权执行。

## 关键路径

- 方法：`scripts/phase0/w13_time_model.py`
- 训练：`scripts/phase0/w13_time_training.py`
- 评估：`scripts/phase0/w13_frozen_horizons.py`
- ODE权重：`outputs/w13_time_program_v1/time_training/training/ode/velocity_router/seed{0,1,2}/model_step4000.pt`（原机）
- b1参考：`outputs/w13_scifate2_direct_continuation_v1/`；原定与步长复算结果分别保留。
- 当前资源上限为物理GPU0–5，6–7保留，不停止其他项目进程；原冻结实验仍沿用各自明确分配。
