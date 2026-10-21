# VeloRoute 交接入口

更新：2026-10-21。请先读 [当前状态](docs/current/STATUS.md)，再读
[最新结果](docs/current/w13_time_results.md) 和 [方法定义](docs/current/w13_time_model.md)。

最新结果汇总：[Traxler完整分布与单细胞velocity](docs/current/w15k_traxler_results_20261021.md)。
单细胞三套KO内打乱已完成，real vs三打乱+0.486%、vs S+U+0.401%、vs U程序+0.027%；群体三打乱为−0.079%。
独立L2仍在训练，无最终L2/L3结论。40项集中回归通过。大规模CV已获用户要求，正在接入，不能称为已启动。

最新执行：[Traxler参数匹配L2与单细胞三打乱](PLANS/w15k_traxler_l2_cell_20261021.md)。
GPU0 PID753617跑12模型L2（CellFlow256×2，384,594参数，与L3 389,704差1.31%）；
GPU7 PID753879跑6个额外单细胞打乱模型，真实模型复用。均5000步，15项相关测试通过，已实际更新。
输出分别`/data/yuchang/veloroute_w15k_traxler_l2_20261021/matched_l2_v1/`和
`/data/yuchang/veloroute_w15k_cell_shuffle_20261021/extra_shuffles_v1/`；训练后自动评分L2/L3及三打乱。
本轮仍非GFG，测试目标保留；RENGE GPU2/3继续、GPU6保留。独立L2不再是“尚未启动”。

最新资源授权：用户明确使用GPU7，现白名单0–5/7，GPU6仍保留，最多6卡可见。
GPU7已启动独立额外打乱permutation1/2×三seed，共6拟合；GPU0原21拟合队列不重复、不停止。
GPU7 PID509543；状态`/data/yuchang/veloroute_w15j_gpu7_20261021/shuffle_controls_v1/status.json`。
GPU策略9项测试通过，共享launcher未改以保持RENGE哈希锁有效。以下GPU7保留/等待GPU0描述为历史。

最新补充队列：[短时、单细胞/群体与冻结router机制](PLANS/w15i_router_followups_20261021.md)。
用户要求必要实验直接排队，共21新拟合：0→6h四臂12个，6→24h单细胞程序三臂9个，复用3个父基线；
另有父模型7种冻结干预，不重训/不重新积分。GPU0忙于其他项目，CPU等待器PID284893已运行，空闲后自动接续。
状态`/data/yuchang/veloroute_w15i_router_followups_20261021/dispatcher_v1/status.json`。
父四臂已12/12完成：velocity vs S+U配对平均Energy+0.149%、vs打乱+0.025%、vs U程序−0.102%，增益很小。
保留测试集不读取，RENGE继续，GPU6/7不使用。以下父实验“训练中”描述为历史进度。

最新执行：[Traxler四臂完整程序router](PLANS/w15h_traxler_program_router_20261021.md)。
用户要求直接排队，GPU0已启动12模型（四臂×三seed、各5000步），K8独立专家ODE。
共同S+U输入，比较零/velocity程序方向/打乱方向/U程序方向5维附加上下文；无硬cos约束。
56训练KO、7验证KO；7测试KO保留，全部84份验证预测后自动评分。10项相关测试通过。
作业PID34799；状态`/data/yuchang/veloroute_w15h_traxler_program_router_20261021/full_distribution_v1_retry/status.json`。
首启cuDNN冲突于训练前失败，已复用RENGE兼容库重启，旧目录保留。
RENGE GPU2/3不动，GPU6/7保留；本轮尚无性能结果。

最新实验：[扰动后响应程序方向](docs/current/w15g_response_program_results.md)。
6→24h probe在7验证KO上相对S+U有1.82–4.49%改善，但56训练KO内部CV不稳定且均输U程序方向。
物理正方向直接修正几乎无收益，尚未建立方向突破口。
0→6h神经网络小实验也已15/15完成，方向版相对direct-U平均Energy改善0.179%，幅度很小。

最新补充：[方向小实验与Traxler扰动接入](docs/current/w15e_direction_results.md)。
方向试验16/16完成，vs原router+1.121%，vs方向打乱约0，尚无正确方向独立增益。
Traxler132个真实小鼠ESM已生成；0→6h五臂×三seed的开发比较已启动，13测试KO保留。

最新执行覆盖下方较早的队列描述：原RENGE L1的576拟合已经完成；用户要求继续RENGE L2/L3。
2026-10-21已完成32折共享源数据包，并启动独立L2 GPU2（672拟合）、L3 GPU3（576拟合），
控制器PID3121968。31项数据/适配器/队列测试通过，训练完成后按各自全网格冻结并自动评分。
配置与完整路径见[本轮执行协议](PLANS/w15c_renge_l2_l3_20261021.md)。
实时控制器：`outputs/w15c_renge_jobs_20261021/controller/status.json`；
训练状态：`/data/yuchang/veloroute_w15c_renge_20261021/l2_l3_transfer_v1/{L2,L3}/status.json`。
本轮尚无性能结论。新数据Traxler继续保留为后续任务。

当前是解析S/U动力学上下文＋soft router＋专家／小ODE的短时群体预测。
W13的1/2/4已完成，步骤3按要求未做；最新模型并非原版GFG/JVP。
已有同系列阳性，不将其说成逐细胞真实命运恢复或未见遗传扰动泛化。

- [所有文档的分类导航](docs/README.md)
- [当前可用主张与证据](docs/current/CLAIMS.md)
- [已撤回、未建立与已过时主张](docs/corrections/CLAIMS.md)
- [有效阴性和边界结果](docs/evidence/boundaries/README.md)
- [计划与保留的协议路径](PLANS/README.md)
- [旧605行交接全文：仅作历史快照](docs/history/handoffs/HANDOFF_before_organization_20261017.md)
- [研究记录发布索引](docs/history/record_exports/research_archive_20261017.md)

W14的81个训练及主评分已完成：24h平均vs static +2.496%，P1通过、P2未通过，direct-U更好。
Git发布失败曾阻断RENGE；10月21日通过独立恢复控制器重新核验后，原576个作业已在GPU1–3开训。
不修改冻结代码、不重训W14；旧失败日志保留，科学完成与发布状态分离。
RENGE两worker曾因根盘满退出；恢复前51份完整模型/预测哈希有效，成功作业不重跑，GPU2/3已续训。
Traxler18个GEX SRA共143.1GB及9份S/U/A矩阵已完成，76,344细胞；
作者QC/单guide保留集合已精确匹配28,303细胞，HTO作者逐细胞判决尚待恢复。
新数据暂只做技术准备，未解封其扰动效应；CellFlow同系列固定开发比较现已完成。
最终ODE路由干预已完成：群体平均后重新积分保留5/10/15h收益的100.33%/101.43%/64.96%，
支持群体配比定位；见[机制报告](docs/current/w15_l1_routing_results.md)。

用户确认三条模型线：[L1小模型、L2 CellFlow插件、L3 CellFlow＋双流router专家](PLANS/w15a_cellflow_kinetic_plugin.md)。
L2官方GPU、15项软件测试及120次合成通过，四static候选已完成，含约960万参数配置，选中128×2。
L3接入官方隐藏骨干的31项CPU回归通过，50维K8四臂短训练已跑通。
选型科学配置已提交a0cde230，并绑定独立冻结记录；L2六臂×三seed（18模型/216窗口）已完成。
群体kin对static的5/10/15h改善15.87%/32.53%/45.47%，但输S摘要、模式比例TV未改善，
局部kin也输单置换shuffle；见[CellFlow完整结果](docs/current/w15a_l2_results.md)。
这支持群体上下文作用，尚未建立velocity特异性。冻结推理遮蔽也已完成：
将实际群体token换成05h参考常量，5/10h平均Energy反而略好，15h不变；当前动态输入没有稳定必要性。
见[独立干预结果](docs/current/w15a_l2_token_intervention.md)，原L2成绩未改。
CPU仍执行L3完整120拟合合成（60万更新）；L3真实训练/预测/评分入口21项CPU测试通过，
源特征附包也已完成，控制器PID195386已等待合成判决，通过后按固定18训练网格接续GPU0，尚未进入真实训练。
另有CPU机制等待器PID99758；不论合成gate通过与否，完整结束后分析专家几何，不读中途分数。
共样本评价统一到W13原源ID/目标缓存，L1原18项复算Energy最大误差2.92×10⁻¹⁰；原有static开发选型记录保留。
L3正式主干继承L2冻结配置，不是两模型输出后期集成。
用户10月21日新授权最多六卡，允许物理GPU0–5，6–7保留；不停止其他项目进程。
实时恢复入口：`outputs/w15_renge_partial_recovery_20261021/controller/status.json`。
上一层`w14_recovery_20261021_v1/status.json`保留根盘满导致的失败，不能再据此判断当前任务已停。
原始数据、模型及优化器检查点留在原机。未跟踪的 `package-lock.json` 与本项目整理无关，未修改。
