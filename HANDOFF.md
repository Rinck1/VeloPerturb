# VeloRoute 交接入口

更新：2026-10-21。请先读 [当前状态](docs/current/STATUS.md)，再读
[最新结果](docs/current/w13_time_results.md) 和 [方法定义](docs/current/w13_time_model.md)。

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
新数据暂只做技术准备，未解封其扰动效应；外部强基线尚无正式成绩。
最终ODE路由干预已完成：群体平均后重新积分保留5/10/15h收益的100.33%/101.43%/64.96%，
支持群体配比定位；见[机制报告](docs/current/w15_l1_routing_results.md)。

用户确认三条模型线：[L1小模型、L2 CellFlow插件、L3 CellFlow＋双流router专家](PLANS/w15a_cellflow_kinetic_plugin.md)。
L2官方GPU、15项软件测试及120次合成通过，GPU0已启动四候选static开发选型，含约960万参数配置。
L3接入官方隐藏骨干的31项CPU回归通过，50维K8四臂短训练跑通；仍待正式合成和配置冻结。
L3正式主干继承L2冻结配置，不是两模型输出后期集成。
用户10月21日新授权最多六卡，允许物理GPU0–5，6–7保留；不停止其他项目进程。
实时恢复入口：`outputs/w14_recovery_20261021_v1/status.json`，不是原失败的W14发布控制器状态。
原始数据、模型及优化器检查点留在原机。未跟踪的 `package-lock.json` 与本项目整理无关，未修改。
