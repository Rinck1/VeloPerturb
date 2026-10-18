# VeloRoute 交接入口

更新：2026-10-17。请先读 [当前状态](docs/current/STATUS.md)，再读
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

下一步建议是最终ODE路由机制实验与方法/结果初稿并行；外部基线及直接S+U对照仍暂缓，
独立复现尚未完成。本次只整理文档，没有启动这些实验。

资源约束：仅使用物理GPU0–3，不使用4–7，不停止其他项目进程。
原始数据、模型及优化器检查点留在原机。未跟踪的 `package-lock.json` 与本项目整理无关，未修改。
