# VeloRoute 交接入口

更新：2026-10-18。请先读 [当前状态](docs/current/STATUS.md)，再读
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

W14固定端点的24h内部跨重复验证现已启动，direct-U/nonlinear-SU对照已纳入；其后排队一次RENGE探索迁移。
Traxler发布矩阵已获取，18个GEX SRA共143.1GB下载并行。最终ODE机制干预、外部强基线仍未执行。
W14失败不能直接否定此前5–15h开发结果；新数据暂只做技术准备，未解封其扰动效应。

本轮仅使用物理GPU1–3，GPU0及4–7不动，不停止其他项目进程。
原始数据、模型及优化器检查点留在原机。未跟踪的 `package-lock.json` 与本项目整理无关，未修改。
