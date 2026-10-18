# 研究记录发布索引（2026-10-17）

> 归档状态：历史发布清单；其文件路径和哈希对应发布当时，不代表当前文档布局。 [当前状态](../../current/STATUS.md) · [撤回与更正清单](../../corrections/CLAIMS.md)

本次按用户“把所有记录上传GitHub”的要求，补齐远端尚未收到的研究记录。
发布范围是本地可访问的W5–W13实验文档、汇总表、训练历史、运行日志、测试、
图表、来源/配置哈希与逐次运行代码快照，包含阳性、阴性和数值诊断。
更早已跟踪的历史和原本地提交均保留；不声称包含完整聊天逐字稿或远端服务器上
未在本机保存的原始历史。

## 从这里阅读

- [最新交接](../../../HANDOFF.md)
- [W13：精确距离、六时间窗与真正ODE](../../current/w13_time_results.md)
- [W13最终机器汇总](../../../outputs/w13_time_program_v1/final_report/RESULTS.md)
- [W13全部视野指标](../../../outputs/w13_time_program_v1/final_report/all_horizon_summary.csv)
- [W13 ODE数值步长对比](../../../outputs/w13_time_program_v1/final_report/refined_ode_comparisons.csv)
- [W13模型数学定义](../../current/w13_time_model.md)
- [项目定位与下一步讨论记录](../../current/w13_project_position_and_next_steps.md)
- [W10–W12完整结果与机制](../../evidence/mainline/w10_recipe_bridge_results.md)
- [W9容量实验审计和完成结果](../../evidence/mainline/w9_large_model_audit.md)

## 全量记录清单

[research_archive_20261017.json](../../research_archive_20261017.json)逐项记录原路径、
字节数、SHA-256，共4,881份记录，68,384,875字节。快照内容保持原样，不改写
实验时保存的路径、代码哈希或研究结果。代码快照中的旧状态是当时的历史记录，
当前状态以最新交接和结果文档为准。

清单中的原路径直接对应仓库路径，不另复制一套不透明压缩包。
本次新增归档排除了1,494个NPZ数组和591个模型/优化器PT文件；它们保留在原机。
本次不新增原始S/U矩阵、单细胞预测数组、模型权重、凭据、环境目录或无关package-lock。
此前本地提交中已跟踪的小型聚合NPZ、归一化/PCA变换与公开源参考投影随原提交历史保留，
不属于本次新增记录清单，也不是公开原始测序计数。

## 校验与解释

- SHA-256可逐项核对已发布记录的字节完整性。
- 含`.npz`/`.pt`文件引用的manifest不表示数组或权重也已上传。
- GitHub检出目录不同会使原机器绝对路径不可直接运行；原路径保留用于来源追踪。
- 本轮W13有27次正式训练和最终39项针对性测试通过；机制实验的下一步建议尚未执行。
- 全部当前阳性仍属于sci-FATE2同系列开发，不标作独立复制或未见遗传扰动泛化。

导出工具为 [export_research_archive.py](../../../scripts/phase0/export_research_archive.py)。
