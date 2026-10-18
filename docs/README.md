# 研究文档导航

本目录按“是否仍有效”和“是否属于当前主线”分别整理，而不是把所有失败实验都归为错误。
新读者只需从 [当前状态](current/STATUS.md) 开始。

| 目录 | 放什么 | 怎么使用 |
|---|---|---|
| [current/](current/README.md) | 当前定位、方法、最新结果、可引用主张 | 写初稿与安排下一步的入口 |
| [evidence/mainline/](evidence/mainline/README.md) | W5–W12的有效前期主线证据 | 支撑当前模型，不照抄旧阶段待办 |
| [evidence/boundaries/](evidence/boundaries/README.md) | 有效阴性、边界与条件性结果 | 不能删除，也不当作实现错误 |
| [evidence/data_audits/](evidence/data_audits/README.md) | 数据来源、定量、对齐与时间标签审计 | 检查数据身份和适用范围 |
| [corrections/](corrections/README.md) | 代码/统计问题和撤回主张的明确清单 | 防止旧错误继续进入论文 |
| [history/](history/README.md) | 旧GFG/EM架构、MeRLin等探索、已过时提纲和旧首页 | 历史证据，不是当前执行指令 |

## 当前读这四份即可

1. [状态与下一步](current/STATUS.md)
2. [方法](current/w13_time_model.md)
3. [W13结果](current/w13_time_results.md)
4. [可引用主张与证据](current/CLAIMS.md)

## 保留原则

- 没有删除研究报告、实验记录或阴性数字。
- `outputs/` 和4,881份记录的哈希保持不变；不去重或重写原代码快照。
- 分类后的报告只新增状态说明并修正路径引用，科学数值不因整理修改。
- “历史报告含撤回推断”不表示其全部数值作废；逐条以[更正清单](corrections/CLAIMS.md)判断。
- 旧入口中少量文件仅作兼容跳转，避免破坏之前发出的链接。
- 程序绑定的9份协议保留原路径和字节，见 [PLANS导航](../PLANS/README.md)。

[文档迁移清单与原始哈希](organization_manifest_20261017.json)

整理验收：51份文档分类迁移，44项针对性测试通过；4,881份归档文件逐项哈希一致。
见 [整理校验结果](organization_validation_20261017.json)。此次未启动训练或改动模型算法。
