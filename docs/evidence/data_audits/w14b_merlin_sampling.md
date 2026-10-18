# W14-B4 MeRLin采样时间核查

2026-10-18。读取已在本机的补充表4与2026正式论文Methods，没有新表达分析或训练。

| 数据 | 采样 | 间隔 | 能否承担当前小时级主任务 |
|---|---|---|---|
| 体内scRNA | Day0、Day21、Day91终点 | 504h、1680h | 否 |
| 体内bulk | Day0、Day21、Day57、Day91 | 天级；Day57不是该匹配scRNA点 | 否 |
| 体外对照/药物 | 处理一周后采样 | 168h处理窗口，不是小时级序列 | 否 |

原本机表4中，scRNA过滤后的总细胞/有可靠barcode的细胞/唯一barcode分别为：

| 样本 | 总细胞 | barcode可靠细胞 | 唯一barcode |
|---|---:|---:|---:|
| 体外CTRL | 11889 | 7375 | 4013 |
| 体外BRAFi/MEKi | 16086 | 5031 | 958 |
| 体内Day0 | 8069 | 4055 | 1127 |
| 体内Day21 | 3613 | 2054 | 581 |
| 体内终点 | 6758 | 2565 | 261 |

这些是原发布表的群体计数，不是本项目重新QC后的细胞数或跨时间共享克隆数。
体内scRNA为多个肿瘤混池，不把混池中细胞当独立小鼠重复。

结论：本轮核查的MeRLin主实验没有可用小时级相邻scRNA采样，按W14-B计划不再启动新的MeRLin训练。
这不否定它的长时克隆任务价值，也不是所有MeRLin衍生实验的穷尽性结论。

[正式论文与Methods](https://link.springer.com/article/10.1186/s12943-026-02622-9)。
机器计数与原表哈希：`outputs/w14b_metadata_audits_v1/merlin_sampling.json`、`merlin_published_counts.csv`。
