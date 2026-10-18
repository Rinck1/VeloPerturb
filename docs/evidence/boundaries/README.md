# 有效阴性、边界与混合结果

这里放的是研究结果，不是“做错的实验”回收站。
具体实现/统计错误及不能继续使用的推断见 [更正清单](../../corrections/CLAIMS.md)。

| 结果 | 可以说明什么 | 不能说明什么 |
|---|---|---|
| [W4 RENGE正式H1未通过](w4_results.md) | 当前协议下velocity增量未建立；辅助直接U有增量 | 所有U或velocity无用 |
| [W3修正后的检验与GFG审计](w3_results.md) | 指定配置、输入和测量下的稳定性/增量边界 | GFG普遍不能用、U没有未来信息 |
| [scNT-seq首轮](w5_scntseq_results.md) | 原预测规格未通过，存在估计器/时间核问题 | 不能由此否定其他短时任务 |
| [scNT-seq修复](w5_scntseq_repair_results.md) | 有探索改善，但速度特异性未充分隔离 | 修复后即独立复制成功 |
| [scNT-seq总量与组成归因](w5_scntseq_U_attribution_results.md) | 总量、组成和聚合方式的作用不同 | 净velocity变换已经优于全部同模态表示 |
| [W11群体路由、W12学生](../mainline/w10_recipe_bridge_results.md) | 从头群体路由/该S-only蒸馏未保住优势 | 不能用冻结群体q有效推导从头训练一定有效 |

旧报告中的历史阶段待办不是当前启动指令；最新见 [STATUS](../../current/STATUS.md)。
