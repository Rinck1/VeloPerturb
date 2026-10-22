# W16 完成结果：小型可复核数字

2026-10-22。这里只保存冻结评分产物的文本副本，不保存原始细胞数据、预测数组或模型权重。
解释与完整方法见[W16结果报告](../../../current/w16_perturbation_execution_20261022.md)。

## Traxler 扰动开发三折

- [summary](traxler_summary.json)：63模型、1323预测及七项真实图对照结果。
- [逐KO×seed×arm指标](traxler_metrics.csv)：1323行；可重新计算配对比较。
- [比较表](traxler_comparisons.csv)：相对Energy改善率及KO bootstrap区间。
- 原始目录：`/data/yuchang/veloroute_w16a_traxler_20261022/graph_cv_v1/evaluation/`。

改善率在JSON/CSV中以小数记录，报告中的百分比为其乘100。
先在KO×seed内配对，再平均seed、等权KO；不是全局均值之比。
原7测试KO未使用；这是63开发KO的固定三折，不是独立确认。
真实v图vs无图−0.366%、vs U图+0.106%、vs三打乱均值−0.049%，均未建立稳定优势。

## scNT-seq KCl 短时机制读出

- [summary](scntseq_summary.json)：固定校准、滚动预测和假设诊断。
- [逐区间指标](scntseq_metrics.csv)：六臂、两区间共12行。
- [标记参照诊断](scntseq_reference_diagnostics.csv)：工作模型的失配记录。
- 原始目录：`/data/yuchang/veloroute_w16b_scntseq_20261022/label_reference_v1/`。

剪接v相对S历史读出+4.400%/+7.239%，相对U驱动−0.127%/−1.565%；
这是复现已有短时线索。标记参照不满足速度真值/上限的资格，不能用于反证所有velocity或蒸馏新GFG。

## 版本说明

summary中的git_commit是**执行时的基准提交**，不是本次上传提交；
未提交执行脚本和配置另有data_hashes记录。原始产物不因上传改写。
JSON副本与服务器对应文件逐字节一致；CSV保留全部行、字段和数值，换行统一为LF。
原始文件与副本哈希见[复制清单](copy_manifest.json)；冻结预测/模型清单仍保存在上述原始目录。
