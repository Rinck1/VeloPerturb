# W15Q：不看未来的velocity方向计数噪声诊断

2026-10-22，在查看本轮拆分结果前固定。只读取现有7开发KO在6h的592个源细胞；
不读取24h细胞计数，不训练模型，不把可重复性称为生物学准确率。

对S/U整数计数做互补Binomial(n,0.5)拆分，五个固定随机种子。
保留原训练折拟合的HVG、PCA、gamma、尺度和响应轴，不在拆分后重新拟合。
比较两半的S状态50维、U50维、velocity50维、中心化velocity程序5维和U程序5维。
主要指标为同细胞两半的余弦、平方余弦及负余弦比例，逐KO/重复完整报告。

三种固定情形用于区分噪声来源：

- both_split：S/U都拆分，各半以该半的全部S深度归一化；
- U_only_split：S不变，U两半乘2后用原S深度归一化；
- S_only_split：S按各半深度归一化，归一化后的U固定为原值。

未选基因的S总计数也独立做二项拆分，以正确传播全部S文库深度；
不需要稠密加载54,232基因，不四舍五入非整数计数。
先确认原始计数经现有公式能复现源pack的z/U/velocity，避免把读取错误当噪声。
半深度会降低可重复性，不能把所得相关性当全深度误差上限；
共享PCA/gamma也会保留系统偏差，不能把高相关当正确方向。

若原始箭头重复性高、中心化程序却低，优先考虑个体残差/程序噪声问题；
若重复性高但方向/未来任务仍失败，则说明仅减少计数噪声未必解决时间尺度或系统性偏差。
这些是解释规则，不用于事后挑选更好的KO或训练样本。

原始计数：`/data/yuchang/veloroute_w4/traxler2025/processed_usa_v1/{B1,B2,B3}.h5ad`
源pack与变换：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_fold_6to24_v1/`
响应轴：`/data/yuchang/veloroute_w15o_neighborhood_20261022/five_arm_v1/training_programs.npz`
输出：`/data/yuchang/veloroute_w15q_countsplit_20261022/source_reliability_v1/`
CPU执行，不占用或打断GPU任务。
