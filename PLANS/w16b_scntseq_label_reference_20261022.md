# W16B：KCl短时预测中的标记动力学参照

2026-10-22，五臂新读出和本轮新分数计算前固定。与W16A的扰动关系监督并行，CPU运行。
本系列已有多轮开发结果，不作为独立确认，不改写原W5的任何数字或判决。

## 固定任务

原配对13,476细胞的S/U/new/total和原0分钟拟合的PCA/HVG均不变，沿用原技术QC。
15→30分钟仅一次拟合读出，30→60与60→120滚动预测；先冻结30→60，才加载60作为下一源。
120分钟只在第二次预测冻结后评分。0→15不作为必须由刺激前速度预测的任务。

五臂：S＋历史S、S＋U驱动、S＋剪接v、S−剪接v、S＋标记支持参照，另附不拟合的identity。
每臂最多两标量、ridge=1，沿用15分钟校准的decay时间核，不另调参数。
S/历史/U系数自由；v、反向v、标记参照的系数非负，从而不能靠翻转系数偷换方向。
剪接v和U驱动沿用旧修复版的全基因S深度归一、pooled shrink gamma、基线场相减。
五臂共用全部原HVG/PCA，不根据新分数选基因或轴。

## 标记参照的假设和量纲

原实验每个样本收样前标记120分钟，不能将new/预测间隔当瞬时速度。
原文的质量作用模型为 `dn/dt=alpha−gamma*n, dr/dt=alpha−gamma*r`；
恒定速率脉冲下 `n=alpha/gamma*(1−exp(−gamma*T))`。
出处：[原论文方法](https://pmc.ncbi.nlm.nih.gov/articles/PMC8103797/)，
[真实120分钟标记设计](https://www.nature.com/articles/s41592-020-0935-4)。

本轮使用该模型的**刺激起点修正扩展**，不是声称逐字复现作者拟合算法：

- 基线0分钟假设群体近稳态，以每基因pooled `k0=N0/R0` 估计 `gamma=−log(1−k0)/120`。
- 仅基线原计数total≥20、new≥5、old≥5且0<k0<1的基因有标记速率支持，其余参照分量为0。
- 假设刺激前alpha0=gamma*R0、刺激后gamma不变；将从刺激到源时刻t的alpha近似为恒定有效合成率。
- t≤120时，当前new中刺激前合成且尚存的贡献为 `R0*(exp(−gamma*t)−exp(−gamma*120))`。
- `alpha_eff=max(0, gamma/(1−exp(−gamma*t)) * [N_t−上述刺激前贡献])`，
  参照净变化为 `alpha_eff−gamma*R_t`。小gamma极限的前因子为1/t。
- 记录负合成估计截断比例；大量截断可能表示假设/相对归一化不匹配，不能隐去。

R/N来自同一labeling定量，先以同源群体全基因total归一；S/U使用其自身S库深。
两套计数不直接相减、不假设 `total=S+U`。
仅以基线每基因 `S0_normalized/R0_normalized` 定义固定正观测映射，再通过
`v_S/(1+S_t)` 和原PCA载荷投影。该映射假设跨时点稳定，无法保证消除两管线偏差。
因此所得是有明确模型假设的相对量纲参考场，不是实测真值、严格性能上限或绝对分子/分钟。

这项扩展需要基线近稳态、标记识别可比、固定gamma、阶段内有效alpha和相对RNA归一化等假设。
alpha变化很快、RNA含量/捕获变化或基因特异检测偏差均可能使其失配。
若参照失败，不能据此推出“准确速度也帮不了任务”；先看假设诊断与两套计数的映射。

## 输出和解释

报告两个区间全部六臂的PCA均值MSE、相对S/直接U/identity增益、原始场与后续位移余弦及读出系数。
不把两个相邻区间或50个轴当独立生物重复，不生成生物学显著性p值。
参照优于S/U而剪接v不优，支持优先改善估计器的候选解释，但不是严格因果分解；
两者都不优，说明本次局部读出未获支持，不升级为全任务无用。
没有自动新增GFG训练、改router、打开保留KO或挑选最有利时间窗。

配置：`/home/yuchang/wangjiaxuan/configs/w16b_scntseq_label_reference_20261022_v1.json`
输入：`/data/yuchang/veloroute_w4/scntseq/processed/scntseq_geo_labeling_dynamo_su_exact.h5ad`
变换：`/home/yuchang/wangjiaxuan/outputs/w5_scntseq_predictions_v1/source_transform.npz`
输出：`/data/yuchang/veloroute_w16b_scntseq_20261022/label_reference_v1/`
日志：`/home/yuchang/wangjiaxuan/outputs/w16b_scntseq_jobs_20261022/cpu/`。
