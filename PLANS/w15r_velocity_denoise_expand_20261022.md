# W15R：扩大velocity关系监督优势的固定去噪实验

2026-10-22，执行前固定。当前W15O/P相对无图平均+0.684%、三打乱均值+0.654%，
但相对无图/U跨seed不稳定。W15Q发现程序方向对计数噪声敏感。
因此首先测试“降低teacher噪声”是否比单纯加大关系约束更有效，不加模型参数。

## 三组与预算

主候选`smooth_w01`：局部S/U计数矩去噪，沿用原图权重0.1。
机制组`smooth_w05`：相同去噪，图权重0.5。
机制组`raw_w05`：不去噪，图权重0.5。
不在验证分数出来后调权重、改邻居或挑主候选。

每组六臂：真实velocity图、U图、全局打乱0/1/2、S局部打乱；各三seed，54个新模型。
复用W15P原始权重0.1的24模型，并额外复用原L3的直接U程序输入及velocity程序输入六模型作强参照。
所有新模型仍389,704参数、K8、5000步、同初始化/采样、同官方CellFlow骨干和独立Tsit5专家。
模型前向仍读原始S50/U50和5恒零槽；去噪只改变训练关系teacher，不改共同模型输入。

## 去噪数学

在每个训练KO的源S50中找5个最近的其他细胞，构造行随机算子：

`T = 0.5 I + 0.5 A_5NN`。

S/U都先按每细胞全部S深度归一化，再计算`S_bar=T S, U_bar=T U`。
之后才计算`v_bar=((U_bar−gamma*S_bar)/(1+S_bar))@PCA.T/scale`，
U对照同样使用`log1p(U_bar)@PCA.T`。
不能直接平均原单位velocity方向冒充这个计数矩估计器，也不能只对velocity去噪而不给U同样处理。
HVG/PCA/gamma不重拟合；中心化参考均值在训练源细胞中重新计算，按KO等权。
响应轴沿用训练KO目标定义的已有5维轴。

原20个S候选/4条程序平方余弦图边不变，打乱在去噪之后进行，置换方式/种子继承W15P。
图loss为原FM加`D*lambda*JS/log2`，lambda只取预定0.1/0.5。
本轮是无参数去噪的判别基线，不是GFG训练；不使用JVP。

## 源侧检查及判读

训练源S/U由原始h5ad按barcode读取，复现原velocity/U后才能构造teacher。
验证源592个细胞做五次互补计数拆分，比较原始与去噪后程序的技术重复性。
其邻域基于完整源S固定，所以这测的是给定源邻域的去噪稳定性，不是独立生物学重复。
同时记录训练KO内程序方差保留，不能以“全变常数”假装获得可靠速度。
这些结果不用于选择邻居、权重或筛KO。

## 评价与扩大标准

三组全部预测冻结后统一评分；不先评分一组再改另两组。
重点比较主候选相对无图、原velocity图、直接U输入、同组U图、三打乱均值及S局部打乱。
目标为相对无图达到2%以上配对Energy改善，同时相对直接U/同组U及随机对照不退步且seed方向一致。
目标不是必然结果；若未达到，完整记录，不拿最优KO或最佳机制组替换主候选。
结构指标同时报告均值、方差误差、路由熵，不把单一均值改善叫多峰预测成功。
数据仍属开发，原7测试KO目标不读取。

## 完整路径

- 原始源计数：`/data/yuchang/veloroute_w4/traxler2025/processed_usa_v1/{B1,B2,B3}.h5ad`
- 原折：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_fold_6to24_v1/`
- 切分：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_trainable_v1/split_6to24.json`
- ESM：`/data/yuchang/veloroute_w15e_traxler_conditions_20261021/esm2_3b_v1/conditions.npz`
- 父图结果：`/data/yuchang/veloroute_w15p_graph_controls_20261022/null_stability_v1/`
- 输出：`/data/yuchang/veloroute_w15r_denoise_expand_20261022/fixed_grid_v1/`
- 日志：`/home/yuchang/wangjiaxuan/outputs/w15r_denoise_expand_jobs_20261022/`

CPU控制器准备计数图与源侧诊断后，GPU0/2/3各跑18模型，完成后统一冻结评分。
不使用他人GPU1或保留GPU6；失败不会终止其他作业；不修改父实验任何代码/结果。
