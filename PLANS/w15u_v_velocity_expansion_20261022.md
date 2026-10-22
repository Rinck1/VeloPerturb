# W15U/V：保留动态差异与目标域估计器，两条独立扩大实验

2026-10-22，用户要求继续寻找扩大方式。W15T 已完成后固定本方案，再启动新训练。

## U：关系监督不只拉近，也检验差异

动机：原正样本 JS 只奖励相近 q，强约束伴随专家使用集中；这不等于已经证明坍缩是唯一失败原因。
本轮以局部对比监督检验：相似动态细胞的 q 应比不同动态细胞的 q 更接近。

同训练 KO 的源 S20近邻中，各取程序最相似4个和最不相似4个（不重合、无自身）。
有符号相似度 `(1+cos)/2`；无符号对照 `cos²`；二者均落在 [0,1]。
每次各抽一个，`m=0.1*(sim_positive−sim_negative)`；源分布一致时 margin 自动趋零。
`L_contrast=mean relu(JS(q_i,q_pos)/log2−JS(q_i,q_neg)/log2+m)`。
总损失仍为原 FM/均衡项加 `50*0.1*L_contrast`，不额外加原正样本 JS。
图和 margin 为冻结源侧 teacher，绝不使用未来细胞推断单细胞真命运。

四臂：有符号velocity、无符号velocity、有符号打乱velocity、有符号U，三seed共12新模型。
共同使用 W15T 条件路由、391,752参数、K8独立Tsit5、5000步、原始S/U输入。
复用 W15T 条件无图和原吸引式velocity图各三seed作为参考。
主候选预先定为有符号velocity；无符号为机制对照，不事后换主候选。
全局反转所有 velocity 不改变成对余弦，因此即使有符号关系更好，也不证明箭头绝对朝向正确。

## V：目标域重训无 JVP 的 S/U 计数估计器

不是复用旧基础模型，也不是原版GFG。采用朴素目标域计数自监督网络；不把网络名称当有效性证据。
仅用训练 KO 的6h原始 S/U，公共冻结的2000基因、PCA、γ、支持基因与 S-library 归一化不变。
双计数输入 `log1p(S_norm),log1p(U_norm)`，MLP `4000→256→64→256→4000`，输出正 S/U 期望率。
训练计数按三套固定 binomial half-split 生成互补视图，轮换两半输入/另一半计数监督。
用目标半的全S深度作为 exposure，拟合原始整数计数的 Poisson 工作损失，S/U两项按训练均值计数尺度平衡。
网络不读 c、时间标签、目标表达或既有velocity；无 JVP、无VQ、无旧权重。
固定2000步、batch128、lr=1e-3，估计器分别seed0/1/2，与下游同seed对应；不按开发分数早停。
冻结网络后计算 `v_gene=(U_hat−γ*S_hat)/(1+S_hat)`，按原支持基因和PCA投影；不改小时量纲。
U对照使用同一网络的 `log1p(U_hat)` 投影，匹配估计器预训练预算。
主模型的源状态和U输入仍是原始冻结变换，只改变训练图teacher。

每个seed训练真实估计v图、其KO内打乱图、同网络U图，共9新下游模型；
复用W15T条件无图/原始velocity图。图仍用原平方余弦和原正样本JS，λ=0.1。
所以U改变监督形式、不改估计器；V改变估计器、不改监督形式，两个结论分别报告，不混成一个胜者。
本轮是先训练并冻结估计器再做下游，尚非端到端联训；源重建更好不自动视为velocity有效。

## 判读与预算

U GPU0，V GPU2；GPU1他人任务不动、GPU6保留。最多两卡同时运行。
每项各自全网格预测冻结后才读7个开发KO目标；原7个测试KO目标始终不读。
所有开发KO等权，不筛选“好看”或多峰条件，不挑seed或调λ。
真实v须均胜本项无图、U、打乱，且各比较三个seed同向，才通过本项筛查。
扩大目标为相对无图至少2%，同时优于W15T原始正样本velocity图；与筛查分开报告。
若失败，保存机制与阴性结果，不把技术重复性当任务成功。

### 计数拆分的解释

互补binomial拆分在条件Poisson抽样模型下可构造稀释视图；对任意过度离散观察分布不声称独立。
训练和半深度敏感性都不是新的生物重复，也不是velocity真值。
[Molecular cross-validation](https://www.biorxiv.org/content/10.1101/786269v1)提供分子拆分的自监督思路；
[RNA velocity count-splitting](https://pmc.ncbi.nlm.nih.gov/articles/PMC13110005/)讨论针对过度离散数据的额外假设。
这里使用固定简单工作模型，不把该文的NB独立性保证套用到我们的binomial拆分。

## 完整路径

- 训练折：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_fold_6to24_v1/`
- 原始计数：`/data/yuchang/veloroute_w4/traxler2025/processed_usa_v1/{B1,B2,B3}.h5ad`
- 原条件模型：`/data/yuchang/veloroute_w15t_conditioned_router_20261022/four_arm_v1/`
- U配置：`/home/yuchang/wangjiaxuan/configs/w15u_velocity_contrast_20261022_v1.json`
- V配置：`/home/yuchang/wangjiaxuan/configs/w15v_target_splicing_estimator_20261022_v1.json`
- U输出：`/data/yuchang/veloroute_w15u_velocity_contrast_20261022/fixed_contrast_v1/`
- V输出：`/data/yuchang/veloroute_w15v_target_estimator_20261022/fixed_estimator_v1/`
- 启动日志：`/home/yuchang/wangjiaxuan/outputs/w15u_v_expansion_jobs_20261022/`
