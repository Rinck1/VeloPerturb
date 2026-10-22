# W15P：固定图监督的打乱稳健性

2026-10-22，在W15O全部完成后写定；不修改已完成的真实模型和五臂结果。
W15O真实velocity图相对无图+0.684%、单次打乱+0.620%、U图+1.288%，但seed不一致，
没有达到原计划的稳定成功判据。这里做必要的机制延伸，不以继续尝试直到阳性为目标。

新增9模型：KO内全局置换seed1/2以及一个S局部置换，各model seed0/1/2。
复用原真实图、无图、表达图、U图、置换seed0，共15个父模型；不重训父模型。
每个新增模型仍389,704参数、5000步、原初始化、原采样、图权重0.1、20候选/4邻居。
只换图边，不加velocity输入，也不改ODE。

全局对照按已有W15I函数构造，与父seed0同一置换算法，仅换预定置换种子。
局部对照在每个训练KO源S的原PCA欧氏距离中贪心选择最近配对并交换velocity程序；
固定随机访问顺序，奇数末尾用3循环，保证无固定点的双射，不读取源U/velocity或未来目标来选配对。
这是近似S匹配的随机对照，不宣称精确条件独立或保留全部S相关结构。

原seed0和新增seed1/2的打乱Energy先在每个KO×模型seed内等权平均，再与真实图比较。
另单独比较局部置换图，以及每个全局置换图；不挑最差打乱当对照。
完整63份新增预测和105份父预测冻结后评分。原7个测试KO目标保持不读。
若真实图不能稳定领先三打乱均值和S局部打乱，不把它解释成已建立的动力学增量；
仍报告全部逐KO/seed数值。无论结果，不能宣称证明velocity箭头，因为图用平方余弦，对反向不敏感。
即使新对照均为正，也不会改变原W15O相对无图/U图的seed不一致或其未通过原稳定规则的事实。

配置：`/home/yuchang/wangjiaxuan/configs/w15p_velocity_graph_controls_20261022_v1.json`
脚本：`/home/yuchang/wangjiaxuan/scripts/phase0/w15p_velocity_graph_controls.py`
父输出：`/data/yuchang/veloroute_w15o_neighborhood_20261022/five_arm_v1/`
新输出：`/data/yuchang/veloroute_w15p_graph_controls_20261022/null_stability_v1/`
数据路径和数学见[W15O固定方案](w15o_velocity_neighborhood_20261022.md)。
使用空闲GPU0，不占GPU1或GPU6；大型缓存和模型只写/data。
