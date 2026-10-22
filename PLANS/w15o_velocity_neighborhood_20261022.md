# W15O：velocity 是否适合教 router 学源侧邻域关系

固定于2026-10-22，查看本轮验证分数前写定。W15N速度重建辅助损失未赢打乱/U，
所以本轮不扫重建权重，换成关系监督。问题是“动力学程序相近的源细胞，是否应该有相近的响应混合配比”，
不是把velocity解释成真实逐细胞终态或未来位移。

## 数学和五臂

所有臂输入同样的源S50、U50以及5个恒零槽；389,704参数，K8，5000步，seed0/1/2，
原生CellFlow骨干、表达OT、固定源侧q以及各专家独立Tsit5 ODE均不变。
无辅助头，无GFG/JVP，无门控。训练KO的响应均值定义已有5维响应轴，不改原训练/验证/测试切分。

对每个训练KO，在源S的PCA欧氏距离中找20个候选邻居，排除自己。
velocity/U程序来自该细胞的源侧数据。用 `1-(a·b)^2` 对单位程序向量选4个邻居，
即不指定velocity正负箭头，而比较程序关系。U臂用同一规则；打乱臂固定KO内velocity置换。
state臂只选4个最近S邻居。无图臂权重为0。

`L = L_original + D * 0.1 * mean(JS(q_i,q_j)/log(2))`，D=50，固定一个权重。
每次从4个邻居均匀抽1个；双端q均可接收梯度；图损失只直接更新router和U编码器。
OT匹配只看表达；图边跟随OT的源索引，包括重复抽到同一个源细胞。
图邻居可以不在本次FM mini-batch内，但始终是训练KO源细胞。
这是平滑先验而非命运标签；常数q也能使图损失为0，因此必须同时检查任务误差、细胞熵及模式使用率。

五臂：no_graph / state_graph / velocity_graph / shuffled_velocity_graph / U_graph。
15个拟合/105份预测；整个网格冻结后一次评分。以KO×seed配对Energy为主，均值、方差和路由统计为辅。
velocity只有同时优于无图、表达图、打乱图和U图，并且seed方向一致时，才作为候选机制继续。
此数据已用于多轮开发；不将其称为独立确认，不挑选最佳KO/seed，不开原7个测试KO目标。

## 先做的源侧诊断

输出每KO的图重叠率、S距离、velocity程序平方余弦、邻域均值预测程序的残差等。
用于区分velocity图是否只是复现U/S邻域，并非主任务有效性的替代证据。
图和预处理只用训练源细胞；读取训练目标仅为复用已有响应轴。

## 完整路径与执行

- 源/目标折：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_fold_6to24_v1/`
- KO切分：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_trainable_v1/split_6to24.json`
- ESM：`/data/yuchang/veloroute_w15e_traxler_conditions_20261021/esm2_3b_v1/conditions.npz`
- 父模型：`/data/yuchang/veloroute_w15i_router_followups_20261021/queue_v1/cell_6to24/`
- 输出：`/data/yuchang/veloroute_w15o_neighborhood_20261022/five_arm_v1/`
- 脚本：`scripts/phase0/w15o_velocity_neighborhood.py`
- 配置：`configs/w15o_velocity_neighborhood_20261022_v1.json`

GPU0空闲时运行，不占GPU1/6、不停止他人作业。大型检查点和缓存放/data。
若失败如实保留；不根据中途验证分数改图、改权重或延长训练。
