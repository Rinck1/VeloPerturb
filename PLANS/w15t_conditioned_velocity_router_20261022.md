# W15T：扰动条件进入 router，重测 velocity 关系监督

2026-10-22。用户同意后启动；本轮只改当前 Traxler 小模型的路由接口。
原接口 q(k|S,U)，新接口 q(k|S,U,c)。原专家场已经读取 c，不能说原整个模型不读扰动。
6h 源细胞处于 KO 背景，S/U 也可能间接包含 c；显式条件是一项待验证结构假设，不是已证明的失败原因。

## 固定设计

- 四臂：有条件 S/U 无图、有条件 S/U＋真实 velocity 图、同结构打乱 velocity 图、同结构 U 图。
- 三训练 seed（0/1/2），共 12 个新模型，各 5000 步；K8、独立 Tsit5 专家 ODE、原学习率/批量不变。
- 全臂输入为原始 S50、U50 和 5 个常零槽；velocity 只在训练源图监督中使用，推理不用图或显式 velocity。
- 图使用原始解析 velocity，原 20 个 S 近邻中选 4 个，平方余弦；不平滑，不增强约束，λ=0.1。
- 新条件分支：56 个训练 KO 的公共 ESM 嵌入拟合冻结 PCA32，训练 KO 等权标准化；不使用验证嵌入拟合。
  将 c32 经零初始化、无偏置 Linear(32→64) 加到原 router 隐层预激活，再 SiLU→logits→softmax。
  仅增加 2048 个参数；四臂均为 391,752 个参数，初始化相同。零分支下应复现原 q 和专家初始化。
- c 进入 router 和原专家骨干，不进入 U/动力学编码器。18h 时间跨度固定，本轮不新增时间分支。
- 不训练 GFG、不用 JVP。后续目标域朴素速度估计器重训另立实验，不与本轮混合。

## 判断与归因

所有 12 个模型、84 个新预测和 84 个原无条件匹配参考预测冻结后，才计算本轮开发分数。
主量为逐 KO×seed 的 `1−Energy(real)/Energy(control)`，KO、seed 等权，不做最佳 seed 或 KO 筛选。
真实 velocity 必须同时优于 **有条件** 无图、打乱图和 U 图，并且三个 seed 同向，才通过此小实验的稳定性筛查。
扩大目标：相对有条件无图达到 2%，且优于原无条件 velocity 图；达不到如实报告。
额外分别报告：条件分支对无图的影响、条件分支对真实图的影响，以及 velocity 的配对绝对增量之差。
条件信息本身的收益不计为 velocity 收益；平方余弦图不能证明正确箭头方向。
这是已经用于开发的 7 个 KO 上的固定设计筛查，不是独立确认。原 7 测试 KO 目标仍不读。

## 完整路径

- 配置：`/home/yuchang/wangjiaxuan/configs/w15t_conditioned_velocity_router_20261022_v1.json`
- 源/训练目标包：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_fold_6to24_v1/`
- 切分：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_trainable_v1/split_6to24.json`
- ESM：`/data/yuchang/veloroute_w15e_traxler_conditions_20261021/esm2_3b_v1/conditions.npz`
- 原始图/对照：`/data/yuchang/veloroute_w15o_neighborhood_20261022/five_arm_v1/`
- 新输出：`/data/yuchang/veloroute_w15t_conditioned_router_20261022/four_arm_v1/`
- 启动记录：`/home/yuchang/wangjiaxuan/outputs/w15t_conditioned_router_jobs_20261022/gpu0/`

使用空闲 GPU0；GPU1 他人作业不动，GPU6 保留。本轮 12 个模型顺序运行，预计约 8–12 分钟，不为占满卡而重复拟合。
