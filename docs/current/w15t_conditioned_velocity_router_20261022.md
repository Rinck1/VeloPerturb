# W15T：条件路由完成，整体小幅改善，velocity 增量未扩大

2026-10-22。12/12 新模型完成，84 个新预测和 84 个旧参考共同冻结后评分。
实际运行约 9 分 25 秒，GPU0 已释放；相关联合 CPU 回归 26/26 通过。

## 结论

**显式扰动条件对当前 router 有小幅帮助，但不是扩大 velocity 特有收益的突破口。**
不改变原始 velocity 和弱图权重时，加条件的真实图版本比旧真实图改善 0.379%，三个 seed 同向；
无图版本加条件改善 0.591%，三个 seed 同向、6/7 KO 改善。
因此条件分支值得保留为候选接口，但不能把这部分改善算作 velocity 的作用。

真正的 velocity 增量要相对同样加条件的控制臂衡量：

| 真实 velocity 图的同结构对照 | 平均配对 Energy 改善 | seed0 | seed1 | seed2 | 改善 KO |
|---|---:|---:|---:|---:|---:|
| 有条件无图 | +0.487% | −0.090% | −1.349% | +2.900% | 4/7 |
| 有条件打乱 velocity 图 | +0.536% | +2.326% | +0.348% | −1.068% | 4/7 |
| 有条件 U 图 | +0.692% | +1.219% | +1.462% | −0.605% | 4/7 |

三个对照的平均方向均为正，但均未三个 seed 同向，原固定筛查规则和 2% 扩大目标均未通过。
原无条件真实图相对无图为 +0.684%，本轮为 +0.487%，并未扩大。
等权绝对 Energy 差同样从 0.00140028 降至 0.00112171，条件×velocity 的增量之差为 −0.00027857。
不是“总体模型变好，所以 velocity 被证明更重要”。

## 模型和评价口径

只把当前 `q(k|S,U)` 改为 `q(k|S,U,c)`；原专家骨干本来已经读取 c。
公共 ESM 的 PCA32 与标准化仅在 56 个训练 KO 上拟合并冻结。
新分支为无偏置、零初始化的 Linear(32→64)，加到原 router 的隐层预激活。
新增 2048 参数，四臂均为 391,752 参数；初始化移除新零分支后与原模型逐字节一致。
c 不进入 U/动力学编码器，K8 独立 Tsit5 ODE、5000 更新和其余预算不变。

所有臂共同输入原始 S/U，附加 5 个程序槽为零。velocity 只定义训练源细胞之间的关系，
以 λ=0.1 的 JS 图损失监督 q；部署不用图或显式 velocity。
图为原始解析程序的平方余弦关系，本轮不能证明正反向箭头；未训练 GFG、未用 JVP。

仍为 Traxler 6→24h（18h early-response）任务，全部 7 个开发 KO 等权，没有效应/多峰筛选。
每个数为 21 个 KO×seed 的配对相对改善再平均，不是平均 Energy 的比值，也不做 seed 集成。
7 个开发 KO 已用于此前探索，本轮不是独立确认；原 7 个测试 KO 目标没有读取。

## 描述性分布指标

| 模型 | 平均 Energy | 方差对数误差 | 有效专家数 |
|---|---:|---:|---:|
| 原无条件无图 | 0.15712111 | 0.084819 | 7.262 |
| 原无条件 velocity 图 | 0.15572082 | 0.086864 | 6.624 |
| 有条件无图 | 0.15619608 | 0.086399 | 7.201 |
| 有条件 velocity 图 | 0.15507437 | 0.086980 | 6.603 |
| 有条件打乱图 | 0.15586673 | 0.084882 | 6.603 |
| 有条件 U 图 | 0.15631002 | 0.088106 | 6.548 |

真实图平均 Energy 最低，但方差误差没有优于有条件无图，不能据此宣布多峰结构优势。
加条件＋真实图相对最早无条件无图的整体改善为 +1.094%，seed 为 +0.571%、−0.252%、+2.963%；
这是后续从固定指标表计算的描述性整体比较，含结构和 velocity 两种变化，不是预注册的 velocity 主比较。

## 后续选择

当前优先保留显式条件接口，不扩大同一组 K、参数量、图强度或平滑网格。
若继续争取 velocity 主贡献，应另立目标域、不用 JVP 的朴素速度估计器重训实验，
并与这里原始估计器和 U 控制同任务比较；本轮未执行该步骤，也不预设一定有效。
本轮不能归因为“router 已学到真实命运”，没有逐细胞配对命运标签。

## 完整产物路径

- 固定方案：[W15T](../../PLANS/w15t_conditioned_velocity_router_20261022.md)。
- 实现：[w15t_conditioned_velocity_router.py](../../scripts/phase0/w15t_conditioned_velocity_router.py)。
- 测试：[test_w15t_conditioned_velocity_router.py](../../tests/test_w15t_conditioned_velocity_router.py)。
- 输出：`/data/yuchang/veloroute_w15t_conditioned_router_20261022/four_arm_v1/`
- 主结果：上述目录的 `summary.json`、`metrics.csv`、`comparisons.csv`、`velocity_condition_interaction.csv`、`RESULTS.md`。
- 模型：`fits/seed{0,1,2}/{no_graph,velocity_graph,shuffled_velocity_graph,U_graph}/model.msgpack`。
- 冻结条件变换：`condition_projection.npz`；冻结网格：`whole_grid_frozen.json`。
- 有效启动记录：`/home/yuchang/wangjiaxuan/outputs/w15t_conditioned_router_jobs_20261022/gpu0_retry1/`。
- 首次启动在 CellFlow 导入前退出，未产生模型/预测；旧日志保留，同一固定配置修正导入顺序后重启。
