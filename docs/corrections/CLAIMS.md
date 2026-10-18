# 撤回、受限与过时主张清单

本表逐条限制推断，不整份否定历史报告。原始数字和代码快照均保留。
“撤回”针对错误/过强的解释；“未建立”不是已经证明不存在；“过时”不是实验做错。

## A. 不得继续引用的错误或过强推断

| 旧主张 | 问题与当前替代表述 | 依据 |
|---|---|---|
| velocity/U在状态之外没有任何未来信息、信息论上无法修复 | 特定阴性、噪声和有限probe不能推出普遍条件信息为零；当前另有短时群体阳性 | [代码/统计复核](audits/velocity_code_review_20261014.md)、[W13结果](../current/w13_time_results.md) |
| 旧R²≈0.989意味着98.9%的逐细胞方差由状态解释 | 旧统计量采用未中心化能量分母，不能如此解释；原方向诊断还存在验证源重拟合问题 | [旧交接的9月21日审计更正](../history/handoffs/HANDOFF_before_organization_20261017.md) |
| energy有提升但置换后消失，因此提升一定来自分散度 | 没有直接分解来源，不能把这种归因写成已证事实 | [复核](audits/velocity_code_review_20261014.md) |
| 简化GFG v2的3/3验收证明速度质量合格 | count-split共享S、坐标不一致、innovation在推理batch重拟合；不能合称独立质量验收 | [复核](audits/velocity_code_review_20261014.md)、[历史v2报告](../history/superseded_claims/w2_gfg_v2_results.md) |
| 把未来Y直接置换就得到给定S的条件增量零分布 | 同时破坏Y与S的依赖，条件零假设并未自动校准 | [统计复核](audits/velocity_code_review_20261014.md) |
| 一种pseudobulk残差置换均值不变，足以证明所有速度置换检验失效 | 限定构造下可有均值不变量，但不能直接升级为普遍可识别性定理 | [历史W1](../history/superseded_claims/w1_results_20261009.md)、[复核](audits/velocity_code_review_20261014.md) |
| 原confirmation克隆/TF仍从未使用过 | 审计发现其已进入W1/W2诊断；重新划分不恢复独立确认资格 | [W2更正](../history/superseded_claims/w2_results_20261010.md)、[复核](audits/velocity_code_review_20261014.md) |
| 原生GFG方向对U破坏不敏感，说明所有GFG/所有U都没有用 | 只能限定于指定权重、输入与训练规格；输出方向敏感性不同于预测信息 | [W3原生GFG审计](../evidence/boundaries/w3_results.md) |

## B. 仍未建立、不能当成已有结果的假设

| 说法 | 当前状态 |
|---|---|
| router恢复了逐细胞真实命运或真实分岔 | 未建立。当前端点机制更支持群体混合比例，专家编号不是观测命运标签 |
| MeRLin同时间关联就是跨时间预测成功 | 不是同一任务；需区分Day0→Day21、Day21内关联、in-vivo 308和in-vitro 315及克隆映射 |
| 当前velocity只作训练监督，部署不需要U | 当前实现不符合此说法；源侧动力学均值与路由仍使用U相关信息 |
| 当前W13提升来自原版/重训GFG或JVP | 不符合实现身份；当前使用解析邻域动力学，不能借用GFG名字归因 |
| 18M必然优于小模型，或W9阴性证明大模型无效 | 均不成立。W9比较同时涉及目标/配方，W10已显示目标改变会影响结果 |
| ODE在所有视野都更好，且专门放大velocity特异性 | 5/10h绝对拟合改善；static也改善，15h无实质积分增益 |
| 已超过外部最强方法、已独立复现、已完成未见扰动泛化 | 尚未执行/建立，不能用内部static/shuffled比较替代 |

证据：[当前可引用主张](../current/CLAIMS.md)、[W10–W12机制](../evidence/mainline/w10_recipe_bridge_results.md)、
[W13结果](../current/w13_time_results.md)、[W3数据与标签审计](../evidence/boundaries/w3_results.md)。

## C. 已过时的状态与执行方向

- “主线是原版GFG decoder-JVP＋完整EM/门控”：是历史架构，现在不是工作版本。
- “router尚未启用”“完全没有真实阳性”：不再反映W6–W13当前结果。
- “RENGE day2/day3还没落地”“MeRLin还没Day0”：后续已取得相关数据，不能照抄旧阻塞。
- “K4/4000或K4/12000就是最终推荐”“W9还在训练”：已被后续比较/完成状态替代。
- “旧Velocity Horizon提纲就是当前论文提纲”：旧提纲已归档，当前方法/结果可以开写。

这些变化不回改旧实验。按日期读历史，以 [STATUS](../current/STATUS.md) 认定当前版本。
