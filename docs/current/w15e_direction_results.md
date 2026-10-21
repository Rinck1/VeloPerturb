# 方向 router 小实验与 Traxler 扰动接入（2026-10-21）

## RENGE：16个短训练已完成

固定选择原32折中按字母排序第一个TF ETS2的两个区间；四臂各seed0/1、2000步、128×2、K8。
四臂为原动力学router、方向偏置、保留每颗细胞幅度的方向打乱、direct-U。
无选参，原全量L2/L3训练照常继续。源和目标各至多128细胞，不能与512细胞正式网格绝对分数混比。

方向偏置为 `q ∝ q_original * exp(0.5*cos(v_abs,F_k(z0,0,c)))`，前300步渐开。
使用原L1零参考包中的绝对64-NN PCA速度，未减对照、未逐维标准化；与专家总体场相比较。
夹角中的专家方向stop-gradient，避免专家通过该项的梯度直接旋转方向。

| 留出区间 | vs 原router | vs 方向打乱 | vs direct-U |
|---|---:|---:|---:|
| ETS2 day2→3 | −0.072% | −0.030% | −18.134% |
| ETS2 day3→4 | +2.314% | +0.020% | +6.609% |
| 两区间等权 | +1.121% | −0.005% | −5.763% |

16次拟合及全部预测冻结后评分，总运行约476秒。两个区间同属一个TF，结果仅用于探索。
当前没有建立正确方向对应关系的优势。

## 补充机制检查：群体内方向几乎一致

在16模型完成后做一次冻结权重分析，保持方向版模型与专家候选不变，只修改方向偏置。
原q为 `q_base*exp(lambda*cos)`，关闭和反向分别精确重加权 `exp(-lambda*cos)`、`exp(-2*lambda*cos)`。
下表为正确方向相对反事实的Energy改善，负值表示反事实更好。

| 区间 | 源速度随机配对方向cos均值 | 正确偏置 vs 关闭偏置 | 正确偏置 vs 反向偏置 |
|---|---:|---:|---:|
| day2→3 | 0.99156 | −1.619% | −3.215% |
| day3→4 | 0.98870 | +0.350% | +0.701% |

因此，组内打乱是较弱的绝对方向干预；它主要检验逐细胞对应关系。冻结反向检查也没有跨区间一致收益。
本轮不把+1.12%解释为已经证明方向有效，也不据此调lambda或继续扩大RENGE小实验。

完整模型/预测/CSV：`/data/yuchang/veloroute_w15e_direction_20261021/pilot_v1/`。
配置：`/home/yuchang/wangjiaxuan/configs/w15e_direction_pilot_20261021_v1.json`。
启动记录：`/home/yuchang/wangjiaxuan/outputs/w15e_direction_jobs_20261021/pilot_gpu2/`。

## Traxler：真实蛋白条件完成，0→6h开发比较已启动

原S/U包保留；132个KO均精确匹配Ensembl93小鼠蛋白，按最长isoform、protein ID打破并列。
使用本机完整ESM2-3B，layer36、窗口1022、不截断长蛋白、残基均值；132/132条件已生成。
参考与官方BSD CHECKSUMS一致。第一次入口缺fair-esm，隔离安装到数据盘后重试成功；没有修改全局环境。

- 蛋白条件：`/data/yuchang/veloroute_w15e_traxler_conditions_20261021/esm2_3b_v1/conditions.npz`。
- 原0→6h数据包：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_fold_0to6_v1/`。
- 冻结KO切分：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_trainable_v1/split_0to6.json`。
- 开发结果目录：`/data/yuchang/veloroute_w15f_traxler_direction_20261021/pilot_v1/`。
- 配置：`/home/yuchang/wangjiaxuan/configs/w15f_traxler_direction_20261021_v1.json`。
- 后台日志：`/home/yuchang/wangjiaxuan/outputs/w15e_direction_jobs_20261021/traxler_direction_gpu3/`。

106个KO训练、13个验证KO评分、13个测试KO保留。五臂static/原动力学/方向/打乱方向/direct-U，各3seed、2000步，共15模型；所有验证预测完成后统一评分。
实际6小时增量输入条件，ODE tau=0→1；不复用24h生物时间条件。
当前源速度为原冻结transform的逐细胞解析残差，整体正标量缩放不改变方向；本轮未加64-NN池化。
各臂使用真实冻结小鼠ESM，而非旧one-hot占位。

该任务源和未来具有相同KO背景，预测感染刺激后的6h变化；属于遗传背景条件化的短时刺激响应。
训练目标现已用于模型拟合，Traxler进入开发阶段；测试目标尚未读取。
HTO来源不是本轮切分轴，不提出供体/来源留出主张。初次开发结果不能称为独立确认。

几何/实际OT与ODE测试4项、蛋白选择1项、Traxler时间与数据边界2项通过，共7项。

仅训练侧的方向检查：106个训练KO中，0h平均速度与0→6h群体平均位移的cos均值−0.00630，
中位数−0.00826，正值比例39.62%。该检查未打开验证/测试目标，也未用于更改上述固定训练配置。
它提示刺激前的内源速度与刺激后有限时间响应方向可能不同；这不是对逐细胞速度准确性的直接检验。
