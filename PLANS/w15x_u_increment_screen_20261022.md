# W15X：训练条件内定位 U 的时间预测增量

2026-10-22，用户同意“先找有用的动态基因/程序”后固定本设计，再计算新分数。
只使用56个原训练KO；不读取7个开发KO或7个保留测试KO的目标。
目标是18h后群体均值响应，不编造逐细胞命运或配对。

## 问题与模型

每KO每基因，Y为6→24h平均log-spliced变化/18。
两种U信号为 A=mean[U/(1+S)]、mean[log1p(U)]，S/U均共享原全S深度归一化。
三种预先固定的信号家族：A的非负系数（正向剪接候选）、A的自由系数、logU自由系数。
原γ残差=A+γB，B=−mean[S/(1+S)]已进入S控制；不把A和该残差的等价创新重复算作新信息。

S控制包括逐基因源mean(logS)、B、mean(logS)²、var(logS)，以及共享源S-PCA6、公共ESM-PCA4、
源mean(log S-depth)、log细胞数和B1/B2比例。PCA、标准化与ridge=1均在相应训练子集拟合。
U条件均值用同一S设计的无惩罚SVD最小二乘投影，避免ridge留下本可由S解释的残差再被算作U增量。
U创新能量低于原U训练方差的1e−6（或绝对1e−12）时，额外系数直接置零。
公共c只作增量诊断的控制变量，不输入任何动力学编码器；本轮没有新神经模型。
先拟合S/c预测Y及U，再用U残差预测响应残差；额外系数同为一维ridge，正向家族截断负系数为0。
三套对照在KO层对残差做无固定点排列，训练与留出查询各自排列；所有对照与真实臂容量匹配。
这只是模型条件化残差对照，不声称任意分布下的精确条件随机化检验。

## 内外层分工

外层固定五折KO留出，每折仅在约44个训练KO内部做四折筛选。
候选基因需同时满足：

- 原冻结U支持基因；内层OOF MSE相对S/c至少改善5%；
- 四个内折至少三个改善；且总OOF误差低于三套打乱；
- 正向家族的系数至少三个内折为正；自由家族不据符号筛选。

外层留出响应完全不参与该折的筛选、系数或程序轴拟合。
所有模型在被选基因上使用同一mask，未选基因仍用S基线；先冻结各折预测，再评分。
另在选中基因的内训练响应残差上做最多5条SVD程序轴，报告程序投影与全PCA结果作为次指标。
每折少于5个基因不算满足支持度，不事后把阈值调低。

主指标是按训练折Y方差标准化的被选基因MSE，再按KO等权；
增量=`1−mean(error_real)/mean(error_control)`，同时附逐KO平均相对增量。
至少三个外折、30个留出KO有足够候选，且相对S和三打乱的KO-bootstrap95%下界均>0，
才称“值得继续做velocity验证的训练内预测信号”。bootstrap只作条件级开发筛查，不是独立外部确认。
三家族并列报告，不把自由U的阳性自动称为物理velocity阳性；本轮没有单基因FDR发现保证。

最后才在全部56训练KO做一次固定四折筛选，与五个外折筛选频率≥3取交集，冻结下一阶段候选清单。
**最终交集不能回头用于外层评分**，因为它间接使用了全部训练标签。
没有基因则输出空清单，不人为添加标签或自动启动大训练。
原基因/PCA来自外训练折，内部CV条件于该冻结表征，并非把所有预处理重拟合的全流程外部验证。

## 完整路径

- 配置：`/home/yuchang/wangjiaxuan/configs/w15x_u_increment_screen_20261022_v1.json`
- 训练均值：`/data/yuchang/veloroute_w15w_rate_calibration_20261022/training_cv_v1/training_mean_features.npz`
- 训练原计数：`/data/yuchang/veloroute_w15v_target_estimator_20261022/fixed_estimator_v1/training_counts.npz`
- 折：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_fold_6to24_v1/`
- 公共条件：`/data/yuchang/veloroute_w15e_traxler_conditions_20261021/esm2_3b_v1/conditions.npz`
- 输出：`/data/yuchang/veloroute_w15x_u_increment_20261022/nested_screen_v1/`

CPU执行，不占训练卡，不训练GFG/router、不用JVP。
