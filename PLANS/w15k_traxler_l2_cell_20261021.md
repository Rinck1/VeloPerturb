# Traxler：参数匹配L2与单细胞三套打乱

用户2026-10-21确认开始。两条并行任务共18新拟合，沿用6→24h的18h early-response任务。

## GPU0：独立L2，12模型

原生官方CellFlow OTFM，不含router/专家。输入和L3相同：固定源S50、训练源标准化U50、
5个单细胞响应程序槽、真实ESM2-3B及源时间6h/增量18h。
源S作为显式冻结条件提供给L2，避免让其仅从移动中的x(t)间接猜初始状态。
四臂CF_S_U / +velocity程序 / +同KO细胞打乱 / +U程序，各seed0/1/2。
均5000步、batch128、Adam3e-4、表达OT epsilon0.1，逐粒子原生Tsit5 ODE。
源条件随OT源索引重排，推理固定出发时刻条件。

L3父模型约389,704参数。L2只按参数数量在预先声明的width128–256、步长8中选最近宽度，
深度、时间编码、条件编码维度和训练预算不变，差距必须<5%；选择不读取验证预测分数。
全部84份L2验证预测冻结后评分，源ID与L3逐一对齐；输出L2内部增量和L2/L3直接对照。

## GPU7：单细胞velocity额外打乱，6模型

针对已完成的单细胞L3父模型，补训within-KO cell permutation1/2各三seed。
保留每个KO的单细胞方向集合，只破坏其与源细胞的对应关系；不同于此前跨KO群体摘要打乱。
父真实velocity、S+U和U程序模型复用，不重训。最终与permutation0等权汇总。

## 数据、输出、约束

- 数据：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_fold_6to24_v1/`。
- 切分：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_trainable_v1/split_6to24.json`。
- 蛋白：`/data/yuchang/veloroute_w15e_traxler_conditions_20261021/esm2_3b_v1/conditions.npz`。
- L3父模型：`/data/yuchang/veloroute_w15i_router_followups_20261021/queue_v1/cell_6to24/`。
- L2输出：`/data/yuchang/veloroute_w15k_traxler_l2_20261021/matched_l2_v1/`。
- 单细胞打乱输出：`/data/yuchang/veloroute_w15k_cell_shuffle_20261021/extra_shuffles_v1/`。
- 日志：`/home/yuchang/wangjiaxuan/outputs/w15k_traxler_jobs_20261021/{l2_gpu0,cell_shuffle_gpu7}/`。
- 配置：`/home/yuchang/wangjiaxuan/configs/w15k_traxler_l2_cell_20261021_v1.json`。
- 脚本：`/home/yuchang/wangjiaxuan/scripts/phase0/w15k_traxler_l2_cell.py`。

56训练KO、7验证KO、7保留测试KO；不读取测试目标。验证已用于开发，不冒充独立确认。
主指标为加权完整分布Energy/sqrt(50)，另报均值/响应程序MSE与方差误差，KO等权、seed等权。
GPU0/7占用时后台等待，不终止其他项目。GPU6保留，RENGE GPU2/3及其冻结源码不动。
大文件/缓存写/data，训练8h上限，排队最多12h，自动落盘模型、预测、CSV、RESULTS.md及来源哈希。

本轮仍使用解析S/U速度。目标域重训、无JVP速度估计器另列下一阶段，不能将本轮称为GFG结果。

## 实际启动记录

15项相关测试通过，包含官方CellFlow两步OT训练、保存恢复、逐细胞ODE、条件重索引和参数计数。
按参数数量选定L2宽度256、参数384,594，较L3的389,704少1.31%；未读取验证分数选型。
GPU0启动器PID753617，GPU7启动器PID753879，两条均已实际进入训练更新。
原生CellFlow训练日志loss按其默认维度平均，L3损失按维度求和；不直接比较两者loss数值，
只比较相同源ID、相同目标样本、相同PCA量纲的终态Energy及其他统一指标。
