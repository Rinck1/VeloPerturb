# Traxler 大规模重复KO五折验证

用户要求在Traxler也进行大规模验证。配置在任何新CV拟合前固定。

## 规模与任务

两次固定重复（分割seed20261021/20261022）×五折×两任务×两模型族×六臂×三模型seed＝720次拟合。
共20个任务/重复/折，L2与L3各360模型，每模型5000步、batch128。

- 6→24h为主任务：原56训练+7验证＝63开发KO，原7测试KO继续保留；这是18h early-response延续。
- 0→6h为边界任务：原106训练+13验证＝119开发KO，原13测试KO继续保留。
- 每个KO在每个任务的两次重复中各留出一次；同任务所有源/目标细胞随KO分组，不拆细胞做伪泛化。
- 五折只按KO身份及固定随机种子生成，不按响应幅度、多峰程度或已有增益选择。

这些仍是已用于研究的开发KO，不因重分折就改称独立复制。不同任务保留各自原始测试集合，
模型/变换不跨任务共享；某KO在另一任务属于开发集不进入本任务的拟合。

## 每折重新处理，禁止复用原开发PCA

从原始S/U稀疏矩阵选取本折允许的源细胞。只在训练KO源侧拟合HVG2000、PCA50、gamma、
velocity尺度及U相关归一化。响应轴/训练参考由本折训练KO的目标−源均值构建。
源侧S/U共享S库深归一化，不复用旧gene_source中独立U归一化的接口。
验证源仅应用训练变换；验证目标只用于评分，不进入训练、路由、特征或模式定义。
训练包不含验证目标表达；评分包只保存原始目标的技术行定位，待完整预测冻结后读取。

## 模型与六臂

L2固定CellFlow256×2，384,594参数；L3固定K8/128×2，389,704参数，差1.31%。
继承已选学习率3e-4、OT epsilon0.1、固定5000更新、时间编码、真实ESM和Tsit5推理。
不在CV分数上重新选宽度、K、时间范围、超参、轴数或训练步数。

六臂：S+U基线、+单细胞velocity程序、+单细胞U程序、+三套独立固定KO内细胞打乱velocity。
四种表示与打乱沿用已验证的5维训练响应轴加工；各臂参数相同，source条件随OT源索引对齐。
L2无router/专家；L3固定源侧q，各专家独立ODE。当前仍是解析S/U速度，非GFG/JVP。

## 评分与统计单位

两族共720模型的全部验证预测冻结后才评分，不根据中途分数改变队列。
相同折/KO/seed用相同源ID和目标抽样，指标为加权Energy/sqrt(50)、均值与响应程序MSE、
方差误差，另报L3专家使用。跨折PCA不同，主比较使用同折配对改善。
每个KO先平均模型seed及两次重复，再以KO等权汇总，bootstrap重采样单位为KO，不是细胞、fold或seed。
比较velocity与S+U、三打乱均值、U程序，并输出同信息/近似参数匹配L2/L3对照。
95%区间作为开发性不确定度汇总，不把重叠训练折当20个独立生物重复。

## 路径与资源

- 配置：`/home/yuchang/wangjiaxuan/configs/w15l_traxler_repeated_cv_20261021_v1.json`。
- 元数据：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_trainable_v1/cells.csv.gz`。
- 原始S/U：`/data/yuchang/veloroute_w4/traxler2025/processed_usa_v1/{A1..C3}.h5ad`。
- 输出根目录：`/data/yuchang/veloroute_w15l_traxler_cv_20261021/repeated_5fold_v1/`。
- 子目录：`pack/folds/`、`L2/`、`L3/`、`evaluation/`；日志`/home/yuchang/wangjiaxuan/outputs/w15l_traxler_cv_jobs_20261021/`。

CPU准备20折，GPU0接续L2，GPU7跑L3；两卡忙时自动等待，不终止其他任务。
GPU2/3的RENGE与其冻结源码不变，GPU6保留。48小时上限，大文件与缓存写/data。
成功模型可按哈希复用，半成品保留且不覆盖，阴性结果原样落盘。
