# W15Z：以现成0→6h训练折复查方差通量

2026-10-22。W15Y的6→24h结果已知后固定，未计算本窗口分数。
这是一项新的开发探针，不能倒称为W15Y最初预注册的一部分。

复用W15Y全部方差通量公式、S/c控制、正向/自由/反向/直接U/三文库内打乱、
五折KO留出、单个共享α及相同判据，不调λ或挑轴。
只换到现成0→6h折的106个训练KO；13开发、13测试KO目标不读取。
输入8741个源细胞、8209个训练目标细胞，均为原冻结PCA50状态。
它和18h任务的KO集合及PCA不同，不能把两个百分比的差异全部归因为时间跨度。
这里只回答短窗口是否出现本窗口控制之上的正向方差信息，不声称完整多峰预测。

配置：`/home/yuchang/wangjiaxuan/configs/w15z_short_variance_flux_20261022_v1.json`
源/目标：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_fold_0to6_v1/{train_source,train_target}.npz`
切分：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_trainable_v1/split_0to6.json`
输出：`/data/yuchang/veloroute_w15z_short_flux_20261022/training_KO_v1/`
CPU运行，不新增神经模型、不用JVP。
