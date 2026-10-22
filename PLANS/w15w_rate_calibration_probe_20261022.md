# W15W：只用训练条件检查时间校准的有效剪接系数

2026-10-22，W15U/V两组仍在训练且本agent未读取其开发分数时固定。
目的不是再做一次平滑，而是检查原稳态γ是否不适合18h响应。

只读56个训练KO的6h源S/U与24h目标S；不读验证/测试目标，不读取目标U。
原S深度归一化、2000基因和PCA保留。
每KO每基因定义 A=mean[U/(1+S)]，B=−mean[S/(1+S)]，
Y=(mean log1p(S24)−mean log1p(S6))/18。
在训练KO的内部固定五折中，用中心化A/B预测中心化Y，加训练折Y均值作共同背景。
比较：共同均值、原γ残差×非负单系数、S单项、U单项、自由两系数、有正β/γ约束的两系数。
各源特征仅用内训练KO拟合RMS尺度，ridge=1；非负两维问题枚举内部解与边界，求精确凸最优。
另报告源S50/源S50+U50的ridge均值响应参考，采用训练折标准化，不做超参搜索。
量为 held-out-in-training-KO 的终态均值PCA MSE，按KO等权。

若时间校准版本平均优于共同均值、原γ残差和S单项，保留为值得另立下游实验的候选；
否则不据此新增正式模型网格。该筛查不是生物瞬时速度真值验证。
β/γ由训练条件终态监督，是任务校准的有效系数，不称为独立估计出的真实物理速率。
表征/PCA来自原外训练折，内部五折只分离响应标签；不是全预处理重拟合的独立外部CV。
没有JVP，没有旧基础模型，没有router训练，没有测试集筛选。

输入：
`/home/yuchang/wangjiaxuan/outputs/w15_traxler_fold_6to24_v1/{train_source,train_target,transform}.npz`
`/data/yuchang/veloroute_w15v_target_estimator_20261022/fixed_estimator_v1/training_counts.npz`
原始训练目标行来自 `/data/yuchang/veloroute_w4/traxler2025/processed_usa_v1/{C1,C2,C3}.h5ad`。
输出：`/data/yuchang/veloroute_w15w_rate_calibration_20261022/training_cv_v1/`。
