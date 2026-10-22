# 训练专用velocity辅助：四臂小实验

2026-10-22，用户要求实际测试velocity辅助训练。

## 与上轮的区别

上轮是训练输入含velocity、推理时关掉该输入。本轮所有臂训练和预测都不把velocity作为输入。
velocity只作为源侧辅助监督目标，辅助头挂在router共享隐藏层，训练后完整丢弃。
该损失不直接提供模式标签，不把RNA velocity当作流匹配目标或专家位移方向。

## 固定设计

Traxler 6→24h，56训练KO、7开发验证KO，原7保留测试KO不读取。
沿用原PCA、响应轴、真实ESM、K8、5000步、batch128、Adam3e-4及独立Tsit5专家ODE。
共同输入源S50、U50和5个常零槽；状态场仅使用移动S状态、扰动及时间。

四臂各三seed，共12个新拟合：

1. 无辅助：辅助权重0。
2. 真实velocity：拟合源细胞5维有符号单位velocity程序。
3. 打乱velocity：相同目标按训练KO内固定无固定点置换，置换不依赖模型seed。
4. U辅助：拟合源细胞相同响应轴上的5维单位U程序。

所有臂辅助头、训练参数量、初始化、采样及更新预算相同；无辅助臂也声明辅助头，但其梯度为0。
辅助预测来自router原64维共享隐藏层的线性64→5头，训练参数增加325；部署参数不增加。
辅助损失对router隐藏层与U编码器可反传，不直接更新专家、状态场或router的最终logits投影。
训练任务仍通过完整router更新logits，间接利用学到的表征。

目标在OT重采样后必须跟随源索引，包括重复源索引；扰动条件不能进入辅助预测头。

## 损失与判据

`L = 原FM与balance损失 + D × λ × mean((辅助预测 − 源侧程序目标)^2)`。
固定D=50、λ=1；等价于每维FM与每维辅助MSE等权，避免FM的维度求和影响尺度理解。
不扫描λ或用开发结果挑权重。本轮无辅助臂λ=0，其他三臂统一λ=1。

主要看真实velocity相对无辅助、打乱辅助、U辅助的同KO/seed Energy配对增量。
同时报告均值、方差误差、逐seed及逐KO结果，不将训练重建辅助目标的成功当作任务增益。
12模型全部部署预测冻结后才读取开发目标评分，不用于早停或改变计划。

## 路径

- 配置：`/home/yuchang/wangjiaxuan/configs/w15n_velocity_auxiliary_20261022_v1.json`
- 数据：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_fold_6to24_v1/`
- 父定义：`/data/yuchang/veloroute_w15i_router_followups_20261021/queue_v1/cell_6to24/`
- 输出：`/data/yuchang/veloroute_w15n_auxiliary_20261022/four_arm_v1/`
- 日志：`/home/yuchang/wangjiaxuan/outputs/w15n_auxiliary_jobs_20261022/gpu0/`

仍是解析velocity，不是GFG重训，也不是命运教师蒸馏。使用空闲GPU0，不干扰其他项目或保留GPU6。
