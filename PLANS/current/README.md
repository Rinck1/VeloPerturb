# 当前计划入口

最新运行状态见 [STATUS](../../docs/current/STATUS.md)。W14主评分已完成并补推，
RENGE恢复训练中；Traxler九份S/U矩阵已完成，技术分样继续。

- [W15-A三条模型线](../w15a_cellflow_kinetic_plugin.md)：L1小模型、L2条件插件、L3双流router专家；
  L2/L3开发多agent并行，正式配置按依赖冻结。
- [W15-B群体分支可行性](../w15b_population_branching.md)：先定义和描述，未过门槛不训练BR-3。

- [短时群体预测基线复现实验方案](short_time_baseline_protocol_20261018.md)：
  CellFlow、CellOT、直接 U/velocity 输入扩展、六个小时级窗口、共享均值结构比较，
  附 sci-FATE2、RENGE、Traxler 的绝对路径和已有模型入口。方案已记录，新增基线尚未开训。
  CellFlow三线的调参、token与不加均值锚规则以10月19日W15-A为准，旧方案仅保留为基线和路径参考。
- 最终 ODE 路由机制实验与方法/结果初稿仍待推进，不能将旧端点干预当作最终 ODE 的新结果。

W13执行规格为 [根目录保留协议](../w13_time_dynamics.md)，其中本轮1/2/4已经完成。
本页导航更新本身不启动训练，也不更改已冻结的运行协议。
