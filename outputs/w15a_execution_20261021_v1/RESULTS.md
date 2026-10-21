# W15-A：完整L2结果与L3自动接续

2026-10-21约12:07 CST状态快照；后续进度以各运行目录为准。

CellFlow固定18模型、216窗口已完成并评分。GPU0冻结输入干预也已结束，
72次新推理、复用36次原预测，未训练新模型、未修改原成绩。
完整开发结果及其独立干预报告见`docs/current/w15a_l2_results.md`与
`docs/current/w15a_l2_token_intervention.md`。

## 已启动的自动接续

- L3控制器：PID195386，CPU，最长24h；启动UTC为2026-10-21T04:06:52。
- 作业记录：`outputs/w15a_jobs_20261021/l3_real_controller_v1/`。
- 实时状态：`/data/yuchang/veloroute_w15a/l3_registered_real_pipeline_v1/status.json`。
- 先等完整120-fit合成通过，随后仅用物理GPU0执行固定18训练、30份预测清单、360窗口；
  GPU进程退出后交CPU统一评分。失败则记录原因，不自动调参或重试。
- 合成机制等待器：PID99758，CPU，最长24h；作业记录
  `outputs/w15a_l3_synthetic_mechanism_jobs_20261021/wait_cpu/`。
  完整120-fit/480候选验证后分析专家几何与路由；原gate失败不阻断描述分析。

此时合成19/120、RENGE196/576。没有L3真实数据成绩，未宣称合成已通过。
原配置、预算、损失、配对与合成判据未修改；GPU6–7保留，不停止其他项目。

## 软件验证

新增L1共样本对齐、源特征、L2报告、L3真实接线与合成机制分析联合CPU回归46项通过。
冻结token干预另6项CPU测试通过。不是完整仓库测试声明，也不是生物学阳性证据。

源计数、细胞级预测和模型权重全部留在原机/data盘，不上传GitHub。
