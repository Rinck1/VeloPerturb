# Traxler：完整分布模型与单细胞velocity结果

2026-10-21。当前速度是解析S/U基线，不是GFG。模型为约39万参数的L3轻量版：
官方CellFlow骨干、K8路由专家、固定源侧q、独立Tsit5 ODE。无均值锚。

## 已完成的结果

主指标为同KO/seed配对平均相对Energy改善，正数为好；Energy采用相同PCA50量纲。
0→6h有13验证KO，6→24h有7验证KO；均三seed。6→24h实际跨度18h。

| velocity使用方式 | vs S+U | vs打乱velocity | vs U程序 |
|---|---:|---:|---:|
| 0→6h群体程序 | −0.139% | −0.297%（一套跨KO打乱） | −0.145% |
| 6→24h群体程序 | +0.149% | −0.079%（三套跨KO打乱均值） | −0.102% |
| 6→24h单细胞程序 | +0.401% | **+0.486%（三套KO内细胞打乱均值）** | +0.027% |

单细胞版相对S+U的三个seed增益为+0.725%、+0.386%、+0.091%。
相对最初单套细胞打乱为+0.489%；新增两套后仍为+0.486%，不是拿群体打乱替代单细胞对照。
其Energy均值0.156487，S+U为0.157121、U程序为0.156556，说明与强U表示仍几乎持平。
单细胞版相对群体velocity版再改善约0.246%，三个seed方向一致。

以上是已有开发KO上的小幅一致信号，不是独立复制、物理方向证明或真实命运标签恢复。
KOs在不同任务中的划分不同；两个任务均保留原测试KO目标不读取。

## 冻结模型的机制干预

在6→24h群体程序模型上，不重训、复用完全相同专家终态，只改变路由权重：

| 干预 | 相对原模型Energy误差增量 |
|---|---:|
| 关闭额外velocity程序槽 | +0.040% |
| 反向velocity程序 | +0.172% |
| 跨KO打乱velocity程序 | −0.268% |
| 将逐细胞q换成全源群体平均q | +21.588% |
| 均匀q | +26.432% |
| 关闭显式U特征槽（保留由U推导的velocity） | −1.108% |

原Energy逐项复现差为0。逐细胞专家分配对该模型很重要，但平均q破坏的也包括S/U状态与专家的对应，
不能把21.6%归因于velocity。输入置零/反向属于推理敏感性检查，不是生物因果干预。

## 独立L2与后续大规模验证

参数匹配的Traxler L2四臂×三seed已启动：原生CellFlow，不含router；与L3同源S/U/5维程序和ESM条件。
L2按参数数量选宽度256、384,594参数，L3为389,704，差1.31%；每模型5000步，未按预测分数选型。
本报告发布时L2尚在训练，不提前报告L2/L3胜负。相关新增脚本和GPU策略集中回归40项通过。

用户已要求Traxler大规模验证；新多折管线待接入，不能把尚未启动的CV说成已有结果。
拟对原开发KO做重复五折、每折重拟合HVG/PCA/velocity/响应轴，原保留测试KO继续封存。

## 服务器完整路径

- 6→24h群体四臂：`/data/yuchang/veloroute_w15h_traxler_program_router_20261021/full_distribution_v1_retry/`。
- 0→6h四臂：`/data/yuchang/veloroute_w15i_router_followups_20261021/queue_v1/cohort_0to6/`。
- 6→24h单细胞四臂：`/data/yuchang/veloroute_w15i_router_followups_20261021/queue_v1/cell_6to24/`。
- 冻结干预：`/data/yuchang/veloroute_w15i_router_followups_20261021/queue_v1/frozen_6to24/`。
- 群体三套打乱：`/data/yuchang/veloroute_w15j_gpu7_20261021/shuffle_controls_v1/`。
- 单细胞三套打乱：`/data/yuchang/veloroute_w15k_cell_shuffle_20261021/extra_shuffles_v1/`。
- 新L2：`/data/yuchang/veloroute_w15k_traxler_l2_20261021/matched_l2_v1/`。

各目录包含summary.json、CSV、冻结预测、模型与数据/代码哈希；大文件不上传Git。
