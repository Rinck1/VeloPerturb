# W5 补充预注册：scNT-seq 分钟级动力学验证

状态：**已冻结**（2026-10-17；在读取本数据任何响应、拟合速度或预测前登记）

冻结提交由运行 manifest 的 `freeze_commit` 标识。本文件与 `configs/w5_scntseq_minutes_v1.json` 同一提交冻结；后续改变另记新版本，不改本版本。RENGE 的 H1/H7 结果不受此分析影响。

## 问题与数据

问题是：在已启动 KCl 响应后，当前 U/S 构造的动力学是否帮助预测接下来几十分钟的群体 S 状态变化？这是独立时间系列的描述性机制验证，不是多扰动留出或谱系命运检验。

- 数据：GSE141851，配对产物 `/data/yuchang/veloroute_w4/scntseq/processed/scntseq_geo_labeling_dynamo_su_exact.h5ad`。
- SHA256：`208cc0b266d5a6d5bfbb18834f9f153402a7a3c93aaa9c9e4c27e29c96cefbd1`。
- 固定 13476 细胞、37007 基因的作者 Ex 命名 S/U 输入与 GEO 精确交集；不添加响应基因或 activity gene 筛选。0/15/30/60/120 min，只有一条刺激系列。
- 每个时点只用 S 总计数≥500、U 总计数≥20 的细胞，阈值依赖预先已有技术 metadata。无细胞类型/响应幅度/未来表现筛选。不因结果更换阈值。
- 同细胞另有 new/old/total。GEO total=new+old，但不等于另一管线的 S+U。new 是 2 小时标记窗读数，只用作辅助特征，不称为瞬时净速度。
- 来源：[GEO GSM4512670](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4512670)；上游队列/计数区别见 `docs/w4_external_data_audit.md`。

## 固定表示

1. 只用 0 min 的合格细胞拟合源变换：全 37007 基因 S 文库归一到 1e4，U 使用同一个 S 文库因子；S 方差选 2000 HVG，log1p S 的 PCA-50，双尾 5% OLS 拟合 γ；β固定1。复用 W4 解析实现，不用 GFG/JVP，不训练神经头。
2. 0 min 变换随后冻结，任何刺激后细胞不参与 HVG/PCA/γ 拟合。γ 是相对比率，真实分钟尺度由下面唯一标量校准，不能说 γ 已标定成物理降解速率。
3. 目标状态为细胞等权 `mean(PCA(log1p(S_norm)))`。
4. `vel_mean` 为逐细胞 `W[(U_norm−γ S_norm)/(1+S_norm)]` 的等权均值，是本轮主要速度臂。`vel_pool` 先合并 raw S/U、全基因 S 深度归一后计算相同导数，是次要群体特征，不冒充上述均值的精确导数。
5. `direct_U_raw` 为 `W log1p(U_pool_norm)`；`direct_U_delta0` 为它与 0 min 同管线特征之差。二者都报告，不能只选择较弱者。
6. new 特征按 GEO total 全基因文库因子归一，映射 `W log1p(new_pool_norm)`；另报相对 0 min 的差。不得把 new 与另一管线的 S 相减当净速度。

## 校准与预测

每个需校准的臂只有 **一个非负标量**：

`a = max(0, <f_15, (zbar_30−zbar_15)/15> / <f_15,f_15>)`。

若分母≤1e−12，a=0并记录；不翻转速度、不按测试集选择 a，也不加截距/高维矩阵。

| 臂 | 特征/参数 |
|---|---|
| identity | 预测当前状态，不拟合参数 |
| past_S | `(zbar_t−zbar_previous)/(t−previous)`，物理分钟尺度外推，不拟合参数 |
| past_S_calibrated | 同上，但用 15→30 校准非负 a |
| vel_mean | 逐细胞速度均值，校准一个非负 a；主要速度臂 |
| vel_pool | 群体池化动力学，校准一个非负 a |
| direct_U_raw | 原始 U 表征，校准一个非负 a |
| direct_U_delta0 | 基线中心化 U 表征，校准一个非负 a |
| new_raw | 标记 RNA 代理，校准一个非负 a；辅助 |
| new_delta0 | 基线中心化标记 RNA 代理，校准一个非负 a；辅助 |

不训练 ESM 条件编码、router/专家或原版 GFG。0→15 不作为必须预知外加刺激的预测考题。15→30 是校准区间，不算测试成绩。

固定两个滚动区间：**30→60、60→120**。先只读取 0/15/30 的计数并冻结第一区间预测，再读取 60 的源计数并冻结第二区间预测；标量仍不更新。60 min 是第一区间目标及第二区间当前源的合法复用，不声称整条预测流程从未读取 60。120 min 的表达只由独立评估入口读取。每个预测为 `zbar_t + Δminutes * a * f_t`，identity/past_S 按表处理。

## 指标与结论

- 每个区间/每个臂报告 PCA-50 MSE、相对 identity 的 SSE 降幅、预测位移与真实位移的 cosine，以及校准尺度/特征模长。零位移 cosine 记空值，不填0当实际方向证据。
- 两区间等权报告 MSE 与相对增益，不把50个PC、基因或细胞当独立实验重复。首轮不算显著性或生物学 CI。
- 主要描述性判据：vel_mean 两个区间都优于 identity，且两区间平均 MSE 同时低于 past_S 和 past_S_calibrated，记为“本系列有短时速度预测优势”；否则记未建立。
- vel_mean 与两种 direct U 均并报。只有还优于两种 U 对照，才描述为“此固定低容量预测器下速度构造优于直接 U”，不是新增信息或因果证明。
- new 两臂只说明另一种动态读数的预测能力，不是 velocity 真值。vel_pool/vel_mean 不因结果选一个改主臂。
- 全部结果与阴性结果落盘。没有新未用时间段；看过本轮后再改模型只能标探索性。

## 实施状态

冻结时只有技术数据配对/QC，尚未执行本轮科学分析。先实现并通过合成流水线测试，再按本冻结配置执行一次。实现错误须记修正原因，不能按真实结果调参数。
