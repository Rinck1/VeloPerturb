# W14-B3：RENGE 固定端点配方的一次性探索迁移

2026-10-18。执行前提交本协议；仅探索，不修改W4 H1/H7判决。

## 任务与预算

- 固定W4共同16TF，不重选亚群；每个TF分别留出day2→3和day3→4。
  两区间各自拟合，不把时间段当独立生物重复。合计32个fold。
- 源是已经接受扰动的当日细胞，不是“对照出发”的de-novo扰动预测。
- 每fold固定b1端点K8/76,248参数，12k步、LR3e-4、batch64、V-statistic、batch中心化、
  直接距离核与原正则。3seed，不续训，不根据测试分数选检查点或重跑。
- 四种语义臂：static、real、direct_U、shuffled_v。打乱3个置换种子分别训练/报告。
  每fold18个作业，共576个12k训练作业。各臂共享同一冻结W4 vel_pool均值锚。

## 源变换与均值

- 复用已冻结`outputs/w4_h1_predictions_v1/`各fold的seed0源变换，并验证其fold_manifest
  中的sha256；PCA50、HVG2000、gamma完全不改，训练/留出行沿用其清单。
- 数据与ESM身份沿用`outputs/w4_protocol_lock_v1/protocol_lock.json`，不重新选择TF。
- 在每fold训练TF（及原允许对照）的源/目标上，按原W4 vel_pool设计矩阵和ridge内部CV
  重现均值训练流程。留出锚应与既有W4 vel_pool预测一致，容差1e-5。
- 专家不额外接入ESM，以固定W13架构做early-response迁移；条件信息仅通过共享W4均值
  和源S/U状态体现。不是检验“显式条件路由器”的新架构。
- 局部特征在每个源TF群体中做S-PCA近邻池化，k=min(64,N)。real为log-PCA导数
  W[(U−gamma*S)/(1+S)]，归一化用全基因S深度。
  本任务没有同TF未扰动D3参考，使用零参考场；共同常数参考在训练特征标准化中抵消。
- direct_U为相同近邻池、相同深度因子的W log1p(U)，不使用gamma。
- 路由归一化只从该fold训练TF源拟合；留出TF的源只用于变换/推理，目标不进入训练。

## 评估与解释

- 先保存全部576作业的预测和哈希，再统一评分；每组至多512源/512目标、不放回，seed0。
  RENGE常少于512，使用全量而不是重复细胞凑数。3训练seed等权混合，置换种子分别报告。
- 精确混合energy为主；逐TF、逐区间、逐臂全部报告，先两区间平均再TF等权。
- 报real相对static、每个shuffled、direct_U的改善；不进行显著性确认，不根据结果改计划。
- 失败只表示固定配方迁移没有成功；不撤销sci-FATE2的开发结果，不宣称velocity普遍无信息。
- 输出`outputs/w14b_renge_router_v1/`，报告`docs/current/w14b_renge_results.md`。
  bug重跑须新目录和偏离记录，不覆盖旧输出。

## 排队

GPU1–3优先给W14。B3可用CPU先准备源侧pack；训练等W14评分结束并释放三卡后启动。
不使用GPU0或4–7。Traxler只做技术准备，不在本轮解封新数据生物效应或训练W15。
