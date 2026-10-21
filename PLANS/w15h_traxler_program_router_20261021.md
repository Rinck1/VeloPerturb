# Traxler 完整响应程序 router：固定训练队列

2026-10-21，用户要求直接排队训练。不是重复 ridge 均值 probe，而是独立专家 ODE 的完整分布预测。

## 数据与任务

- 数据：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_fold_6to24_v1/`。
- 固定 KO 切分：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_trainable_v1/split_6to24.json`。
- 真实小鼠 ESM2-3B：`/data/yuchang/veloroute_w15e_traxler_conditions_20261021/esm2_3b_v1/conditions.npz`。
- 输出：`/data/yuchang/veloroute_w15h_traxler_program_router_20261021/full_distribution_v1_retry/`。
- 作业日志：`/home/yuchang/wangjiaxuan/outputs/w15h_program_router_jobs_20261021/router_gpu0_retry/`。

56训练KO、7验证KO、7保留测试KO；相同KO背景下观察6h源细胞，预测24h的刺激响应分布。
跨度18h，是early-response延续，不称为5h预测。沿用已固定的源S/U变换。
验证KO已用于此前开发；本轮仍为开发比较，不冒充独立确认。测试目标不读取。

## 四臂、三seed，共12次训练

四臂模型、初始参数、KO分层采样和训练预算相同，差别仅为源侧router的5个额外特征槽：

| 臂 | router输入 |
|---|---|
| S_U_router | S50 + 编码(U50, 零5) |
| S_U_velocity_program | S50 + 编码(U50, velocity程序方向5) |
| S_U_shuffled_velocity_program | S50 + 编码(U50, 打乱的velocity程序方向5) |
| S_U_U_program | S50 + 编码(U50, U程序方向5) |

训练KO的目标−源群体均值，减训练共同响应后，SVD定义5条响应轴。
各KO的源velocity均值减训练KO均值，投影至响应轴并单位化，作为群体上下文；U臂同样加工。
打乱臂仅在训练/验证各自KO集合内固定无自映射置换，保留源侧特征集合。
cell U按训练源细胞均值/标准差缩放；5维程序方向不逐维重标，零臂保留同样参数槽。

官方CellFlow隐藏骨干128×2，K=8专家；S-only表达OT，FM时钟0→1，生物源时间6h、增量18h显式进入条件。
router仅在源侧计算，权重固定；每个专家分别用Tsit5积分，输出加权终态分布。
采用原有专家期望FM损失和使用率正则，不强加velocity与运输方向cos约束。
无均值锚、原版GFG、JVP或新EM模块；当前velocity是解析S/U基线。
每臂seed0/1/2，各5000步，batch128，Adam lr3e-4，无根据本轮分数自动改参。

## 自动产出与资源

全84份验证预测先冻结再自动评分。主指标为加权完整混合分布Energy（除以sqrt(50)，与现有实现相同）；
另报群体均值MSE、响应轴均值MSE、方差比/对数方差误差、router有效专家数和熵。
主汇总为KO等权、seed等权的配对比较，不将三个seed当独立生物重复，不将专家编号称为真实命运标签。

GPU0运行，GPU2/3的RENGE L2/L3不动；GPU6/7保留，不停止其他项目。
大文件/缓存/临时文件写`/data`，后台作业6小时上限，可脱离对话继续。
结果落盘`RESULTS.md`、`metrics.csv`、`comparisons.csv`、模型/预测与源代码/数据哈希。

首启`full_distribution_v1`在模型初始化前因系统cuDNN9.5与JAX要求9.8冲突失败，未训练模型。
重启PID34799沿用RENGE已有`/data/yuchang/veloroute_w15a/runtime/cuda_deps/nvidia/cudnn/lib`，
不改模型/配置，不覆盖首启审计目录。

下一步根据真实velocity相对S+U、打乱和U程序特征的表现决定是否扩大，而非预先认定有独特方向优势。
