# Traxler 补充队列：短时、单细胞与群体、冻结路由机制

2026-10-21，用户要求需要跑的实验直接排队。本轮不改父实验、RENGE冻结代码或测试集。
父实验12/12完成，6→24h程序velocity相对S+U配对平均Energy改善0.149%，
相对打乱0.025%，相对U程序特征−0.102%；结果不足以支持扩大参数。
因此补充以下明确缺口，不做按结果自动调参或条件筛选。

## 队列顺序

1. 冻结6→24h模型：原始q、关闭velocity特征、反向velocity、跨KO打乱velocity、
   群体平均q、均匀q、关闭U特征。仅重算router权重，复用完全相同的专家ODE终态。
   不训练、不重新积分，先冻结147份权重预测再评分，原始Energy必须复现父实验。
   说明模型在推理时依赖哪些信息，不将输入反向/置零当作生物因果证据。
2. 0→6h真正短时任务：106训练KO、13验证KO、13保留测试KO；
   同样四臂S+U / +velocity程序 / +跨KO打乱程序 / +U程序，各三seed，共12新拟合。
   与此前0→6h的原始cos先验pilot不同：使用与6→24h一致的强S+U主输入、程序特征和5000步预算。
3. 6→24h单细胞程序：与父实验同轴、同归一化、同种子；将群体均值程序替换为每颗源细胞程序。
   新训velocity、KO内部细胞打乱velocity、U程序三臂×三seed，共9新拟合。
   严格复用父实验的三个S+U零程序基线，不重复训练；验证初始参数、源ID和响应轴一致。
   与父实验比较可观察群体摘要或单细胞异质信息哪种更有效；两种打乱分别针对不同层级。

合计21新拟合、3复用模型；骨干128×2、K8、batch128、lr3e-4、每拟合5000步。
生物时间与增量按真实任务输入，ODE时钟0→1。解析S/U velocity，非GFG/JVP。
不新增硬cos约束、EM或18M扩容，不读取测试目标。

## 路径与资源

- 配置：`/home/yuchang/wangjiaxuan/configs/w15i_program_router_followups_20261021_v1.json`。
- 脚本：`/home/yuchang/wangjiaxuan/scripts/phase0/w15i_program_router_followups.py`。
- 父实验：`/data/yuchang/veloroute_w15h_traxler_program_router_20261021/full_distribution_v1_retry/`。
- 0→6h数据：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_fold_0to6_v1/`。
- 6→24h数据：`/home/yuchang/wangjiaxuan/outputs/w15_traxler_fold_6to24_v1/`。
- 输出：`/data/yuchang/veloroute_w15i_router_followups_20261021/queue_v1/`。
- 日志：`/home/yuchang/wangjiaxuan/outputs/w15i_router_followup_jobs_20261021/gpu0/`。
- CPU等待器：`/home/yuchang/wangjiaxuan/outputs/w15i_router_followup_jobs_20261021/controller/`；
  状态：`/data/yuchang/veloroute_w15i_router_followups_20261021/dispatcher_v1/status.json`。

GPU0在父作业结束后已有其他项目占用，CPU等待器不初始化CUDA、不终止其他进程；
等GPU0没有计算进程且显存不高于256MiB时自动接续。GPU2/3原RENGE队列继续，GPU6/7保留。
排队最多24小时，GPU训练队列8小时上限，自动训练、冻结、评分，输出CSV/RESULTS.md和数据/模型/代码哈希。
主指标为同口径加权混合Energy，另报均值/响应轴MSE、方差误差和路由使用率。
开发比较中的小幅差异如实报告，不将seed当生物重复或将统计响应轴称为真实谱系。

排队前14项相关测试通过。CPU等待器PID284893已实际启动；GPU0当前属于其他任务，状态为等待空闲，
并非已开始21次新增训练。等待器与GPU worker均为脱离会话的持久后台进程。

## 追加资源授权与独立GPU7任务

随后GPU0已接续运行，用户再明确授权GPU7。GPU0原队列不重复/不重启；GPU7并行补训父实验
两套额外跨KO打乱（permutation1/2各三seed，共6新拟合），与原permutation0等权报告。
主模型/S+U/U程序9个模型复用，只训练打乱臂；仍5000步、K8和相同初始参数。

- 配置：`/home/yuchang/wangjiaxuan/configs/w15j_gpu7_shuffle_controls_20261021_v1.json`。
- 日志：`/home/yuchang/wangjiaxuan/outputs/w15j_gpu7_jobs_20261021/shuffle_controls/`。
- 输出：`/data/yuchang/veloroute_w15j_gpu7_20261021/shuffle_controls_v1/`。
- 启动器PID509543；新增白名单授权7但继续禁用6，9项资源策略测试通过。

GPU7任务4小时上限、自动冻结和评分；共享launcher文件未改，避免触发RENGE运行中的源码哈希锁。
