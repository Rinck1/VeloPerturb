# VeloRoute

**当前主线：RNA动力学上下文引导的短时群体分布预测。**

2026-10-18：W14跨重复验证已启动；RENGE探索迁移在其后自动排队，Traxler原始数据下载并行。
实时执行状态和边界见 [当前状态](docs/current/STATUS.md)。以下结果仍为已完成的W13，不混作W14新成绩。

源侧S/U经解析邻域动力学估计进入soft router，校准混合响应模式配比；
零均值专家与小型群体耦合ODE预测分布，保留独立的冻结均值分支。
当前阳性来自sci-FATE2分化序列，不是已完成的未见遗传扰动泛化。

## 从这里开始

- [当前状态：唯一最新入口](docs/current/STATUS.md)
- [当前方法与数学定义](docs/current/w13_time_model.md)
- [最新实验结果](docs/current/w13_time_results.md)
- [完整研究导航与目录分类](docs/README.md)
- [错误、撤回主张与正确替代表述](docs/corrections/CLAIMS.md)
- [交接入口](HANDOFF.md)

## 当前结果

最终小ODE为76,760参数，三seed、固定模型、3小时最大推理步长。
下面是三seed预测分布混合后的energy相对改善，不把随机seed当生物学重复。

| 预测视野 | Energy ↓ | vs 同类静态路由 | vs 打乱动力学 | vs 匹配端点时间版 |
|---|---:|---:|---:|---:|
| 5h | 0.589504 | +3.723% | +4.620% | +1.226% |
| 10h | 1.027032 | +6.358% | +7.418% | +2.821% |
| 15h | 1.007833 | +2.415% | +1.573% | +0.027% |

ODE改善短时绝对拟合，但没有扩大动力学相对静态的差距；15h没有实质ODE额外收益。
“静态”指局部router对照，各臂共享冻结的解析均值，不是整条模型不读U。
当前动力学不是原版GFG、decoder JVP，也不是已标定的单细胞物理速度。

[最终机器结果与CSV](outputs/w13_time_program_v1/final_report/RESULTS.md)

## 代码与复现

当前实现位于 `scripts/phase0/w13_time_model.py`、`w13_time_training.py`、
`w13_frozen_horizons.py`；冻结均值及前期端点代码保留在W5–W10脚本中。
旧 `src/veloroute` 全量GFG/EM/门控实现属于历史路线，不是当前小ODE工作版。

测试请在已安装项目依赖的Python环境中运行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m pytest \
  tests/test_w9_population_router.py tests/test_w10_recipe_bridge.py \
  tests/test_w11_cohort_router.py tests/test_w12_privileged_router.py \
  tests/test_w12_source_mass_projection.py tests/test_w13_time_model.py -q
```

最近一轮上述针对性测试为39项通过。模型权重、原始矩阵和环境目录不在GitHub归档中，
源码与结果记录不等于已打包所有运行资产。严禁覆盖已有输出目录；原运行代码快照见各次
`outputs/**/code_snapshot/`，协议原路径见 [计划导航](PLANS/README.md)。

## 历史与记录

[公开记录清单](docs/history/record_exports/research_archive_20261017.md)中的4,881指文件数，
不是实验数，其中3,190是重复保存的代码/配置快照。原始实验记录与哈希不因本次整理改写。
历史报告中的“当前”“下一步”均按其原日期理解；最新待办以当前状态页为准。
