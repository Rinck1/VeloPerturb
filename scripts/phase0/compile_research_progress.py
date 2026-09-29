"""Assemble completed development results without rerunning/choosing experiments.

Only frozen development summaries/predictions are read; no confirmation pack is
opened. Source output files are hash-checked against their completed provenance.
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

from veloroute.artifacts import Run, read_csv, save_csv, save_json, sha256


def compile_report(output):
    roots = {name: Path('outputs')/folder for name, folder in {
        'velocity_probe': 'veloroute_velocity_exploratory_20260913',
        'minimal_router': 'veloroute_router_increment_20260913',
        'full_training': 'veloroute_full_real_exploratory_20260913_seed0',
        'full_gate': 'veloroute_full_development_evaluation_20260914_seed0',
        'training_quality': 'veloroute_training_quality_20260913',
        'expert_audit': 'veloroute_expert_separation_20260914',
        'frozen_candidates': 'veloroute_frozen_candidates_20260914',
        'cpu_full_smoke': 'veloroute_full_smoke_cpu_20260913',
        'gpu_full_smoke': 'veloroute_full_smoke_gpu_20260914',
    }.items()}
    summaries, inputs, inventory = {}, [Path(__file__).resolve()], []
    for name, root in roots.items():
        provenance = json.loads((root/'provenance.json').read_text())
        if provenance['status'] != 'complete':
            raise ValueError(f'{name} is not a completed run')
        inputs.append(root/'provenance.json')
        for filename in ('summary.json', 'decision.json', 'metrics.csv', 'comparisons.json',
                         'candidate_separation.csv', 'frozen_predictions.json', 'RESULTS.md'):
            path = root/filename
            if path.exists():
                if provenance['outputs'][filename] != sha256(path):
                    raise ValueError(f'Completed output changed: {path}')
                inputs.append(path)
        if (root/'summary.json').exists():
            summaries[name] = json.loads((root/'summary.json').read_text())
        inventory.append(dict(experiment=name, directory=str(root), status='complete',
                              provenance_sha256=sha256(root/'provenance.json')))
    junit = Path('outputs/veloroute_full_tests_20260914_v2/junit.xml')
    tests = ET.parse(junit).getroot().find('testsuite').attrib
    if any(int(tests[key]) for key in ('failures', 'errors', 'skipped')):
        raise ValueError('The reported test suite did not fully pass')
    inputs += [junit, Path('README.md'), Path('docs/veloroute_full_runbook_20260914.md'),
               Path('docs/veloroute_execution_plan_v1.1.md'),
               Path('docs/veloroute_full_scope_amendment_20260913.md')]
    config = dict(report_date='2026-09-14', scope='completed_exploratory_development_only',
                  confirmation='sealed_not_opened', metric='energy_V_statistic_not_square_rooted',
                  no_new_model_selection=True)
    with Run(output, stage='research_progress_compilation', kind='engineering',
             config=config, inputs=inputs) as run:
        comparisons, rows, arm_means = [], [], []
        for name in ('velocity_probe', 'minimal_router', 'frozen_candidates'):
            for record in json.loads((roots[name]/'comparisons.json').read_text()):
                comparisons.append(dict(experiment=name, real_arm=record['real_arm'], control_arm=record['control_arm'],
                    gain=record['gain'], ci_low=record['ci_low'], ci_high=record['ci_high'],
                    confidence=record['confidence'], n_conditions=record['n_conditions'], n_seeds=record['n_seeds']))
        for name in ('velocity_probe', 'minimal_router', 'full_gate', 'frozen_candidates'):
            by_arm_condition = defaultdict(list)
            for metric in read_csv(roots[name]/'metrics.csv'):
                rows.append(dict(experiment=name, arm=metric['arm'], condition=metric['condition'], seed=metric['seed'],
                    energy_distance=metric['energy_distance'], variance_ratio=metric.get('variance_ratio', ''),
                    condition_centered_gene_mean_mse=metric.get('condition_centered_gene_mean_mse', '')))
                by_arm_condition[metric['arm'], metric['condition']].append(float(metric['energy_distance']))
            for arm in sorted({arm for arm, _ in by_arm_condition}):
                means = [np.mean(values) for (a, _), values in by_arm_condition.items() if a == arm]
                arm_means.append(dict(experiment=name, arm=arm, mean_energy_distance=float(np.mean(means)),
                                      conditions=len(means), aggregation='seeds_then_equal_conditions'))
        routing = []
        for seed in (0, 1, 2):
            path = roots['frozen_candidates']/f'real_seed{seed}/predict/predictions.npz'
            reference = roots['frozen_candidates']/f'static_seed{seed}/predict/predictions.npz'
            for prediction in (path, reference):
                prov = json.loads((roots['frozen_candidates']/'provenance.json').read_text())
                if sha256(prediction) != prov['outputs'][str(prediction.relative_to(roots['frozen_candidates']))]:
                    raise ValueError('Source-side frozen prediction changed')
            with np.load(path, allow_pickle=False) as a, np.load(reference, allow_pickle=False) as b:
                q = a['q']
                routing.append(dict(seed=seed, real_mean_entropy=float(-(q*np.log(np.maximum(q, 1e-12))).sum(1).mean()),
                    real_static_q_mean_L1=float(np.abs(q-b['q']).sum(1).mean()),
                    real_static_sampled_mode_disagreement=float((a['modes'] != b['modes']).mean())))
        save_csv(run.directory/'metrics.csv', rows)
        save_csv(run.directory/'comparisons.csv', comparisons)
        save_csv(run.directory/'arm_means.csv', arm_means)
        save_csv(run.directory/'experiment_inventory.csv', inventory)
        save_csv(run.directory/'frozen_router_source_sensitivity.csv', routing)
        save_json(run.directory/'source_summaries.json', summaries)
        training, gate, quality = (summaries[key] for key in ('full_training', 'full_gate', 'training_quality'))
        summary = dict(status='COMPLETED_EXPLORATORY_ROUND_REPORTED', tests_passed=int(tests['tests']),
            full_training_seeds=1, velocity_probe_seeds=3, minimal_router_seeds=3, frozen_candidate_seeds=3,
            full_deployment_parameters=training['deployment_parameters_including_frozen_fallback'],
            formal_G1='NOT_RUN', velocity_router_benefit='NOT_ESTABLISHED',
            learned_gate_static_noninferiority='NOT_ESTABLISHED', confirmation_evaluated=False,
            next_step='fixed_version_train_only_VeloVI_adapter_audit_then_one_fixed_budget_estimator_comparison')
        save_json(run.directory/'summary.json', summary)
        lines = [
            '# VeloRoute 研究进度与交接结果（2026-09-14）', '',
            '全量框架已经实现并在真实 RENGE day4→day5 上完成训练。当前结果不支持 velocity/router 的正增益；工程成功与科学主张尚未成立必须分开。', '',
            '这是用户授权的开发侧探索，不是正式 G1 判决。14/4/5 TF 分属训练/开发/最终确认；4 个开发 TF 为 LIN28A、NANOG、POU5F1、ZIC3。最终确认条件未读效果、未用于拟合或调参。', '',
            '## 1. 本轮完成了什么', '',
            f'- {summary["tests_passed"]} 项回归全部通过；CPU 完整配置文件闭环及修复后 GPU 完整配置闭环通过。',
            '- 可训练状态编码器、动力学编码器/rho、set-attention、router/专家、延迟老师/EM、参考场/沿途代价、内源场、门控/corruption、自适应 K、模式内噪声、ODE 与完整恢复均已接通。',
            '- 真实全模型 A=1,000 / B=2,000 / C=12,000 步完成，557 次老师更新、3,000 个 corruption batch，8 个模式保持活跃；活跃不代表具有真实命运含义。',
            '- A/B 七臂×3 seeds（21 次预测）；最小五训练臂×3 seeds 加两简基线（21 次预测）；固定候选四臂×3 seeds（12 次预测）；full 三种门控干预（3 次预测）。每个实验族全部预测冻结后才读开发目标。',
            f'- 部署参数 {training["deployment_parameters_including_frozen_fallback"]:,}，其中额外冻结静态回退 {training["frozen_fallback_parameters"]:,}；训练期老师另有 {training["teacher_training_only_parameters"]:,}。full 尚不是严格参数匹配的静态比较。',
            '- 框架 scratch 训练；实际参考 velocity 是训练折 steady-state residual，不是 GFG/VeloVI。真实组合、多时间点和 de novo 尚未验证。', '',
            '## 2. 主要增量比较', '',
            '主指标为固定 PCA-50 的 energy V-statistic（不取平方根），越低越好。gain = 对照误差 − real 误差，正数才是增益。先平均 seed、再按条件 bootstrap 10,000 次；下表区间按各实验族固定比较数量做 Bonferroni 调整。只有 4 个开发条件，区间精度与可推广性有限，不能把 seeds 当额外生物重复。', '',
            '| 实验 | real / 对照 | gain | 调整后区间 | 置信水平 |',
            '| --- | --- | ---: | --- | ---: |',
        ]
        for comparison in comparisons:
            lines.append(f'| {comparison["experiment"]} | {comparison["real_arm"]} / {comparison["control_arm"]} | '
                f'{comparison["gain"]:+.8f} | [{comparison["ci_low"]:+.8f}, {comparison["ci_high"]:+.8f}] | '
                f'{100*comparison["confidence"]:.3f}% |')
        lines += [
            '', 'A0=S，A1=S+real v，A2=S+shuffled v；B1=S+U，B2=S+训练模型预测 U，B3=S+条件化置换残差 U。原始 CSV 和每条件结果全量保留。', '',
            '解读：当前简单 velocity 与 U 没有正增量；最小 real-router 相对静态的差异约 10⁻⁵ 且方向为负，没有实际改善。比 identity 好说明学到了时间变化，不能归因于 velocity；训练全局均值位移基线更好。一次固定候选专家后续验证也未胜过静态、置换或均匀路由。不同实验族预算不同，不作跨族公平排行榜。', '',
            '## 3. 全模型门控不是无损保险', '',
            '| 同一训练 checkpoint | 平均 ED |', '| --- | ---: |',
        ]
        for arm, value in gate['mean_energy_by_arm'].items():
            lines.append(f'| {arm} | {value:.8f} |')
        lines += [
            '', f'full 相对冻结静态 gain = {gate["gain_full_vs_frozen_static"]:+.8f}，误差约高 '
            f'{100*(gate["mean_energy_by_arm"]["full"]/gate["mean_energy_by_arm"]["full_frozen_static"]-1):.1f}%。',
            '门控相对全动态分支缓解了退化，但 corruption 检测没有保证任务收益。仅 1 个训练 seed，不能从这三臂单独归因 router、内源场或噪声。g=0 的工程精确回退保证仍成立。', '',
            '## 4. 积极线索与限制', '',
            f'- 训练侧 50% 分子下采样后的 velocity 方向余弦中位数 {quality["median_condition_seed_thinning_direction_cosine"]:.4f}：估计数值有一定稳定性，不等于生物学方向正确。',
            f'- 有 U 支持的基因 {quality["supported_velocity_genes"]}/2,000；其零截距稳态相图 R² 中位数 {quality["median_gene_zero_intercept_R2_supported"]:.4f}，当前稳态模型假设值得优先检查。',
            f'- 13 个可比 TF 的两条 guide 时间变化方向余弦中位数 {quality["median_guide_time_change_cosine"]:.4f}；不替代编辑逃逸/双细胞审计。',
            '- 训练 mean velocity 与 day4→day5 均值变化余弦约 0.296，但 day5 已进入该估计器拟合，绝不是独立时间锚。',
            '- 原联合训练专家场离散度约占总场 RMS 的 1.6%–2.3%，EM 行聚合模式标签归一化熵约 0.99998–0.99999，提示路由教学信号近均匀。',
            '- 固定候选方案用训练表达 OT 位移去掉条件均值后分两簇，硬分组训练 600 步，随后冻结专家、仅训练路由 600 步。其源侧场相对离散度 40.3%–45.5%，专家确实分开；但候选统计簇不是真实命运，分离也没转化为 velocity 增量。两次离散度审计取样不同，不作精确倍数效应比较。',
            '- 固定候选 follow-up 的设计是看到训练侧专家问题后记录并执行的单次预算；开发集此前已查看，不能冒充一次新的独立确认。源侧 q 敏感性另存 CSV，不把敏感性当有用性。', '',
            '## 5. 修复、环境与未完成项', '',
            '- 修复：门控时间支持从原计划 [2,5] 改为本轮实际 [4,5]，训练器新增一致性拒绝检查。旧 day4→day5 预测数值不变；旧 checkpoint 不得用于主张 day2/day3 已训练。旧配置/权重/结果不覆盖，直接恢复旧宽范围配置会被严格检查拒绝。',
            '- CPU 环境 torch 2.6.0+cpu / NumPy 1.26.4；GPU 私有 overlay 使用 torch 2.7.1+cu126 / NumPy 1.26.4。共享环境未修改；不是原计划 CUDA 12.4 栈认证，继承的非本项目软件存在依赖冲突，不宣称全局 pip check 无冲突。',
            '- 已有测试失败记录保留：9 月 13 日一次随机 toy Sinkhorn 测试失败；随后将测试设为明确的对称场来隔离 K=1 耦合代价，未放宽数值收敛守卫或改写失败产物。',
            '- 未完成：正式 guide/编辑/双细胞审核、正式 G1 效应阈值与预注册、GFG/VeloVI 接入、官方 CellFlow 15 指标、更多全模型 seeds 和严格容量/预算匹配消融、其他时间点/数据集与最终确认。', '',
            '## 6. 下一轮执行决定', '',
            '保留 velocity 主线。下一步是固定版本 VeloVI 的独立环境和训练折适配审计，不追加当前 full 的长训练，也不持续搜索 router 超参数。已有 scvi-tools 0.20.3 不含 VeloVI，单独 velovi 未安装；须在项目私有环境解决，不能升级共享环境。', '',
            '先验证冻结训练估计器对未参与拟合的源细胞可推断、S/U 归一化、基因/barcode 对齐和速度向量投影（不对向量减 PCA 均值），再做一次预算事前记录的替代估计器比较。所有邻居库/kinetic 参数只在训练 TF 拟合；不能拼入开发未来目标或确认细胞。替代仍阴性，则记录当前 ER-short 无增益，进入预定独立时间锚/数据检查，而不是包装成已成功。', '',
            '## 7. 产物索引', '',
            '| 内容 | 完成目录 |', '| --- | --- |',
        ]
        for item in inventory:
            folder = Path(item['directory']).name
            lines.append(f'| {item["experiment"]} | [{folder}](../{folder}/RESULTS.md) |')
        lines += ['', '操作命令和架构边界见 [全量手册](../../docs/veloroute_full_runbook_20260914.md)。'
                  '本目录同时保存全部指标、比较区间、实验清单、输入/源码/环境哈希和源结果汇总；没有打开最终确认数据。', '']
        (run.directory/'RESULTS.md').write_text('\n'.join(lines))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    print(json.dumps(compile_report(parser.parse_args().output), ensure_ascii=False))
