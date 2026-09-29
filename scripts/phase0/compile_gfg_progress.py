"""Compile measured GFG evidence; distinguish engineering, toy, and real effects."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from veloroute.artifacts import Run, save_csv, save_json

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--pilot-evaluation')
parser.add_argument('--direct-evaluation')
parser.add_argument('--time-anchor-inventory')
parser.add_argument('--tests', default='outputs/veloroute_gfg_all_tests_20260914_v4/junit.xml')
parser.add_argument('--full-training')
args = parser.parse_args()
paths = {
    'parity': Path('outputs/veloroute_gfg_reference_parity_20260914/summary.json'),
    'full_smoke': Path('outputs/veloroute_gfg_full_smoke_20260914/summary.json'),
    'resume': Path('outputs/veloroute_gfg_actual_resume_verified_20260914/summary.json'),
    'multimodality': Path('outputs/veloroute_gfg_multimodal_screen_20260914/summary.json'),
    'synthetic_original': Path('outputs/veloroute_gfg_synthetic_20260914/summary.json'),
    'synthetic_inputU': Path('outputs/veloroute_gfg_synthetic_inputU_recovered_20260914/summary.json'),
    'source_audit': Path('outputs/veloroute_gfg_source_audit_20260914_seed0/summary.json'),
}
if args.pilot_evaluation:
    paths['pilot'] = Path(args.pilot_evaluation)/'summary.json'
if args.direct_evaluation:
    paths['direct'] = Path(args.direct_evaluation)/'summary.json'
if args.time_anchor_inventory:
    paths['time_anchors'] = Path(args.time_anchor_inventory)/'summary.json'
for path in paths.values():
    if json.loads((path.parent/'provenance.json').read_text())['status'] != 'complete':
        raise ValueError(f'Cannot report unfinished evidence as complete: {path}')
test_path = Path(args.tests)
evidence = {name: json.loads(path.read_text()) for name, path in paths.items()}
tests = ET.parse(test_path).getroot().find('testsuite').attrib
if any(int(tests[key]) for key in ('errors', 'failures', 'skipped')):
    raise ValueError('Do not claim a fully passed test suite with errors/failures/skips')
with Run(args.output, stage='GFG_joint_progress_compilation', kind='development',
         config=dict(confirmation='SEALED', prior_negative_results_preserved=True),
         inputs=[*paths.values(), test_path]) as run:
    synth = evidence['synthetic_inputU']['mean_ratio_error']
    rows = [dict(scope='synthetic', metric=f'{system}/{arm}/ratio_MAE', value=value)
            for system, arms in synth.items() for arm, value in arms.items()]
    rows += [dict(scope='training_source', metric=key, value=value)
             for key, value in evidence['source_audit']['metrics'].items()]
    rows += [dict(scope='engineering', metric='passed_tests', value=int(tests['tests']))]
    full_status = 'NOT_STARTED'
    if args.full_training:
        directory = Path(args.full_training)
        full_status = json.loads((directory/'status.json').read_text())
    pilot = evidence.get('pilot')
    direct = evidence.get('direct')
    if pilot and direct and not (pilot['all_primary_intervals_positive'] and direct['all_primary_intervals_positive']):
        if not args.full_training:
            full_status = 'DEFERRED_REAL_INCREMENT_NOT_ESTABLISHED'
    if direct:
        rows += [dict(scope='direct_router_development', metric=f'{arm}/{metric}', value=value)
                 for arm, metrics in direct['means'].items() for metric, value in metrics.items()]
    if pilot:
        rows += [dict(scope='real_development', metric=f'{arm}/{metric}', value=value)
                 for arm, metrics in pilot['means'].items() for metric, value in metrics.items()]
    summary = dict(evidence=evidence, tests=tests, full_training=full_status,
        gpu_policy=dict(allowed_physical=[0, 1, 2, 3], reserved_for_others=[4, 5, 6, 7]),
        real_velocity_advantage=('SUPPORTED_IN_EXPLORATORY_PILOT' if pilot and pilot['all_primary_intervals_positive']
                                 else 'NOT_ESTABLISHED'), formal_G1='NOT_RUN', confirmation='SEALED')
    lines = ['# GFG 联训研究进度', '',
        '实际 GFG 双流/JVP 已与 router 联合训练；合成优势成立，真实增量须单独判断。', '',
        '## 已验证', '',
        '- 原实现数值对齐通过；不是旧 MLP 改名，也不是 VeloVI。',
        '- 全宽 20,021,704 参数、8 专家、老师/EM/参考场/内源/门控/噪声短验收通过。',
        f"- {tests['tests']} 项回归通过；真实中断恢复的 207 个张量与不中断版本完全相同。",
        '- 初始化是本地 MouseBrain seed0，不宣称官方 foundation release；rho 是 RNA 重构误差代理。', '',
        '## 优势场景（合成，不是真实生物学结果）', '',
        '| 系统 | 静态 | 冻结 GFG | 联合 GFG | 前编码 U 置换 |',
        '|---|---:|---:|---:|---:|']
    for system, arms in synth.items():
        lines.append('| '+system+' | '+' | '.join(f'{arms[a]:.4f}' for a in ('static', 'gfg_frozen', 'gfg_joint', 'gfg_joint_shuffled'))+' |')
    lines += ['', '数字为三 seed 终态组分比例 MAE，越低越好。训练只看未配对群体分布，逐细胞真值命运仅评价。',
        'S×U 对应关系决定比例时，联合 GFG 胜过静态及前编码 U 置换；尚不足以证明联训本身优于冻结 GFG。',
        '这是隐藏状态群体组成 toy，不是生物学标定的 RNA 反应仿真；未比较直接 U 编码器，不证明 GFG 特异的机制优势。单峰 K=1 是预先给定，不证明自动关门或恢复 K。',
        '同 S 内部的后编码速度置换保留群体多重集合，不能要求边际退化。原后编码置换未通过的记录完整保留。',
        '合成 U 置换实验被中断后复用 40 个 checkpoint，补训 8 个单峰对照；恢复产物中的训练 trace 仅涵盖补训部分。', '',
        '## 真实数据与风险', '',
        '- 当前只覆盖同 KO 的 RENGE day4→day5，不是 de novo。',
        '- 14 个训练 TF 中 0 个通过本轮稳定多峰筛查；这是有限功效候选筛查，不是正式 D4，也不能断言所有条件单峰。',
        '- GFG 对条件内 U 置换的速度相对 RMS 改变 0.3155，但 router 概率 MAE 仅 0.000237；教学信号可能仍然太弱。',
        '- GFG 速度范数从初始化约 30.39 降至 2.29，方向中位余弦约 0.938；低原生损失不等于方向或物理时间正确。',
        '- 以前所有稳态残差 velocity 阴性实验保留，不能把它们称作 GFG 阴性。', '']
    if pilot:
        lines += ['## 固定四臂三 seed 开发评价', '',
            '全部 12 份源侧预测冻结后统一评价；四个开发 TF 已反复查看，不能视为独立确认。', '',
            '| 比较：联合 GFG 相对 | 指标 | 增益 | 校正后区间 |', '|---|---|---:|---|']
        for row in pilot['comparisons']:
            lines.append(f"| {row['control_arm']} | {row['metric']} | {row['gain']:.7g} | [{row['ci_low']:.7g}, {row['ci_high']:.7g}] |")
        lines += ['', '增益 = 对照误差 − 联合 GFG 误差。区间先平均 seed，再按条件 bootstrap，并对 9 项比较作 Bonferroni 校正。',
            'frozen 臂在 C 阶段不再更新 GFG 原生损失，与 joint 的预训练更新预算差异单列，不冒充严格的联训因果增益。', '']
    else:
        lines += ['真实固定四臂尚未统一评价；不能从合成结果推断 RENGE 已有增量。', '']
    if direct:
        lines += ['## 唯一一轮直接 router 监督修复', '',
            '同 seed 静态 checkpoint 提供所有臂完全一致的冻结专家候选；仅 router/GFG 更新，训练只用未配对训练终态。',
            '3 臂 × 3 seed × 600 步已完成。精确混合分布指标排除模式抽样噪声；六项主比较全部区间跨零。', '',
            '| 比较：联合 GFG 相对 | 指标 | 增益 | 校正后区间 |', '|---|---|---:|---|']
        for row in direct['comparisons']:
            lines.append(f"| {row['control_arm']} | {row['metric']} | {row['gain']:.7g} | [{row['ci_low']:.7g}, {row['ci_high']:.7g}] |")
        lines += ['', '停止在反复查看的 ER-short 开发 TF 上追加损失/seed/训练时长来追阳性。固定结果不改写。', '']
    if 'time_anchors' in evidence:
        lines += ['## 下一步：新增时间锚数据', '',
            'day2/day3 发布矩阵存在，但 S/U 尚未落地；新原始数据转到专属数据盘，先处理 day3。',
            '只读资产清单与原始获取不打开确认集；时间点不是独立生物学复现。正式比较另行冻结。', '']
    lines += ['## 当前全量状态', '', json.dumps(full_status, ensure_ascii=False), '',
        '工程全组件 PASS + 合成组分检验 PASS，并不能抵消两轮真实增量未成立。暂不启动 A300/B2000/C12000 长时全量。',
        '资源硬限制：本项目仅 GPU 0–3；GPU 4–7 不使用。其他项目已有进程不由本项目停止。', '',
        '正式 G1 未运行，最终 5 个确认 TF 封存。任何全量工程运行都不能自动升级为真实速度增益或多峰优势。']
    save_csv(run.directory/'metrics.csv', rows)
    save_json(run.directory/'summary.json', summary)
    (run.directory/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(status='GFG_PROGRESS_COMPILED', output=str(run.directory))))
