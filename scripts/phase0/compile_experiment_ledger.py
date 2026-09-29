"""Hash-audit existing evidence and index the ledger; never train or open targets."""
import argparse
from collections import Counter
import csv
from decimal import Decimal
import gzip
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from veloroute.artifacts import Run, save_csv, save_json, sha256

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/veloroute_experiment_ledger_20260914.md'
EVIDENCE = {
    'legacy': 'outputs/veloroute_research_progress_20260914',
    'parity': 'outputs/veloroute_gfg_reference_parity_20260914',
    'full_smoke': 'outputs/veloroute_gfg_full_smoke_20260914',
    'resume': 'outputs/veloroute_gfg_actual_resume_verified_20260914',
    'synthetic_original': 'outputs/veloroute_gfg_synthetic_20260914',
    'synthetic_input_U': 'outputs/veloroute_gfg_synthetic_inputU_recovered_20260914',
    'multimodality': 'outputs/veloroute_gfg_multimodal_screen_20260914',
    'source_audit': 'outputs/veloroute_gfg_source_audit_20260914_seed0',
    'pilot': 'outputs/veloroute_gfg_pilot_evaluation_20260914',
    'direct': 'outputs/veloroute_gfg_direct_router_evaluation_20260914',
    'day3_raw': 'outputs/veloroute_raw_day3_20260914',
    'time_inventory': 'outputs/veloroute_time_anchor_inventory_20260914',
}


def read_json(path):
    return json.loads(Path(path).read_text())


def complete_evidence(directory, filename):
    provenance = read_json(directory/'provenance.json')
    if provenance['status'] != 'complete':
        raise ValueError(f'Unfinished evidence: {directory}')
    if provenance['outputs'].get(filename) != sha256(directory/filename):
        raise ValueError(f'Evidence hash mismatch: {directory/filename}')
    return provenance


def verify_numeric_tables(document, evidence):
    """Validate displayed rounding against the original JSON, not pasted values."""
    checks = []

    def check(scope, label, displayed, expected):
        value = Decimal(displayed)
        tolerance = 0.51*float(Decimal(10)**value.as_tuple().exponent)+1e-15
        if displayed == '0':
            tolerance = 0.
        if abs(float(value)-expected) > tolerance:
            raise ValueError(f'Ledger numeric mismatch: {scope}/{label}: {displayed} vs {expected}')
        checks.append(dict(scope=scope, metric=label, displayed=displayed, expected=expected,
                           absolute_difference=abs(float(value)-expected), passed=True))

    arm_map = {'static': 'static', 'frozen GFG': 'gfg_frozen', 'joint GFG': 'gfg_joint',
               'joint U 置换': 'gfg_joint_shuffled', 'U 置换': 'gfg_joint_shuffled', 'frozen': 'gfg_frozen'}
    metric_map = {'energy': 'energy_distance', 'exact energy': 'exact_mixture_energy',
                  'log 方差误差': 'log_variance_error', '中心化均值 MSE': 'condition_centered_gene_mean_mse'}
    for scope, first, end, main_metric in [('pilot', 9, 10, 'energy_distance'),
                                         ('direct', 10, 11, 'exact_mixture_energy')]:
        section = document.split(f'\n## {first}.', 1)[1].split(f'\n## {end}.', 1)[0]
        compared, means = 0, 0
        for line in section.splitlines():
            if not line.startswith('|'):
                continue
            cells = [s.strip() for s in line.strip('|').split('|')]
            if len(cells) != 4 or cells[0] not in arm_map:
                continue
            arm = arm_map[cells[0]]
            if cells[1] in metric_map:
                metric = metric_map[cells[1]]
                row = next(r for r in evidence[scope]['comparisons']
                           if r['control_arm'] == arm and r['metric'] == metric)
                bounds = [x.strip() for x in cells[3].strip('[]').split(',')]
                for key, display in zip(('gain', 'ci_low', 'ci_high'), (cells[2], *bounds)):
                    check(scope, f'{arm}/{metric}/{key}', display, row[key])
                compared += 1
            else:
                for metric, display in zip((main_metric, 'log_variance_error', 'condition_centered_gene_mean_mse'), cells[1:]):
                    check(scope, f'{arm}/{metric}/mean', display, evidence[scope]['means'][arm][metric])
                means += 1
        if compared != len(evidence[scope]['comparisons']) or means != len(evidence[scope]['means']):
            raise ValueError(f'Missing numeric table entries for {scope}')
    synth_rows = {'same-S，隐藏组分变化': 'same_S_velocity_branch',
                  'S×U 对应关系决定组分': 'state_velocity_interaction',
                  '静态位置决定组分': 'position_branch', '单峰、无增量': 'no_increment'}
    for line in document.splitlines():
        if not line.startswith('|'):
            continue
        cells = [s.strip() for s in line.strip('|').split('|')]
        if len(cells) == 5 and cells[0] in synth_rows:
            system = synth_rows[cells[0]]
            for arm, display in zip(('static', 'gfg_frozen', 'gfg_joint', 'gfg_joint_shuffled'), cells[1:]):
                check('synthetic', f'{system}/{arm}/MAE', display,
                      evidence['synthetic_input_U']['mean_ratio_error'][system][arm])
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError('Use a new ledger output directory; preserve older records')
    document = DOC.read_text()
    inputs = [DOC, Path(__file__).resolve()]
    evidence, records = {}, []
    for name, relative in EVIDENCE.items():
        directory = ROOT/relative
        provenance = complete_evidence(directory, 'summary.json')
        evidence[name] = read_json(directory/'summary.json')
        inputs.extend([directory/'summary.json', directory/'provenance.json', directory/'config.yaml'])
        records.append(dict(evidence=name, directory=str(directory), status=provenance['status'],
                            started_utc=provenance['started_utc'], finished_utc=provenance['finished_utc'],
                            summary_sha256=sha256(directory/'summary.json')))
    tests = ROOT/'outputs/veloroute_gfg_all_tests_20260914_v5/junit.xml'
    suite = ET.parse(tests).getroot().find('testsuite').attrib
    if int(suite['tests']) != 123 or any(int(suite[k]) for k in ('errors', 'failures', 'skipped')):
        raise ValueError('The 123-passed claim does not match the referenced JUnit')
    inputs.append(tests)
    checks = verify_numeric_tables(document, evidence)
    indices = []
    for family, pattern, expected in [('pilot', 'outputs/veloroute_gfg_pilot_20260914/*/selection.json', 12),
                                     ('direct', 'outputs/veloroute_gfg_direct_router_20260914/seed*/*/selection.json', 9)]:
        files = sorted(ROOT.glob(pattern))
        if len(files) != expected:
            raise ValueError(f'Incomplete {family} selection grid')
        seen = set()
        for selection_path in files:
            selected = read_json(selection_path)
            key = (selected['arm'], selected['seed'])
            if key in seen:
                raise ValueError('Duplicate arm/seed')
            seen.add(key)
            training, prediction = ROOT/selected['training'], ROOT/selected['prediction']
            provenance = complete_evidence(training, 'model.pt')
            complete_evidence(prediction, 'predictions.npz')
            complete_evidence(prediction, 'prediction_manifest.json')
            if sha256(training/'model.pt') != selected['model_sha256'] or sha256(prediction/'predictions.npz') != selected['prediction_sha256']:
                raise ValueError('Selection changed since frozen evaluation')
            pm = read_json(prediction/'prediction_manifest.json')
            if pm['future_target_read'] is not False or pm['role'] != 'validation' or (pm['velocity_arm'], pm['seed']) != key:
                raise ValueError('Source-only validation prediction contract failed')
            complete_evidence(training, 'summary.json')
            inputs.extend([selection_path, training/'summary.json', training/'provenance.json',
                           prediction/'prediction_manifest.json', prediction/'provenance.json'])
            indices.append(dict(family=family, arm=key[0], seed=key[1], training=str(training), prediction=str(prediction),
                                model_sha256=selected['model_sha256'], prediction_sha256=selected['prediction_sha256'],
                                started_utc=provenance['started_utc'], finished_utc=provenance['finished_utc'],
                                recorded_cuda_visible_devices=provenance.get('cuda_visible_devices'),
                                hash_audit_pass=True, future_target_read=False))
    # Compare with the evaluation-time frozen grids, not just current selectors.
    for family in ('pilot', 'direct'):
        directory = ROOT/EVIDENCE[family]
        complete_evidence(directory, 'frozen_grid.json')
        frozen = {(r['arm'], r['seed']): r['sha256'] for r in read_json(directory/'frozen_grid.json')}
        current = {(r['arm'], r['seed']): r['prediction_sha256'] for r in indices if r['family'] == family}
        if current != frozen:
            raise ValueError('Selection differs from the evaluation-time grid')
        inputs.append(directory/'frozen_grid.json')
    kang_path = Path('/data/dataset/perturbation_v1/kang_2018/GSE96583_batch2.total.tsne.df.tsv.gz')
    with gzip.open(kang_path, 'rt') as stream:
        reader = csv.reader(stream, delimiter='\t')
        next(reader)
        metadata = list(reader)
    labels = Counter(r[7] for r in metadata)
    singlets = [r for r in metadata if r[7] == 'singlet']
    kang = dict(metadata_rows=len(metadata), status_counts=dict(labels), singlet_donors=len({r[3] for r in singlets}),
                singlet_celltypes=dict(Counter(r[6] for r in singlets)), response_values_read=False, final_QC=False)
    if len(metadata) != 29065 or len(singlets) != 24679 or kang['singlet_donors'] != 8:
        raise ValueError('Kang metadata count differs from the written record')
    inputs.append(kang_path)
    links = []
    for label, target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)', document):
        if target.startswith(('https://', 'http://', '#')):
            continue
        path = (DOC.parent/target).resolve()
        # The self-link is published by this run (or points to the canonical record).
        expected_self = ROOT/'outputs/veloroute_experiment_ledger_20260914/RESULTS.md'
        exists = path.exists() or path == expected_self
        if not exists:
            raise ValueError(f'Broken local evidence link: {label}: {path}')
        links.append(dict(label=label, path=str(path), exists_or_pending_self=exists))
    config = dict(record_kind='retrospective_documentation_not_new_preregistration',
                  no_training=True, confirmation='SEALED', seed=None,
                  current_gpu_policy=dict(allowed=[0, 1, 2, 3], reserved=[4, 5, 6, 7]),
                  full_training_decision='DEFERRED_REAL_INCREMENT_NOT_ESTABLISHED')
    with Run(output, stage='experiment_ledger_audit', kind='engineering', config=config,
             inputs=list(dict.fromkeys(inputs))) as run:
        save_csv(run.directory/'evidence_index.csv', records)
        save_csv(run.directory/'run_index.csv', indices)
        save_csv(run.directory/'document_numeric_checks.csv', checks)
        save_csv(run.directory/'document_local_links.csv', links)
        save_json(run.directory/'kang_metadata_summary.json', kang)
        metrics, comparisons = [], []
        for scope in ('pilot', 'direct'):
            for arm, values in evidence[scope]['means'].items():
                metrics.extend(dict(scope=scope, arm=arm, metric=metric, value=value) for metric, value in values.items())
            comparisons.extend(dict(scope=scope, metric=r['metric'], control_arm=r['control_arm'],
                                    gain=r['gain'], ci_low=r['ci_low'], ci_high=r['ci_high'],
                                    confidence=r['confidence'], n_conditions=r['n_conditions'], n_seeds=r['n_seeds'])
                               for r in evidence[scope]['comparisons'])
        for system, arms in evidence['synthetic_input_U']['mean_ratio_error'].items():
            metrics.extend(dict(scope='synthetic/'+system, arm=arm, metric='ratio_MAE', value=value) for arm, value in arms.items())
        save_csv(run.directory/'metrics.csv', metrics)
        save_csv(run.directory/'comparisons.csv', comparisons)
        save_json(run.directory/'summary.json', dict(status='EXPERIMENT_LEDGER_AUDITED',
            real_runs_verified=len(indices), completed_evidence_groups=len(records), numerical_checks=len(checks),
            local_links_checked=len(links), referenced_tests_passed=123, new_training_runs=0,
            synthetic_mechanism='SUPPORTED_IN_SPECIFIED_TOY', real_velocity_increment='NOT_ESTABLISHED',
            formal_G1='NOT_RUN', confirmation='SEALED', day3_raw_complete=True, day3_su_complete=False))
        (run.directory/'document_snapshot.md').write_text(document)
        (run.directory/'RESULTS.md').write_text(
            '# 实验总账审计\n\n'
            '详细记录见 [实验总账](../../docs/veloroute_experiment_ledger_20260914.md)。本包只核验与汇编旧产物，没有新训练。\n\n'
            f'- 完成证据组：{len(records)}；真实训练/预测哈希及评价冻结网格核验：{len(indices)} 次。\n'
            f'- 文档数值逐项核验：{len(checks)} 项；本地证据链接：{len(links)} 个。\n'
            '- 原始回归记录：123 passed；本次不把文档审计计入模型测试数。\n'
            '- 合成机制阳性与真实增量未成立分别记录；正式 G1 未运行，确认 TF 封存。\n'
            '- day3 raw 完成，S/U 未完成；Kang 只统计元数据，其他候选只记录已有文献核查。\n\n'
            '文件：metrics.csv / comparisons.csv 为原精度结果；run_index.csv 为权威恢复路径和哈希；'
            'evidence_index.csv 为完成状态与时间戳；document_numeric_checks.csv 为显示舍入核验；'
            'document_snapshot.md 为当次文档文本快照（相对链接按 docs 原位置解释）；provenance.json 记录输入、环境和哈希。\n')
    print(json.dumps(dict(status='EXPERIMENT_LEDGER_AUDITED', output=str(output), verified_runs=len(indices), numeric_checks=len(checks))))


if __name__ == '__main__':
    main()
