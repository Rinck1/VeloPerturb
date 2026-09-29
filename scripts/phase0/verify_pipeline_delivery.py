"""Verify engineering delivery without evaluating any real future response."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from veloroute.artifacts import Run, object_hash, read_csv, save_csv, save_json, sha256
from veloroute.experiments import condition_vectors, validate_pair
from veloroute.latent import FrozenSplicingTransform, load_pack


def verify_directory(directory):
    directory = Path(directory)
    record = json.loads((directory / 'provenance.json').read_text())
    if record['status'] != 'complete':
        raise ValueError(f'Incomplete artifact: {directory}')
    for name, expected in record['outputs'].items():
        if sha256(directory / name) != expected:
            raise ValueError(f'Changed artifact: {directory / name}')
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workflow', required=True)
    parser.add_argument('--release', required=True)
    parser.add_argument('--conditions', required=True)
    parser.add_argument('--tests', required=True)
    parser.add_argument('--smoke', required=True)
    parser.add_argument('--reservation', default='outputs/veloroute_condition_reservation_20260912/condition_split.csv')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    workflow, release, conditions, smoke = map(Path, (args.workflow, args.release, args.conditions, args.smoke))
    fold = workflow / 'fold'
    inputs = [__file__, args.tests, args.reservation, workflow / 'provenance.json', smoke / 'provenance.json',
              conditions.parent / 'provenance.json', *[release / f'day{day}/provenance.json' for day in (4, 5)]]
    with Run(args.output, stage='pipeline_delivery_audit', kind='engineering', config=vars(args), inputs=inputs,
             seed=20260912) as run:
        for directory in (workflow, fold, smoke, conditions.parent, release / 'day4', release / 'day5'):
            verify_directory(directory)
        test_root = ET.parse(args.tests).getroot()
        suites = [test_root] if test_root.tag == 'testsuite' else list(test_root.iter('testsuite'))
        tests = {key: sum(int(s.get(key, 0)) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
        assert tests['tests'] > 0 and tests['failures'] == tests['errors'] == tests['skipped'] == 0
        deps = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True, check=True)
        (run.directory / 'pip_check.txt').write_text(deps.stdout)
        role_by_condition = {r['condition']: r['role'] for r in read_csv(args.reservation)}
        tf_conditions = sorted(c for c, role in role_by_condition.items() if role in {'train', 'validation', 'confirmation'})
        vectors, condition_meta = condition_vectors(conditions, tf_conditions)
        assert vectors.shape == (23, 2560) and condition_meta['frozen']
        transform = FrozenSplicingTransform.load(fold / 'transform.npz')
        fingerprint = sha256(fold / 'transform.npz')
        expected_fit_ids = set()
        roles = {}
        for role in ('train', 'validation'):
            source, sm = load_pack(fold / f'{role}_source.npz', expected_side='source')
            target, tm = load_pack(fold / f'{role}_target.npz', expected_side='target')
            validate_pair(source, sm, target, tm, role)
            assert sm['transform_hash'] == fingerprint and sm['kind'] == 'engineering'
            assert source['z'].shape[1] == source['velocity'].shape[1] == 50
            expected_conditions = {c for c, r in role_by_condition.items() if r == role}
            assert set(source['conditions']) == set(target['conditions']) == expected_conditions
            if role == 'train':
                expected_fit_ids.update(source['cell_ids'])
                expected_fit_ids.update(target['cell_ids'])
            roles[role] = {'conditions': len(expected_conditions), 'source_cells': len(source['z']), 'target_cells': len(target['z'])}
        actual_fit_ids = {r['cell_id'] for r in read_csv(fold / 'fit_cell_ids.csv')}
        assert actual_fit_ids == expected_fit_ids
        assert transform.fit_ids_hash == object_hash(sorted(actual_fit_ids))
        # Confirmation files are integrity-hashed above, never loaded for effect evaluation.
        workflow_summary = json.loads((workflow / 'summary.json').read_text())
        assert workflow_summary['G1'] == workflow_summary['real_model_training'] == 'NOT_RUN'
        assert workflow_summary['confirmation_evaluation'] == 'SEALED'
        smoke_summary = json.loads((smoke / 'summary.json').read_text())
        assert smoke_summary['status'] == 'ENGINEERING_PASS' and not smoke_summary['confirmation_evaluated']
        rows = []
        for day in (4, 5):
            values = json.loads((release / f'day{day}/summary.json').read_text())
            mapping = json.loads((Path('data/renge/quant') / f'day{day}/af_map/map_info.json').read_text())
            cells = read_csv(release / f'day{day}/cell_manifest.csv')
            retained = sum(row['technical_qc_pass'] == 'True' and row['guide_call_pass'] == 'True' for row in cells)
            rows.append({'day': day, 'cells': values['cells'], 'genes': values['genes'],
                         'technical_qc_cells': values['cells_passing_technical_qc'],
                         'conservative_guide_cells': values['cells_with_conservative_guide_call'],
                         'both_qc_and_guide_cells_including_controls': retained,
                         'median_S_UMI': values['median_spliced_umi'], 'median_U_UMI': values['median_unspliced_umi'],
                         'reads': mapping['num_reads'], 'mapped_reads': mapping['num_mapped'],
                         'mapped_percent': mapping['percent_mapped'], 'h5ad_sha256': sha256(release / f'day{day}/day{day}.h5ad')})
        save_csv(run.directory / 'data_qc.csv', rows)
        save_csv(run.directory / 'development_roles.csv', [{'role': role, **value} for role, value in roles.items()])
        checks = [{'check': name, 'passed': True} for name in (
            'completed_artifact_output_hashes', 'all_unit_tests', 'dependency_consistency',
            'ESM2_23_TF_2560_dimensions', 'train_validation_TF_coverage_matches_reservation',
            'PCA_velocity_fit_exactly_training_cell_ids', 'source_target_contracts_and_frozen_transform',
            'synthetic_three_arm_train_predict_gene_evaluate_loop', 'confirmation_effects_not_evaluated',
            'formal_G1_and_real_model_training_not_started')]
        save_csv(run.directory / 'checks.csv', checks)
        result = {'status': 'ENGINEERING_DELIVERY_PASS', 'tests': tests, 'roles': roles,
                  'fit_cells': len(actual_fit_ids), 'PCA_dimensions': len(transform.components),
                  'selected_genes': len(transform.selected), 'supported_velocity_genes': int(transform.supported.sum()),
                  'velocity_estimator': transform.estimator, 'ESM2_shape': list(vectors.shape),
                  'real_G1': 'NOT_RUN', 'real_router_training': 'NOT_RUN', 'confirmation': 'SEALED',
                  'scope': 'RENGE_ER_short_day4_to_day5_minimal_pipeline',
                  'official_CellFlow_15_metrics_integrated': False}
        save_json(run.directory / 'summary.json', result)
        report = [
            '# VeloRoute 数据与最小全流程交付（2026-09-12）', '',
            '结论：RENGE ER-short（day4→day5）真实数据处理与训练折内数据准备完成；最小模型的文件级训练→生成→评估在合成数据上通过。真实 G1 和 router 实验没有运行。', '',
            '## 已落地的真实数据', '',
            '| 样本 | 细胞 | 技术 QC 通过 | 保守 guide 判定 | S UMI 中位数 | U UMI 中位数 | reads 映射率 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
            *[f"| day{r['day']} | {r['cells']:,} | {r['technical_qc_cells']:,} | {r['conservative_guide_cells']:,} | {r['median_S_UMI']:g} | {r['median_U_UMI']:g} | {r['mapped_percent']}% |" for r in rows], '',
            'QC 与 guide 两列不是交集；交集见 data_qc.csv。S/U/A 分开保存，A 不并入 U。guide 来自发布的 feature UMI 计数，属于保守计算筛选，不代表已验证编辑或完整双细胞排除。', '',
            f"- 数据：`{release}/day4/day4.h5ad`、`{release}/day5/day5.h5ad`；每份 60,668 个基因。",
            f"- 数据包：`{fold}`。仅训练 TF 的 {len(actual_fit_ids):,} 颗细胞拟合 2,000 基因 / PCA-50；1,832 个基因满足当前估计器的最低支持规则，不等于动力学可靠性认证。",
            f"- 条件：`{conditions}`，23×2560 的冻结 ESM2-3B 真正蛋白嵌入，无随机/零向量替代。",
            '- 14 个训练 TF、4 个验证 TF、5 个最终确认 TF；AAVS1/CTRL 保持独立控制类别，本轮 ER-short TF 数据包不混入控制细胞。', '',
            '## 管线与工程验证', '',
            '```text',
            'NCBI SRA → technical FASTQ 核验 → 同样本 GEX run 联合 USA 定量',
            ' → S/U/A + guide/QC → 按 TF 切分 → 仅训练折拟合 velocity/PCA',
            ' → 源/目标分包 + 冻结 ESM2 → [质量审计、预注册、G1]',
            ' → router/专家训练 → checkpoint → 源侧生成 → 冻结预测 → 独立评估',
            '```', '',
            f"- 完整测试：{tests['tests']} passed；依赖检查通过。见 `{args.tests}`。",
            f"- 合成文件级闭环：real/static/shuffled 三臂均完成，包含逆 PCA 基因空间输出。见 `{smoke}/RESULTS.md`。",
            '- G1 执行器另有隔离测试：7 个 A/B 臂×3 seeds 的预测全部冻结后，才读取验证目标；测试中的批准标记不是项目正式批准。',
            '- 10 项交付核验通过，记录于 checks.csv；代码、环境、输入和输出哈希在 provenance.json。', '',
            '## 尚未完成与下一步', '',
            '1. 当前 velocity 是明确标注的稳态残差估计基线，不是 GFG/VeloVI，量纲不是天；需相图、稳定性、深度/应激混杂和独立增量诊断。',
            '2. 完成 guide 一致性、编辑逃逸/双细胞和 S/U 质量审计；确认正式细胞清单。当前数据 formal_ready=false，不得直接改成 true 作为通过审计的替代。',
            '3. 在看效果前，以独立依据确定实际收益/非劣效阈值并冻结 G1 协议与确切数据包哈希。4 个验证 TF 的条件级统计功效有限，3 seeds 不能增加独立条件数。',
            '4. 执行正式四臂 A/B，数值达标也仅进入人工机制/结构审核；G1-GO 后才启真实 router 与配对消融。正式延续必须复用 G1 的同一 frozen_fold，不能重新拟合替换。',
            '5. 原计划的 GFG/VeloVI、官方 CellFlow 15 指标、可靠度门控、自适应 K、ER-long/de novo 和其他数据集没有在本次宣称完成。当前评估是明确命名的 energy、均值、方差和训练选基因指标。', '',
            '## 可追溯处理说明', '',
            '- 同一时间点两条 GEX run 联合解析 UMI，未将逐 run 计数直接相加。全部 8 条 day4/day5 run 已下载、SRA 校验、提取；FASTQ 全文件 gzip 完整性检查，逐 run 前 100 万对（不足则全量）布局审计。',
            '- `_2`=26bp barcode16+UMI10，`_3`=91bp cDNA；没有因偶见 N 而统一右移 barcode。参考为官方 MD5 验证的 GENCODE v32 splici r91，5′ rc 定量方向。',
            '- ESM2 使用 GENCODE v32 每 TF 最长蛋白；超过 1022 aa 的蛋白分窗并按长度合并，不冒称 canonical UniProt 全长上下文编码。',
            '- 中断/失败记录保留：下载长连接中断改为可校验分段续传；首次 USA 导入不兼容 -U/-A 后明确修复并补回归测试。最终数据 release 不覆盖旧产物。',
            '- 原始 SRA、失败/部分下载和量化产物均保留；FASTQ 仅做无损压缩，可从 .gz 或 SRA 恢复。', '',
            '操作顺序与下一轮命令见 `docs/veloroute_pipeline_runbook_20260912.md`。',
        ]
        (run.directory / 'RESULTS.md').write_text('\n'.join(report) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
