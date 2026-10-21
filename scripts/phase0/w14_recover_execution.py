"""Resume the prepared RENGE grid after verified W14 computation, not Git success.

Frozen W14/B3 sources, failed controller records and original scores are retained.
Only a new orchestration/recovery run is created. No model or objective changes.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from w4_artifacts import Artifacts, sha256
from w7_router_scaling import atomic_json, write_csv
from w14_loro import freeze_grid, verify_lock
from w14b_renge_router import verify as verify_renge
from w14_publication_recovery import publish_records


ROOT = Path(__file__).resolve().parents[2]


def scientific_completion(evaluation):
    evaluation = Path(evaluation)
    state = json.loads((evaluation / 'status.json').read_text())
    result = json.loads((evaluation / 'summary.json').read_text())
    if state['status'] != 'complete' or result['status'] != 'W14_GRID_EVALUATED':
        raise ValueError('Complete scientific evaluation required, independently of publication')
    for name in ('metrics.csv', 'comparisons.csv', 'frozen_grid.json', 'RESULTS.md'):
        if not (evaluation / name).is_file():
            raise ValueError(f'Missing completed result: {name}')
    return result


def finish_supplementary(primary, recovery):
    old = ROOT / 'outputs/w14_supplementary_v2'
    out = recovery.directory / 'supplementary'
    out.mkdir()
    result = json.loads((old / 'summary.json').read_text())
    for name in ('per_seed_metrics.csv', 'descriptive_cell_intervals.csv'):
        shutil.copyfile(old / name, out / name)
    rows = list(csv.DictReader((primary / 'metrics.csv').open()))
    retention = []
    for rep, day in sorted({(r['heldout'], r['source_day']) for r in rows}):
        by = {r['arm']: r for r in rows if (r['heldout'], r['source_day']) == (rep, day)}
        static = float(by['static']['energy'])
        real = 1 - float(by['real']['energy']) / static
        cohort = 1 - float(by['real']['cohort_average_energy']) / static
        retention.append(dict(heldout=rep, source_day=int(day), real_gain=real, cohort_gain=cohort,
            retained_gain_fraction=cohort / real if real > 1e-12 else None,
            interpretation='frozen_recentered_q_intervention' if real > 1e-12 else
                'not_applicable_when_original_gain_nonpositive'))
    write_csv(out / 'routing_retention.csv', retention)
    real = np.mean([r['real_gain'] for r in retention])
    result.update(overall_retained_gain_fraction=float(np.mean([r['cohort_gain'] for r in retention]) / real)
        if real > 1e-12 else None, primary_metrics_sha256=sha256(primary / 'metrics.csv'),
        recovery='Finish saved descriptive calculations without refitting or rescoring predictions',
        original_summary_sha256=sha256(old / 'summary.json'))
    atomic_json(out / 'summary.json', result)
    report = '# W14 supplementary scores and routing retention\n\n' + json.dumps(result, indent=2) + '\n'
    (out / 'RESULTS.md').write_text(report)
    doc = ROOT / 'docs/current/w14_supplementary_results.md'
    if doc.exists():
        raise ValueError('Existing supplementary document retained')
    doc.write_text(report)
    return [doc] + [out / n for n in ('summary.json', 'RESULTS.md', 'per_seed_metrics.csv',
        'descriptive_cell_intervals.csv', 'routing_retention.csv')]


def try_publish(run, name, files, message):
    try:
        result = publish_records(ROOT, files, message)
    except Exception as error:
        result = dict(status='publication_pending', error=repr(error),
            computation_unchanged=True, blocks_downstream=False)
    run.json(name, result)
    return result


def launch(gpu, directory, arguments):
    subprocess.run([sys.executable, str(ROOT / 'scripts/phase0/launch_gpu_job.py'), '--gpu', str(gpu),
        '--script', 'scripts/phase0/w14b_renge_router.py', '--output', str(directory), '--', *arguments],
        cwd=ROOT, check=True)


def run(output, job_root, *, publish):
    primary = ROOT / 'outputs/w14_loro_v1/evaluation'
    w14 = ROOT / 'outputs/w14_loro_v1'
    renge = ROOT / 'outputs/w14b_renge_router_v1'
    output, job_root = Path(output).resolve(), Path(job_root).resolve()
    if not output.is_relative_to(ROOT / 'outputs') or not job_root.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Recovery outputs must be in repository outputs/')
    recovery = Artifacts(output, seed=[0, 1, 2], inputs=[Path(__file__), primary / 'summary.json'])
    try:
        result = scientific_completion(primary)
        frozen = freeze_grid(w14, verify_lock(w14))
        recorded = json.loads((primary / 'frozen_grid.json').read_text())['predictions']
        if frozen != recorded:
            raise ValueError('Original complete prediction grid changed')
        m = verify_renge(renge)
        if len(m['jobs']) != 576:
            raise ValueError('Expected original fixed 576-job RENGE grid')
        if any((renge / 'training').glob('*/*/*/status.json')):
            raise ValueError('Existing RENGE training retained; explicit partial recovery needed')
        old = {str(p.relative_to(ROOT)): json.loads(p.read_text()) for p in (
            ROOT / 'outputs/w14_program_v1/status.json', renge / 'queue_status.json',
            ROOT / 'outputs/w14_jobs_20261018/renge_controller/status.json')}
        recovery.json('previous_controller_states.json', old)
        recovery.json('scientific_completion.json', dict(status='verified', primary_result=result,
            primary_summary_sha256=sha256(primary / 'summary.json'), frozen_predictions=len(frozen),
            W14_manifest_sha256=sha256(w14 / 'study_manifest.json'),
            RENGE_manifest_sha256=sha256(renge / 'study_manifest.json'),
            old_failed_records_preserved=True, publication_is_not_computation=True))
        # Start already-authorized unchanged workers BEFORE optional publication.
        for i, gpu in enumerate((1, 2, 3)):
            launch(gpu, job_root / f'renge_worker{i}_gpu{gpu}', ['worker', '--root', str(renge), '--worker', str(i)])
        atomic_json(renge / 'queue_status.json', dict(status='training', recovery=str(output),
            dependency='verified_W14_evaluation_not_Git_publication', original_jobs=576))
        recovery.json('status.json', dict(status='renge_training', original_jobs=576))
        if publish:
            files = [ROOT / 'docs/current/w14_results.md'] + [primary / n for n in (
                'RESULTS.md', 'summary.json', 'metrics.csv', 'comparisons.csv', 'frozen_grid.json')]
            try_publish(recovery, 'w14_publication.json', files, 'W14: publish complete 81-job cross-replicate results')
            try:
                files = finish_supplementary(primary, recovery)
                for dirname, document, names in (
                    ('w14b_data_readiness_v1', 'w14b_data_results.md', ('RESULTS.md', 'summary.json', 'source_records.json')),
                    ('w14b_traxler_su_v1', 'w14b_su_results.md', ('RESULTS.md', 'summary.json', 'technical_qc.csv'))):
                    files += [ROOT / 'docs/current' / document] + [ROOT / 'outputs' / dirname / n for n in names]
                try_publish(recovery, 'supplementary_data_publication.json', files,
                    'W14-B: publish completed S/U counts and recovered supplementary results')
            except Exception as error:
                recovery.json('supplementary_data_publication.json', dict(status='publication_pending',
                    error=repr(error), blocks_downstream=False))
        deadline = time.monotonic() + 7 * 86400
        while time.monotonic() < deadline:
            states = []
            for i, gpu in enumerate((1, 2, 3)):
                p = job_root / f'renge_worker{i}_gpu{gpu}' / 'status.json'
                states.append(json.loads(p.read_text()) if p.exists() else dict(status='launching'))
            if any(s['status'] == 'failed' for s in states):
                raise RuntimeError(f'RENGE scientific worker failed: {states}')
            if all(s['status'] == 'complete' for s in states):
                break
            recovery.json('status.json', dict(status='renge_training', workers=states))
            time.sleep(20)
        else:
            raise TimeoutError('RENGE training timeout')
        evaluation_job = job_root / 'renge_evaluation_gpu1'
        launch(1, evaluation_job, ['evaluate', '--root', str(renge), '--output', str(renge / 'evaluation')])
        while time.monotonic() < deadline:
            p = evaluation_job / 'status.json'
            state = json.loads(p.read_text()) if p.exists() else dict(status='launching')
            if state['status'] == 'failed':
                raise RuntimeError('RENGE scientific evaluation failed')
            if state['status'] == 'complete':
                break
            recovery.json('status.json', dict(status='renge_evaluating'))
            time.sleep(20)
        else:
            raise TimeoutError('RENGE evaluation timeout')
        evaluated = renge / 'evaluation'
        result = json.loads((evaluated / 'summary.json').read_text())
        doc = ROOT / 'docs/current/w14b_renge_results.md'
        if doc.exists():
            raise ValueError('Existing RENGE report retained')
        doc.write_text((evaluated / 'RESULTS.md').read_text())
        publication = None
        if publish:
            publication = try_publish(recovery, 'renge_publication.json', [doc] + [evaluated / n for n in (
                'RESULTS.md', 'summary.json', 'metrics.csv', 'gains.csv', 'frozen_grid.json')],
                'W14-B3: report fixed-recipe RENGE transfer after orchestration recovery')
        recovery.json('summary.json', dict(status='RECOVERED_RENGE_PROGRAM_COMPLETE', result=result, publication=publication))
        recovery.json('status.json', dict(status='complete', computation_complete=True, publication=publication))
        atomic_json(renge / 'queue_status.json', dict(status='complete', recovery=str(output), publication=publication))
    except Exception as error:
        recovery.json('status.json', dict(status='failed', error=repr(error), original_artifacts_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--job-root', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    run(args.output, args.job_root, publish=args.publish)
