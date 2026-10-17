"""Queue synthetic H1 design calibration after technical cohort QC completes.

This reads only QC/status JSON, never biological expression, and does not freeze
the protocol or launch the biological replication. Executed calibration code is
pinned in a snapshot when this queue starts.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from w4_artifacts import Artifacts, sha256


ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_CODE = ('w4_h1_calibration.py','w4_streams.py','w4_arms.py','w4_artifacts.py')


def validate_qc(qc):
    tfs = qc.get('H1_TFs', [])
    if (qc.get('status') != 'TECHNICAL_COHORT_QC_COMPLETE'
            or qc.get('biological_effects_read') is not False
            or qc.get('minimum_TF_cells_per_day') != 50
            or not isinstance(tfs, list) or any(not isinstance(t, str) for t in tfs)
            or tfs != sorted(set(tfs)) or len(tfs) < 5
            or set(tfs) & {'CTRL','AAVS1','unassigned'}):
        raise ValueError('Complete technical-only QC and at least five registered TFs required')
    return len(tfs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--qc', default='outputs/w4_cohort_qc_v1/summary.json')
    parser.add_argument('--calibration-output', default='outputs/w4_h1_calibration_formal_v1')
    parser.add_argument('--upstream-status', nargs='+', default=[
        'outputs/w4_renge_day2_job_v2/status.json', 'outputs/w4_renge_cohort_wait_job_v1/status.json'])
    parser.add_argument('--wait-timeout', type=int, default=21600)
    parser.add_argument('--calibration-timeout', type=int, default=14400)
    args = parser.parse_args()
    os.chdir(ROOT)
    qc_path, destination = Path(args.qc).resolve(), Path(args.calibration_output).resolve()
    if destination.exists():
        raise FileExistsError('Preserving existing calibration output; choose a new directory')
    inputs = [Path(__file__)]+[Path(__file__).with_name(name) for name in CALIBRATION_CODE]
    run = Artifacts(args.output, seed=20261017, inputs=inputs)
    try:
        start = time.monotonic()
        while not qc_path.is_file():
            for status_file in args.upstream_status:
                if Path(status_file).is_file():
                    upstream = json.loads(Path(status_file).read_text())
                    if upstream.get('status') == 'failed':
                        raise RuntimeError(f'Upstream job failed: {status_file}')
            if time.monotonic()-start > args.wait_timeout:
                raise TimeoutError('Technical cohort QC did not complete')
            run.json('status.json', dict(status='waiting_for_technical_QC', qc_path=str(qc_path),
                biological_effects_read=False, preregistration_frozen=False))
            time.sleep(30)
        qc = json.loads(qc_path.read_text())
        count = validate_qc(qc)
        command = [sys.executable, str(run.directory.resolve()/'code_snapshot/w4_h1_calibration.py'),
            '--mode','formal','--n-tfs',str(count),'--replicates','200',
            '--bootstrap','10000','--seed','20261017','--qc-manifest',str(qc_path),
            '--output',str(destination)]
        run.json('calibration_launch.json', dict(command=command, H1_TFs=qc['H1_TFs'],
            qc_manifest_sha256=sha256(qc_path), biological_effects_read=False,
            preregistration_frozen=False, scope='synthetic_design_calibration_only'))
        run.json('status.json', dict(status='calibrating', n_TFs=count,
            biological_effects_read=False, preregistration_frozen=False))
        environment = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
        with open(run.directory/'calibration.log','x') as log:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                stderr=subprocess.STDOUT, timeout=args.calibration_timeout)
        if completed.returncode != 0:
            raise RuntimeError(f'Synthetic calibration failed with exit code {completed.returncode}')
        summary_path = destination/'summary.json'
        summary = json.loads(summary_path.read_text())
        if (summary.get('status') != 'complete' or not summary.get('adequate_formal_replication_count')
                or not summary.get('formal_QC_count_bound')):
            raise ValueError('Completed calibration does not satisfy its registered execution contract')
        run.json('summary.json', dict(status='complete', calibration_summary=str(summary_path),
            calibration_summary_sha256=sha256(summary_path), qc_manifest_sha256=sha256(qc_path),
            n_TFs=count, biological_effects_read=False, preregistration_frozen=False,
            biological_replication_started=False))
        run.json('status.json', dict(status='complete', biological_effects_read=False,
            preregistration_frozen=False, biological_replication_started=False))
    except BaseException as error:
        run.json('status.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
            biological_effects_read=False, preregistration_frozen=False))
        raise


if __name__ == '__main__':
    main()
