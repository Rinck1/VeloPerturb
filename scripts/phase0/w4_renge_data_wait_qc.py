"""Wait for completed day2 technical counts, export evidence, then run cohort QC.

Only status/provenance JSON is inspected while waiting. This never freezes the
scientific protocol and never computes expression effects or prediction scores.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from veloroute.artifacts import Run, save_json, sha256
from veloroute.raw import run_command

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--day2-stage', default='/data/yuchang/veloroute_w4/renge/day2_pipeline_v2/day2')
    p.add_argument('--day2-job', default='outputs/w4_renge_day2_job_v2/status.json')
    p.add_argument('--day2-evidence', default='outputs/w4_renge_day2_technical_evidence_v2')
    p.add_argument('--qc-output', default='outputs/w4_cohort_qc_v1')
    p.add_argument('--output', required=True)
    p.add_argument('--timeout-seconds', type=int, default=21600)
    args = p.parse_args(); os.chdir(ROOT); start = time.monotonic()
    config = dict(vars(args), git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  seed=None, argv=sys.argv, script_sha256=sha256(__file__))
    stage = Path(args.day2_stage)
    with Run(args.output, stage='w4_technical_cohort_wait_queue', kind='engineering', config=config,
             inputs=[__file__, ROOT/'scripts/phase0/w4_cohort_qc.py', ROOT/'scripts/phase0/w4_renge_data_export.py']) as run:
        def state(status, **kwargs):
            save_json(run.directory/'status.json', dict(config, status=status,
                      elapsed_seconds=time.monotonic()-start, **kwargs))
        try:
            while True:
                if Path(args.day2_job).exists():
                    job = json.loads(Path(args.day2_job).read_text())
                    if job.get('status') == 'failed':
                        raise RuntimeError('Day2 acquisition/processing job failed; no cohort QC started')
                provenance_file = stage/'provenance.json'
                if provenance_file.exists():
                    provenance = json.loads(provenance_file.read_text())
                    if provenance.get('status') == 'failed':
                        raise RuntimeError('Day2 technical processing failed')
                    if provenance.get('status') == 'complete':
                        break
                if time.monotonic()-start > args.timeout_seconds:
                    raise TimeoutError('Day2 technical-count completion timeout')
                state('waiting_for_day2_completed_provenance')
                time.sleep(30)
            summary = json.loads((stage/'summary.json').read_text())
            if (summary.get('status') != 'COUNTS_TECHNICAL_QC_COMPLETE' or summary.get('day') != 2
                    or summary.get('replication_effects_read') is not False):
                raise ValueError('Day2 completion contract mismatch')
            if sha256(summary['h5ad']) != summary['h5ad_sha256']:
                raise ValueError('Day2 completed count file changed')
            state('exporting_day2_technical_evidence')
            run_command([sys.executable, str(ROOT/'scripts/phase0/w4_renge_data_export.py'),
                         '--source', str(stage), '--output', args.day2_evidence],
                        run.directory/'export_day2', timeout_seconds=600)
            state('running_metadata_only_cohort_QC')
            run_command([sys.executable, str(ROOT/'scripts/phase0/w4_cohort_qc.py'),
                         '--output', args.qc_output], run.directory/'cohort_qc', timeout_seconds=1800,
                        environment={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
            state('complete', cohort_summary=str(Path(args.qc_output)/'summary.json'),
                  preregistration_frozen=False, biological_effects_read=False)
        except Exception as error:
            state('failed', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    main()
