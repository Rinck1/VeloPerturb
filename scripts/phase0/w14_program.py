"""Continue all frozen W14 folds through complete-grid scoring and publication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from w4_artifacts import Artifacts
from w7_router_scaling import atomic_json


ROOT=Path(__file__).resolve().parents[2]


def launch(gpu,directory,arguments):
    subprocess.run([sys.executable,str(ROOT/'scripts/phase0/launch_gpu_job.py'),'--gpu',str(gpu),
        '--script','scripts/phase0/w14_loro.py','--output',str(directory),'--',*arguments],cwd=ROOT,check=True)


def run(root,output,job_root,publish):
    root,job_root=Path(root).resolve(),Path(job_root).resolve()
    run=Artifacts(output,seed=[0,1,2],inputs=[root/'study_manifest.json',Path(__file__)])
    try:
        for index,gpu in enumerate((1,2,3)):
            launch(gpu,job_root/f'fold{index}_gpu{gpu}',['worker','--root',str(root),'--worker',str(index)])
        deadline=time.monotonic()+7*86400
        while time.monotonic()<deadline:
            states=[]
            for i,gpu in enumerate((1,2,3)):
                p=job_root/f'fold{i}_gpu{gpu}'/'status.json'
                states.append(json.loads(p.read_text()) if p.exists() else dict(status='launching'))
            if any(s['status']=='failed' for s in states):raise RuntimeError(f'W14 worker failed: {states}')
            run.json('status.json',dict(status='training_folds',workers=states))
            if all(s['status']=='complete' for s in states):break
            time.sleep(15)
        else:raise TimeoutError('W14 exceeded seven days')
        eval_job=job_root/'evaluation_gpu1'
        launch(1,eval_job,['evaluate','--root',str(root),'--output',str(root/'evaluation')])
        while time.monotonic()<deadline:
            p=eval_job/'status.json'
            status=json.loads(p.read_text()) if p.exists() else dict(status='launching')
            if status['status']=='failed':raise RuntimeError('W14 evaluation failed; predictions retained')
            if status['status']=='complete':break
            run.json('status.json',dict(status='evaluating',evaluation=status));time.sleep(15)
        else:raise TimeoutError('W14 evaluation timeout')
        result=json.loads((root/'evaluation/summary.json').read_text())
        report=(root/'evaluation/RESULTS.md').read_text()
        destination=ROOT/'docs/current/w14_results.md'
        if destination.exists():raise ValueError('Refuse to overwrite a prior W14 report')
        destination.write_text(report+'\nFull CSV: `outputs/w14_loro_v1/evaluation/metrics.csv`.\n')
        if publish:
            files=[str(destination.relative_to(ROOT))]+[str((root/'evaluation'/name).relative_to(ROOT)) for name in
                ('RESULTS.md','summary.json','metrics.csv','comparisons.csv','frozen_grid.json')]
            subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
            subprocess.run(['git','commit','-m','W14: record complete fixed-recipe cross-replicate grid'],cwd=ROOT,check=True)
            subprocess.run(['git','push','origin','master'],cwd=ROOT,check=True)
        run.json('summary.json',dict(status='W14_PROGRAM_COMPLETE',result=result,report=str(destination),published=publish))
        run.json('status.json',dict(status='complete'))
    except Exception as error:
        run.json('status.json',dict(status='failed',error=repr(error),automatic_retuning=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='outputs/w14_loro_v1');p.add_argument('--output',required=True)
    p.add_argument('--job-root',default='outputs/w14_jobs_20261018');p.add_argument('--publish',action='store_true');a=p.parse_args()
    run(a.root,a.output,a.job_root,a.publish)
