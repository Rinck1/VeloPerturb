"""Repeated positive/null implementation fixture; never reads biological counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from w4_artifacts import Artifacts
from w7_router_scaling import write_csv
from w14_loro import energy
from w14_model import fit,source_predictions


def fixture(seed,positive):
    rng=np.random.default_rng(seed);z=rng.normal(0,.02,(64,2)).astype('float32')
    def group(v):
        p=1/(1+np.exp(-2*v)) if positive else .5
        y=rng.binomial(1,p,128).astype(float)*3
        y-=y.mean();target=np.c_[y,rng.normal(0,.03,len(y))].astype('float32')
        return dict(z=z.copy(),velocity=np.tile([v,0.],(len(z),1)).astype('float32'),
            global_rate=np.zeros(2,dtype='float32'),target=target)
    return [group(v) for v in (-1.5,-.5,.5,1.5)],[group(v) for v in (-1.,1.)]


def run(config_path,output,device):
    config_path=Path(config_path);base=json.loads(config_path.read_text())
    run=Artifacts(output,seed=list(range(base['synthetic_repetitions_per_scenario'])),
        inputs=[config_path,Path(__file__),Path(__file__).with_name('w14_model.py')])
    config=dict(base,experts=2,hidden=24,expert_rank=8,router_hidden=16,
        steps=base['synthetic_steps'],checkpoint_steps=[],source_batch=32,target_batch=64,learning_rate=.004,
        log_interval=base['synthetic_steps'])
    norm=dict(z_mean=np.zeros(2),z_scale=np.ones(2),velocity_mean=np.zeros(2),velocity_scale=np.ones(2))
    rows=[]
    for positive in (True,False):
        for seed in range(base['synthetic_repetitions_per_scenario']):
            train,test=fixture(seed+1000,positive);scores={}
            for arm in ('static','real'):
                model,_=fit(train,norm,config,arm=arm,seed=seed,device=device)
                values=[]
                for i,g in enumerate(test):
                    result=source_predictions(model,g,arm,seed,0,i,24)
                    values.append(energy(result['candidates'].reshape(-1,2),result['q'].ravel()/len(result['q']),g['target'],device))
                scores[arm]=float(np.mean(values));del model
            row=dict(positive=positive,seed=seed,static_energy=scores['static'],real_energy=scores['real'],
                gain=1-scores['real']/scores['static'])
            rows.append(row);run.json('status.json',dict(status='calibrating',completed=len(rows),latest=row))
    pos=[r['gain'] for r in rows if r['positive']];null=[r['gain'] for r in rows if not r['positive']]
    finite=bool(np.isfinite(pos+null).all())
    passed=finite and float(np.median(pos))>0 and float(np.mean(null))<=0
    result=dict(status='W14_SYNTHETIC_IMPLEMENTATION_CHECK_PASSED' if passed else 'W14_SYNTHETIC_IMPLEMENTATION_CHECK_FAILED',
        repetitions_per_scenario=base['synthetic_repetitions_per_scenario'],threshold=base['synthetic_detection_gain'],
        positive_mean_gain=float(np.mean(pos)),positive_median_gain=float(np.median(pos)),
        null_mean_gain=float(np.mean(null)),null_median_gain=float(np.median(null)),
        positive_fraction_over_threshold=float(np.mean(np.array(pos)>base['synthetic_detection_gain'])),
        null_fraction_over_threshold=float(np.mean(np.array(null)>base['synthetic_detection_gain'])),
        biological_data_read=False,formal_false_positive_control_claim=False,
        interpretation='Positive median gain and no positive average null gain required; individual null gains occur and are not significance evidence')
    write_csv(run.directory/'replicates.csv',rows);run.json('summary.json',result)
    run.json('status.json',dict(status='complete' if passed else 'failed'))
    (run.directory/'RESULTS.md').write_text('# W14 synthetic fixture\n\n'+json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not passed:raise ArithmeticError('Positive synthetic fixture did not improve')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/w14_loro_v1.json')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda',choices=['cpu','cuda']);a=p.parse_args()
    if a.device=='cuda':
        from veloroute.gpu_policy import enforce_gpu_policy
        enforce_gpu_policy('cuda')
    run(a.config,a.output,a.device)
