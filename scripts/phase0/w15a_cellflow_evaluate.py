"""Frozen prediction development scores; new data are never opened here."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from w4_artifacts import Artifacts,sha256
from w4_shorttime import read_count_rows,read_metadata
from w4_streams import SourceTransform
from w15a_cellflow_adapter import ROOT
from w15a_cellflow_train import load_npz


def score(x,y,centers):
    energy=float(2*cdist(x,y).mean()-cdist(x,x).mean()-cdist(y,y).mean())
    px=np.bincount(cdist(x,centers).argmin(1),minlength=len(centers))/len(x)
    py=np.bincount(cdist(y,centers).argmin(1),minlength=len(centers))/len(y)
    return dict(energy=energy,energy_per_sqrt_dimension=energy/np.sqrt(x.shape[1]),
        mean_mse=float(np.mean((x.mean(0)-y.mean(0))**2)),
        mode_proportion_TV=float(.5*np.abs(px-py).sum()),
        variance_ratio=float(x.var(0).sum()/max(y.var(0).sum(),1e-12)))


def evaluate(prediction_directories,output):
    # Freeze ALL supplied seeds before reading any development target.
    manifests=[]
    for directory in prediction_directories:
        directory=Path(directory);m=json.loads((directory/'prediction_manifest.json').read_text())
        if m['target_read'] or sha256(directory/'predictions.npz')!=m['predictions_sha256']:
            raise ValueError('Predictions must be frozen and source-only before evaluation')
        manifests.append((directory,m))
    if len({m['arm'] for _,m in manifests})!=1 or len({m['seed'] for _,m in manifests})!=len(manifests):
        raise ValueError('Evaluate one arm with unique seeds at a time')
    directory=Path(manifests[0][1]['pack_directory']);pm=json.loads((directory/'manifest.json').read_text());c=pm['config']
    if any(m['pack_directory']!=str(directory) for _,m in manifests):raise ValueError('Seed packs differ')
    run=Artifacts(output,seed=[m['seed'] for _,m in manifests],inputs=[p/'prediction_manifest.json' for p,_ in manifests]+[Path(__file__)])
    run.json('frozen_grid.json',dict(predictions=[dict(path=str(p.resolve()),sha256=m['predictions_sha256']) for p,m in manifests]))
    transform=SourceTransform.load(ROOT/c['transform_path']);data=ROOT/c['data_path'];obs,_=read_metadata(data)
    centers=np.load(directory/'training_target_mode_centers.npy',allow_pickle=False)
    targets={};rows=[];mixtures={}
    for prediction,m in manifests:
        values=load_npz(prediction/'predictions.npz')
        for window in m['windows']:
            stop=int(window['target_hours'])
            if stop not in targets:
                rows_target=np.flatnonzero((obs['rep'].astype(str).to_numpy()=='r4')&
                    (obs['timepoint'].astype(str).to_numpy()==f'{stop:02d}h'))
                s,u=[read_count_rows(data,layer,rows_target) for layer in ('spliced','unspliced')]
                keep=(np.asarray(s.sum(1)).ravel()>=500)&(np.asarray(u.sum(1)).ravel()>=20)
                z=transform.state(s[keep]);choice=np.random.default_rng(c['evaluation_seed']).choice(len(z),
                    min(len(z),c['evaluation_target_cells']),replace=False)
                targets[stop]=z[choice]
            x=values[window['key']];y=targets[stop]
            rows.append(dict(arm=m['arm'],seed=m['seed'],key=window['key'],source_hours=window['source_hours'],
                target_hours=stop,kernel=window['kernel'],source_cells=len(x),target_cells=len(y),**score(x,y,centers)))
            mixtures.setdefault(window['key'],[]).append(x)
    mixed=[]
    for w in manifests[0][1]['windows']:
        x=np.concatenate(mixtures[w['key']]);y=targets[int(w['target_hours'])]
        mixed.append(dict(arm=manifests[0][1]['arm'],seed='mixture',key=w['key'],source_hours=w['source_hours'],
            target_hours=w['target_hours'],kernel=w['kernel'],source_cells=len(x),target_cells=len(y),**score(x,y,centers)))
    selected=[r['energy'] for r in mixed if r['kernel']=='decay' and
        [r['source_hours'],r['target_hours']] in c['selection_intervals_hours']]
    with (run.directory/'metrics.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows+mixed)
    result=dict(status='CELLFLOW_DEVELOPMENT_SCORED',arm=manifests[0][1]['arm'],
        seeds=[m['seed'] for _,m in manifests],selection_energy=float(np.mean(selected)),
        equal_weight_selection_windows=c['selection_intervals_hours'],seed_mixture_metrics=mixed,
        development_only=True,Traxler_read=False,static_hyperparameters_can_be_selected=True)
    run.json('summary.json',result);run.json('status.json',dict(status='complete'))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--predictions',nargs='+',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(evaluate(args.predictions,args.output),indent=2))
