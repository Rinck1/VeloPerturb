"""Source-only inference from frozen official-CellFlow checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from w4_artifacts import Artifacts,sha256
from w15a_cellflow_adapter import (add_cellflow_paths,condition_inputs,time_schedule,predict_cellflow_tau)
from w15a_cellflow_train import create_solver,load_normalizer,read_pack


def predict(training,output):
    add_cellflow_paths()
    from flax import serialization
    training=Path(training);summary=json.loads((training/'summary.json').read_text())
    if sha256(training/'model.msgpack')!=summary['model_sha256']:
        raise ValueError('Frozen CellFlow checkpoint changed')
    config=summary['config'];directory=Path(summary['pack_directory'])
    manifest=json.loads((directory/'manifest.json').read_text());norm=load_normalizer(directory)
    run=Artifacts(output,seed=summary['seed'],inputs=[training/'summary.json',training/'model.msgpack',
        directory/'manifest.json',directory/'token_normalizer.npz',Path(__file__),
        Path(__file__).with_name('w15a_cellflow_adapter.py')])
    solver=None;predictions={};records=[];choice_by_source={}
    for record in manifest['development_sources']:
        if sha256(record['path'])!=record['sha256']:raise ValueError('Frozen prediction source pack changed')
        pack=read_pack(record['path'],record['source_key'])
        if record['source_key'] not in choice_by_source:
            choice_by_source[record['source_key']]=np.random.default_rng(config['evaluation_seed']).choice(
                len(pack['z']),min(len(pack['z']),config['evaluation_source_cells']),replace=False)
        indices=choice_by_source[record['source_key']]
        for kernel in ('decay','linear'):
            schedule=time_schedule(record['delta_hours'],kernel=kernel)
            cond=condition_inputs(pack,norm,summary['arm'],time_hours=record['time_hours'],
                condition_delta_hours=schedule['condition_delta_hours'],
                cell_permutation=np.random.default_rng(0).permutation(len(pack['z'])))
            if solver is None:
                solver=create_solver(config,{k:v[:1] for k,v in cond.items()},pack['z'].shape[1],summary['seed'])
                state=serialization.from_bytes(solver.vf_state_inference,(training/'model.msgpack').read_bytes())
                solver.vf_state_inference=state;solver.vf_state=state
            pred=predict_cellflow_tau(solver,pack['z'][indices],{k:v[indices] for k,v in cond.items()},
                tau_end=schedule['tau_end'],per_particle=True)
            if not np.isfinite(pred).all():raise FloatingPointError('Nonfinite CellFlow ODE output')
            key=f"{int(record['time_hours']):02d}_{int(record['stop_hours']):02d}_{kernel}"
            predictions[key]=pred
            predictions[key+'_source_ids']=pack['source_ids'][indices]
            records.append(dict(key=key,source_key=record['source_key'],source_hours=record['time_hours'],
                target_hours=record['stop_hours'],source_cells=len(pred),population_reference_is_self=record['population_reference_is_self'],
                **schedule))
            run.json('status.json',dict(status='predicting',completed_windows=len(records)))
    np.savez_compressed(run.directory/'predictions.npz',**predictions)
    result=dict(status='CELLFLOW_PREDICTIONS_FROZEN',arm=summary['arm'],seed=summary['seed'],
        training=str(training.resolve()),model_sha256=summary['model_sha256'],
        predictions_sha256=sha256(run.directory/'predictions.npz'),
        pack_directory=str(directory.resolve()),windows=records,target_read=False,mean_anchor=False)
    run.json('prediction_manifest.json',result);run.json('status.json',dict(status='complete'))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--training',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(predict(args.training,args.output),indent=2))
