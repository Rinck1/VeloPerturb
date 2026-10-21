"""Approved fixed L2 development grid: six arms x three seeds, one GPU.

The existing static seed0 is reused by configuration/checkpoint hash. Every
source-only prediction is frozen before the shared W13 targets are scored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from w4_artifacts import Artifacts,sha256
from w15a_cellflow_adapter import ROOT
from w15a_cellflow_train import train
from w15a_cellflow_common_evaluation import predict_common,evaluate_common,freeze_common_grid


ARMS=('cf_static','cf_kin','cf_Ssum','cf_directU','cf_kin_cell','cf_kin_cell_shuf')
SEEDS=(0,1,2)


def jobs():
    return [dict(arm=arm,seed=seed,local_shuffle_seed=0 if arm=='cf_kin_cell_shuf' else None,
        reuse_static_seed0=arm=='cf_static' and seed==0) for arm in ARMS for seed in SEEDS]


def commit_bytes(commit,configuration_path):
    relative=Path(configuration_path).resolve().relative_to(ROOT)
    return subprocess.check_output(['git','show',f'{commit}:{relative.as_posix()}'],cwd=ROOT)


def validate_freeze(configuration_path,freeze_path,pack):
    configuration_path,freeze_path,pack=map(Path,(configuration_path,freeze_path,pack))
    freeze=json.loads(freeze_path.read_text());config=json.loads(configuration_path.read_text())
    if not freeze.get('freeze_commit') or not freeze.get('cf_static_frozen'):
        raise ValueError('Committed static-backbone freeze is required')
    if sha256(configuration_path)!=freeze['configuration_sha256']:
        raise ValueError('Selected scientific configuration hash mismatch')
    if commit_bytes(freeze['freeze_commit'],configuration_path)!=configuration_path.read_bytes():
        raise ValueError('Configuration differs from its frozen commit bytes')
    if sha256(pack/'manifest.json')!=freeze['source_pack_manifest_sha256']:
        raise ValueError('Source pack changed after static selection')
    if (config['width'],config['depth'],config['steps'],config['learning_rate'])!=(128,2,5000,.0003):
        raise ValueError('This approved grid inherits exactly the selected fixed recipe')
    for name,digest in freeze['source_hashes'].items():
        if sha256(Path(__file__).with_name(name))!=digest:
            raise ValueError('Published scientific core changed')
    return config,freeze


def reuse_static_training(freeze,configuration):
    directory=Path(freeze['winner']['directory'])/'train'
    summary=json.loads((directory/'summary.json').read_text())
    if (summary['arm'],summary['seed'])!=('cf_static',0) or summary['config']!=configuration:
        raise ValueError('Existing static seed0 does not match the frozen recipe')
    if summary['model_sha256']!=sha256(directory/'model.msgpack'):
        raise ValueError('Existing static seed0 checkpoint changed')
    if not summary['history'] or summary['history'][-1]['step']!=configuration['steps']:
        raise ValueError('Static seed0 is not a complete fixed-budget training')
    return directory,summary['model_sha256']


def run_grid(configuration_path,freeze_path,pack,reference,output,*,approved=False):
    if not approved:raise ValueError('Root approval required for fixed 18-model grid')
    config,freeze=validate_freeze(configuration_path,freeze_path,pack)
    reused,reused_sha=reuse_static_training(freeze,config)
    run=Artifacts(output,seed=list(SEEDS),inputs=[Path(configuration_path),Path(freeze_path),
        Path(reference)/'study_manifest.json',Path(reference)/'source_samples.json',Path(__file__),
        Path(__file__).with_name('w15a_cellflow_common_evaluation.py')])
    spec=dict(status='REGISTERED_L2_GRID_BOUND',jobs=jobs(),models=18,new_trainings=17,
        prediction_windows=216,configuration_sha256=freeze['configuration_sha256'],
        freeze_commit=freeze['freeze_commit'],reused_static0_training=str(reused),reused_static0_model_sha256=reused_sha,
        arms=list(ARMS),seeds=list(SEEDS),local_shuffle_permutation_seeds=[0],
        three_training_seeds_are_not_three_permutation_seeds=True,
        population_cf_kin_shuf='NOT_RUN_SINGLE_CONDITION',population_CF_H3='NOT_TESTED',
        no_plugin_HPO=True,Traxler_read=False,reference=str(Path(reference).resolve()),
        pack=str(Path(pack).resolve()))
    run.json('study_manifest.json',spec);frozen_paths=[];trained=[]
    for job in spec['jobs']:
        arm,seed=job['arm'],job['seed'];directory=run.directory/arm/f'seed{seed}'
        directory.mkdir(parents=True)
        # Recheck immutable source before each new model, not an outcome-based gate.
        validate_freeze(configuration_path,freeze_path,pack)
        if job['reuse_static_seed0']:
            training=reused;model_sha=reused_sha
        else:
            training=directory/'train'
            run.json('status.json',dict(status='training',arm=arm,current_seed=seed,
                completed_models=len(trained),total_models=18))
            fitted=train(pack,configuration_path,training,arm=arm,seed=seed)
            model_sha=fitted['model_sha256']
        prediction=directory/'common_predict'
        run.json('status.json',dict(status='source_only_common_prediction',arm=arm,current_seed=seed,
            completed_models=len(trained),total_models=18))
        predicted=predict_common(training,prediction,reference)
        trained.append(dict(**job,training=str(training.resolve()),prediction=str(prediction.resolve()),
            model_sha256=model_sha,predictions_sha256=predicted['predictions_sha256']))
        frozen_paths.append(prediction)
        run.json('completed_models.json',dict(completed=trained,total=18))
    # Independent complete grid check, then a unique pre-evaluation index.
    _,frozen=freeze_common_grid(frozen_paths,reference_root=reference,expected_seeds=SEEDS,expected_arms=ARMS)
    run.json('whole_grid_frozen.json',dict(models=frozen,model_count=len(frozen),window_count=216,
        target_caches_opened_by_this_grid=False))
    run.json('status.json',dict(status='evaluating_complete_frozen_grid',models=18,windows=216))
    result=evaluate_common(frozen_paths,run.directory/'evaluation',reference_root=reference,
        expected_seeds=SEEDS,expected_arms=ARMS,
        mode_centers=Path(pack)/'training_target_mode_centers.npy')
    energies={}
    for arm in ARMS:
        rows=[r for r in result['metrics'] if r['arm']==arm and r['kernel']=='decay'
            and (r['source_hours'],r['target_hours']) in ((5,10),(10,15),(15,20))]
        if len(rows)!=3:raise ValueError('Incomplete primary descriptive windows')
        energies[arm]=sum(r['energy'] for r in rows)/3
    summary=dict(status='REGISTERED_L2_DEVELOPMENT_GRID_COMPLETE',models=18,new_trainings=17,
        primary_three_window_energy=energies,
        kin_gain_vs_controls={control:1-energies['cf_kin']/energies[control]
            for control in ('cf_static','cf_Ssum','cf_directU')},
        local_cell_gain_vs_single_perm_shuffle=1-energies['cf_kin_cell']/energies['cf_kin_cell_shuf'],
        population_CF_H3='NOT_TESTED_SINGLE_CONDITION',local_shuffle_permutation_seeds=[0],
        formal_biological_confirmation=False,development_only=True,Traxler_read=False,
        all_216_source_only_windows_frozen_before_any_scoring=True)
    run.json('summary.json',summary);run.json('status.json',dict(status='complete'))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--configuration',required=True);p.add_argument('--freeze',required=True)
    p.add_argument('--pack',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--approved',action='store_true');args=p.parse_args()
    print(json.dumps(run_grid(args.configuration,args.freeze,args.pack,args.reference,args.output,approved=args.approved),indent=2))
