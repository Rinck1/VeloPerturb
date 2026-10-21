"""Two source-token interventions on frozen cf_kin weights; no model training.

Original endpoints are reused verbatim. Both masked three-seed twelve-window
grids must be frozen before reading a target cache for intervention evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from w4_artifacts import Artifacts,sha256
from w15a_cellflow_adapter import add_cellflow_paths,array_hash,condition_inputs,predict_cellflow_tau,time_schedule
from w15a_cellflow_common_evaluation import (KEYS,align_source_ids,evaluate_common,
    freeze_common_grid,reference_sources,save_common_predictions)
from w15a_cellflow_train import create_solver,load_normalizer,load_npz,read_pack
from w15a_l2_reporter import verify_completed_grid,write_csv


MODES=('training_mean','source05_reference')
ARMS=('cf_kin','cf_kin_mask_training_mean','cf_kin_mask_source05_reference')
SEEDS=(0,1,2)


def source_record(manifest,start):
    rows={(r['path'],r['sha256'],r['source_key']) for r in manifest['development_sources'] if int(r['time_hours'])==start}
    if len(rows)!=1:raise ValueError('Expected a unique complete source cohort at each origin')
    path,digest,key=rows.pop()
    if sha256(path)!=digest:raise ValueError('Frozen intervention source pack changed')
    return read_pack(path,key),dict(path=path,sha256=digest,source_key=key)


def mask_conditions(conditions,mode,reference_standardized):
    if mode not in MODES:raise ValueError('Only the two approved token masks are permitted')
    token=np.asarray(conditions['population_token'])
    reference=np.asarray(reference_standardized,dtype=token.dtype)
    if token.ndim!=3 or token.shape[1]!=1 or reference.shape!=(token.shape[-1],) or not np.allclose(token,token[:1]):
        raise ValueError('Expected a constant population token on the full source cohort')
    result={k:np.asarray(v).copy() for k,v in conditions.items()}
    result['population_token']=(np.zeros_like(token) if mode=='training_mean' else
        np.broadcast_to(reference,token.shape).copy())
    if not np.isfinite(result['population_token']).all():raise ValueError('Nonfinite masked token')
    return result


def restore_solver(training,summary,conditions,dimension):
    add_cellflow_paths()
    from flax import serialization
    solver=create_solver(summary['config'],{k:v[:1] for k,v in conditions.items()},dimension,summary['seed'])
    solver.vf_state_inference=serialization.from_bytes(solver.vf_state_inference,(Path(training)/'model.msgpack').read_bytes())
    return solver


def predict_masked_pair(training,output,reference_root,*,callback=None):
    training=Path(training);tm=json.loads((training/'summary.json').read_text())
    if tm['arm']!='cf_kin' or tm['seed'] not in SEEDS:raise ValueError('Only completed cf_kin three-seed weights are allowed')
    checkpoint=training/'model.msgpack';before=sha256(checkpoint)
    if before!=tm['model_sha256']:raise ValueError('Frozen kinetic checkpoint changed')
    pack_directory=Path(tm['pack_directory']);manifest_path=pack_directory/'manifest.json'
    pm=json.loads(manifest_path.read_text());norm=load_normalizer(pack_directory)
    reference=reference_sources(reference_root);ref_pack,ref_record=source_record(pm,5)
    raw_ref=np.asarray(ref_pack['population']['kin'])
    if not np.array_equal(raw_ref,np.zeros_like(raw_ref)):
        raise ValueError('Expected the frozen05h self-reference raw token to equal zero exactly')
    ref_standardized=np.asarray(norm.apply('kin',raw_ref),dtype='float32')
    forecasts={mode:{} for mode in MODES};token_audit=[];solver=None;completed=0
    for key,(start,stop,clock) in KEYS.items():
        pack,record=source_record(pm,start);common=reference['sources'][start]
        if set(pack['source_ids'])!=set(common['all_cell_ids']):
            raise ValueError('Masked context must retain the complete original source cohort')
        indices=align_source_ids(common['cell_ids'],pack['source_ids'])
        np.testing.assert_allclose(pack['z'][indices],common['z'],rtol=1e-6,atol=2e-5)
        schedule=time_schedule(stop-start,kernel=clock)
        conditions=condition_inputs(pack,norm,'cf_kin',time_hours=float(start),
            condition_delta_hours=schedule['condition_delta_hours'])
        if solver is None:solver=restore_solver(training,tm,conditions,pack['z'].shape[1])
        unchanged_keys=sorted(set(conditions)-{'population_token'})
        for mode in MODES:
            masked=mask_conditions(conditions,mode,ref_standardized)
            for name in unchanged_keys:np.testing.assert_array_equal(masked[name],conditions[name])
            x=predict_cellflow_tau(solver,common['z'],{k:v[indices] for k,v in masked.items()},
                tau_end=schedule['tau_end'],per_particle=True)
            forecasts[mode][key]=dict(x=x,source_ids=common['cell_ids'])
            token_audit.append(dict(seed=tm['seed'],key=key,mode=mode,source_hours=start,
                target_hours=stop,kernel=clock,population_reference_self=start==5,
                original_token_norm=float(np.linalg.norm(conditions['population_token'][0,0])),
                masked_token_norm=float(np.linalg.norm(masked['population_token'][0,0])),
                token_change_norm=float(np.linalg.norm(masked['population_token'][0,0]-conditions['population_token'][0,0])),
                unchanged_condition_sha256=array_hash(*(conditions[k] for k in unchanged_keys)),
                source_context_cells=len(pack['z']),terminal_source_cells=len(indices),
                source_pack_sha256=record['sha256']))
            completed+=1
            if callback:callback(dict(seed=tm['seed'],completed_ode_windows=completed,total_ode_windows=24))
    if sha256(checkpoint)!=before:raise ValueError('Frozen checkpoint mutated during intervention')
    paths=[]
    for mode in MODES:
        path=Path(output)/mode
        save_common_predictions(path,forecasts[mode],reference_root=reference_root,
            arm='cf_kin_mask_'+mode,seed=tm['seed'],model_family='CellFlow_L2_frozen_token_intervention',
            extra=dict(training=str(training.resolve()),model_sha256=before,base_arm='cf_kin',mask=mode,
                source05_standardized_token_sha256=array_hash(ref_standardized),
                source05_reference_record=ref_record,normalizer_sha256=sha256(pack_directory/'token_normalizer.npz'),
                raw_source05_kin_exact_zero=True,weights_updated=False,pack_directory=str(pack_directory.resolve())))
        paths.append(path)
    return paths,token_audit,dict(source05_standardized_token=ref_standardized.tolist(),
        source05_standardized_token_norm=float(np.linalg.norm(ref_standardized)),
        masks_identical=bool(np.array_equal(ref_standardized,np.zeros_like(ref_standardized))),
        training_mean_mask='normalized_token_zero_equals_training_mean',
        source05_reference_mask='normalize_frozen_raw05h_token_zero_equals_minus_mean_over_scale')


def endpoint_changes(original_records,masked_records):
    original={r['seed']:load_npz(r['path']) for r in original_records};rows=[]
    for record in masked_records:
        masked=load_npz(record['path'])
        for key,(start,stop,clock) in KEYS.items():
            dx=np.asarray(masked[key],dtype=float)-original[record['seed']][key]
            rows.append(dict(arm=record['arm'],seed=record['seed'],key=key,source_hours=start,target_hours=stop,
                horizon_hours=stop-start,kernel=clock,population_reference_self=start==5,
                mean_endpoint_l2_change=float(np.linalg.norm(dx,axis=1).mean()),
                endpoint_coordinate_rmse=float(np.sqrt(np.mean(dx*dx))),
                max_absolute_coordinate_change=float(np.abs(dx).max()),
                endpoint_bitwise_identical=bool(np.array_equal(masked[key],original[record['seed']][key]))))
    return rows


def score_comparisons(metrics):
    baseline={r['key']:r for r in metrics if r['arm']=='cf_kin'};windows=[];horizons=[]
    for arm in ARMS[1:]:
        for row in metrics:
            if row['arm']!=arm:continue
            original=baseline[row['key']];start=row['source_hours'];stop=row['target_hours']
            windows.append(dict(mask_arm=arm,key=row['key'],source_hours=start,target_hours=stop,
                horizon_hours=stop-start,kernel=row['kernel'],population_reference_self=start==5,
                original_energy=original['energy'],masked_energy=row['energy'],
                masked_minus_original_energy=row['energy']-original['energy'],
                relative_error_increase=None if original['energy']<=1e-12 else row['energy']/original['energy']-1.,
                original_mean_mse=original['sampled_prediction_mean_mse'],masked_mean_mse=row['sampled_prediction_mean_mse'],
                original_mode_TV=original['mode_proportion_TV'],masked_mode_TV=row['mode_proportion_TV']))
        for clock in ('decay','linear'):
            for duration in (5,10,15):
                group=[r for r in windows if r['mask_arm']==arm and r['kernel']==clock and r['horizon_hours']==duration]
                original=float(np.mean([r['original_energy'] for r in group]));masked=float(np.mean([r['masked_energy'] for r in group]))
                horizons.append(dict(mask_arm=arm,kernel=clock,horizon_hours=duration,windows=len(group),
                    original_energy=original,masked_energy=masked,masked_minus_original_energy=masked-original,
                    relative_error_increase=None if original<=1e-12 else masked/original-1.,
                    self_reference_windows=sum(r['population_reference_self'] for r in group),
                    independent_biological_replicates_inferred=False))
    return windows,horizons


def run_intervention(grid,output,*,approved=False):
    if not approved:raise ValueError('Root approval required before any frozen-weight GPU intervention')
    grid=Path(grid);study,completed,reference,all_original,old_metrics=verify_completed_grid(grid)
    selected=[r for r in completed if r['arm']=='cf_kin'];original=[r for r in all_original if r['arm']=='cf_kin']
    if {r['seed'] for r in selected}!=set(SEEDS):raise ValueError('All three original kinetic seeds are required')
    run=Artifacts(output,seed=list(SEEDS),inputs=[Path(__file__),grid/'study_manifest.json'])
    run.json('intervention_protocol.json',dict(original_grid=str(grid.resolve()),modes=list(MODES),arms=list(ARMS),
        seeds=list(SEEDS),new_training=False,new_ode_predictions=72,reused_original_predictions=36,
        source_IDs_unchanged=True,clocks_unchanged=True,targets_unchanged=True,model_weights_unchanged=True,
        purpose='does_actual_source_token_change_frozen_weight_inference_not_merely_training_history',
        origin05_group_token='raw_zero_self_reference_standardized_constant_no_current_source_increment'))
    paths=[Path(r['prediction']) for r in selected];audit=[];definition=None
    for row in sorted(selected,key=lambda r:r['seed']):
        def update(status):run.json('status.json',dict(status='frozen_weight_predicting',**status))
        added,tokens,current=predict_masked_pair(row['training'],run.directory/f"seed{row['seed']}",study['reference'],callback=update)
        if definition is not None and current!=definition:raise ValueError('Kinetic seeds do not share the same frozen token definition')
        definition=current;paths+=added;audit+=tokens
    ref,frozen=freeze_common_grid(paths,reference_root=study['reference'],expected_seeds=SEEDS,expected_arms=ARMS)
    run.json('whole_intervention_frozen.json',dict(models=9,prediction_windows=108,predictions=frozen,
        targets_opened_for_intervention=False,original_weight_hashes={str(r['seed']):r['model_sha256'] for r in selected}))
    changes=endpoint_changes(original,[r for r in frozen if r['arm']!='cf_kin'])
    ref05=[r for r in changes if r['arm']=='cf_kin_mask_source05_reference' and r['source_hours']==5]
    max_self_ref_error=max(r['max_absolute_coordinate_change'] for r in ref05)
    if max_self_ref_error>1e-5:raise ValueError('Source05 reference intervention must reproduce original self-reference endpoints before scoring')
    evaluated=evaluate_common(paths,run.directory/'evaluation',reference_root=study['reference'],
        expected_seeds=SEEDS,expected_arms=ARMS,mode_centers=Path(study['pack'])/'training_target_mode_centers.npy')
    prior={r['key']:r['energy'] for r in old_metrics if r['arm']=='cf_kin'}
    original_error=max(abs(r['energy']-prior[r['key']]) for r in evaluated['metrics'] if r['arm']=='cf_kin')
    if original_error>1e-12:raise ValueError('Original kinetic Energy changed under the shared unchanged scorer')
    windows,horizons=score_comparisons(evaluated['metrics'])
    for name,rows in (('token_audit.csv',audit),('endpoint_changes.csv',changes),
                      ('window_comparisons.csv',windows),('horizon_comparisons.csv',horizons)):
        write_csv(run.directory/name,rows)
    artifacts=[Path(__file__),grid/'study_manifest.json',grid/'whole_grid_frozen.json',grid/'evaluation/summary.json',
        Path(study['pack'])/'manifest.json',Path(study['pack'])/'token_normalizer.npz']
    artifacts += [Path(r['training'])/name for r in selected for name in ('summary.json','model.msgpack')]
    artifacts += [Path(r[field]) for r in frozen for field in ('path','manifest')]
    artifacts += [Path(r['path']) for r in ref['target_caches'].values()]
    manifest=json.loads((Path(study['pack'])/'manifest.json').read_text())
    artifacts += [Path(r['path']) for r in manifest['development_sources']]
    run.json('provenance.json',dict(artifacts=[dict(path=str(p.resolve()),sha256=sha256(p),bytes=p.stat().st_size)
        for p in sorted(set(artifacts),key=str)],reference_hashes=reference['hashes'],model_weights_updated=False))
    result=dict(status='FROZEN_L2_TOKEN_INTERVENTION_COMPLETE',definition=definition,horizon_comparisons=horizons,
        original_energy_reproduction_max_error=original_error,origin05_reference_endpoint_max_error=max_self_ref_error,
        models_trained=0,new_ode_predictions=72,reused_original_predictions=36,
        all108_predictions_frozen_before_intervention_scoring=True,original_L2_grid_unchanged=True,
        all_sources_targets_IDs_clocks_weights_unchanged=True,Traxler_read=False,
        interpretation='frozen_model_input_sensitivity_and_utility_not_biological_causality_or_new_confirmatory_grid')
    run.json('summary.json',result)
    lines=['# Frozen CellFlow kinetic-token intervention','',
        'No training. Original three-seed checkpoints, source IDs, target caches, clocks and other conditions stay fixed.',
        'Original36 windows are reused; masked72 ODE forecasts are all frozen before intervention scoring.',
        'Training-mean mask sets the STANDARDIZED group token to0; source05 mask uses raw0 normalized by frozen statistics.',
        'These masks are distinct unless the training mean is exactly0. Origin05 group context was already a self-reference constant.',
        '', '| Mask | Clock | Horizon | Original Energy | Masked Energy | Relative error increase |',
        '|---|---|---:|---:|---:|---:|']
    for r in horizons:
        gain='undefined' if r['relative_error_increase'] is None else f"{100*r['relative_error_increase']:+.2f}%"
        lines.append(f"| {r['mask_arm']} | {r['kernel']} | {r['horizon_hours']}h | {r['original_energy']:.6f} | {r['masked_energy']:.6f} | {gain} |")
    lines += ['', 'Positive error increase means actual token is useful relative to that frozen-weight mask; negative means masking improves this metric.',
        'Endpoint changes test whether the model uses the token at inference. They do not alone show that the token is beneficial.',
        'Origin05 reference-mask endpoints should be unchanged; improvements/deterioration from the training-mean mask there only probe a constant context.',
        'Windows overlap; seeds are optimization repeats, not independent biological replicates. This is an exploratory post-main-grid mechanism intervention.', '']
    (run.directory/'RESULTS.md').write_text('\n'.join(lines));run.json('status.json',dict(status='complete'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grid',required=True);p.add_argument('--output',required=True)
    p.add_argument('--approved',action='store_true');args=p.parse_args()
    print(json.dumps(run_intervention(args.grid,args.output,approved=args.approved),indent=2))
