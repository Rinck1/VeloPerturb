"""Explain a COMPLETE frozen L3 synthetic gate without altering its decision.

No training or new threshold. All120 fits and480 candidate files are verified
and indexed before synthetic truth is regenerated for descriptive scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from w4_artifacts import Artifacts,sha256
from w7_router_scaling import write_csv


ARMS=('moe_static','moe_kin','moe_kin_shuf')
SCENARIOS=('positive','null')


def require_complete(directory):
    state=json.loads((Path(directory)/'status.json').read_text())
    if state.get('status')!='complete':
        raise ValueError('Complete120-fit grid required; partial candidate values are not read')


def candidate_values(candidates,q):
    x=np.asarray(candidates,dtype=float);q=np.asarray(q,dtype=float)
    if x.ndim!=3 or q.shape!=x.shape[:2] or not len(x) or x.shape[-1]<1:
        raise ValueError('Invalid candidate/q dimensions')
    if not np.isfinite(x).all() or not np.isfinite(q).all() or (q<0).any():
        raise ValueError('Invalid candidate/q values')
    if not np.allclose(q.sum(1),1.,atol=1e-6):raise ValueError('Candidate q is not row stochastic')
    return x,q/q.sum(1,keepdims=True)


def entropy(q,axis=-1):
    return -np.sum(np.where(q>0,q*np.log(np.maximum(q,1e-300)),0.),axis=axis)


def mechanism_metrics(candidates,q,*,mode_amplitude,observed_high_probability):
    """Known +/-mode scoring and geometry; entropy alone is not an expert count."""
    x,q=candidate_values(candidates,q)
    if mode_amplitude<=0 or not 0<=observed_high_probability<=1:
        raise ValueError('Invalid known-mode scoring truth')
    mean=np.einsum('nk,nkd->nd',q,x)
    residual=x-mean[:,None,:]
    within=np.einsum('nk,nkd,nkd->n',q,residual,residual)
    within_axis=np.einsum('nk,nk,nk->n',q,residual[:,:,0],residual[:,:,0])
    uniform_residual=x-x.mean(1,keepdims=True)
    distances=np.linalg.norm(x[:,:,None,:]-x[:,None,:,:],axis=-1)
    pairs=q[:,:,None]*q[:,None,:]
    pair_distance=np.einsum('nkl,nkl->n',pairs,distances)
    distinct_probability=1-np.sum(q*q,axis=1)
    distinct_sum=float(distinct_probability.sum())
    average=q.mean(0);population_entropy=float(entropy(average))
    individual_entropy=float(np.mean(entropy(q)))
    gap=population_entropy-individual_entropy
    if gap < -1e-10:raise ArithmeticError('Entropy concavity violated')
    axis=x[:,:,0];weights=q/len(q);high=axis>0;low=axis<0
    mass_high=float(weights[high].sum());mass_low=float(weights[low].sum())
    center_high=float(np.sum(weights[high]*axis[high])/mass_high) if mass_high>0 else None
    center_low=float(np.sum(weights[low]*axis[low])/mass_low) if mass_low>0 else None
    high_error=abs(center_high-mode_amplitude) if center_high is not None else None
    low_error=abs(center_low+mode_amplitude) if center_low is not None else None
    between=float(np.mean(np.sum((mean-mean.mean(0))**2,axis=1)))
    within_mean=float(within.mean());total=within_mean+between
    return dict(weighted_within_source_expert_variance=within_mean,
        weighted_within_source_expert_axis0_variance=float(within_axis.mean()),
        uniform_within_source_expert_variance=float(np.mean(np.sum(uniform_residual**2,axis=2))),
        weighted_expert_pairwise_distance=float(pair_distance.mean()),
        weighted_distinct_expert_pairwise_distance=float(pair_distance.sum()/distinct_sum) if distinct_sum>0 else None,
        conditional_mean_between_source_variance=between,total_predictive_variance=total,
        within_source_expert_variance_fraction=within_mean/total if total>0 else None,
        population_q_entropy=population_entropy,mean_cell_q_entropy=individual_entropy,
        entropy_gap=max(0.,gap),effective_usage_exp_population_entropy=float(np.exp(population_entropy)),
        population_q_json=json.dumps(average.tolist()),
        predicted_high_mass=mass_high,predicted_low_mass=mass_low,predicted_zero_axis_mass=float(weights[axis==0].sum()),
        observed_high_mass=float(observed_high_probability),
        mode_mass_absolute_error=abs(mass_high-observed_high_probability),
        weighted_absolute_axis_distance_from_known_modes=float(np.sum(weights*np.abs(np.abs(axis)-mode_amplitude))),
        positive_conditional_center=center_high,negative_conditional_center=center_low,
        positive_center_absolute_error=high_error,negative_center_absolute_error=low_error,
        two_sided_center_mean_absolute_error=(high_error+low_error)/2 if high_error is not None and low_error is not None else None,
        missing_positive_side=mass_high==0.,missing_negative_side=mass_low==0.,
        missing_side_count=int(mass_high==0.)+int(mass_low==0.))


def verify_completed_grid(directory,plan,summary):
    directory=Path(directory)
    if plan['mode']!='frozen_L2_full_20_positive_20_null' or len(plan['repetition_seeds'])!=20:
        raise ValueError('Only the full20-positive/20-null plan is analyzed')
    if tuple(plan['scenario_order'])!=SCENARIOS or tuple(plan['arms'])!=ARMS:
        raise ValueError('Original scenario/arm grid changed')
    if summary['status']!='L3_POPULATION_SYNTHETIC_GRID_COMPLETE' or summary['completed_fits']!=120:
        raise ValueError('Full120 completed fits required irrespective of gate decision')
    if summary['plan_sha256']!=sha256(directory/'plan.json'):
        raise ValueError('Completed summary does not bind the verified plan')
    records=[];fits=[]
    for scenario in SCENARIOS:
        for seed in plan['repetition_seeds']:
            initials=[]
            for arm in ARMS:
                fit=directory/'fits'/scenario/f'repetition{seed:02d}'/arm
                record=json.loads((fit/'summary.json').read_text())
                if (record['scenario'],record['repetition'],record['arm'])!=(scenario,seed,arm):
                    raise ValueError('Frozen fit identity mismatch')
                if sha256(fit/'model.msgpack')!=record['model_sha256']:
                    raise ValueError('Frozen synthetic model changed')
                history=json.loads((fit/'training_history.json').read_text())
                if not history or history[-1]['step']!=plan['model_configuration']['steps']:
                    raise ValueError('Incomplete training budget')
                initials.append(record['initial_parameter_sha256'])
                fits.append(dict(scenario=scenario,repetition=seed,arm=arm,model_sha256=record['model_sha256'],
                    summary_path=str((fit/'summary.json').resolve()),summary_sha256=sha256(fit/'summary.json'),
                    original_mean_energy=record['mean_energy']))
                for group_index,_ in enumerate(plan['test_composition_ratios']):
                    group=f'test_{group_index}';path=fit/f'{group}_predictions.npz'
                    with np.load(path,allow_pickle=False) as f:
                        x,q=candidate_values(f['candidates'],f['q'])
                        if x.shape!=(plan['cells_per_cohort'],plan['experts'],plan['dimensions']):
                            raise ValueError('Candidate cohort size/expert/dimension differs from frozen plan')
                    records.append(dict(scenario=scenario,repetition=seed,arm=arm,group=group,
                        group_index=group_index,path=str(path.resolve()),sha256=sha256(path)))
            if len(set(initials))!=1:raise ValueError('Original equal-initialization contract violated')
    actual={p.resolve() for p in (directory/'fits').glob('*/*/*/test_*_predictions.npz')}
    if len(fits)!=120 or len(records)!=480 or actual!={Path(r['path']) for r in records}:
        raise ValueError('Entire original120-fit/480-candidate grid must be complete')
    return records,fits


METRICS=('weighted_within_source_expert_variance','weighted_within_source_expert_axis0_variance',
    'uniform_within_source_expert_variance','weighted_expert_pairwise_distance',
    'population_q_entropy','mean_cell_q_entropy','entropy_gap','mode_mass_absolute_error',
    'weighted_absolute_axis_distance_from_known_modes','two_sided_center_mean_absolute_error',
    'missing_side_count','total_predictive_variance')


def aggregate(rows,fits):
    repetitions=[]
    for fit in fits:
        selected=[r for r in rows if (r['scenario'],r['repetition'],r['arm'])==
                  (fit['scenario'],fit['repetition'],fit['arm'])]
        row={k:fit[k] for k in ('scenario','repetition','arm','original_mean_energy')}
        row['cohorts']=len(selected)
        for metric in METRICS:
            values=[r[metric] for r in selected if r[metric] is not None]
            row[metric]=float(np.mean(values)) if values else None
            row[metric+'_defined_cohorts']=len(values)
        repetitions.append(row)
    comparisons=[]
    for real in [r for r in repetitions if r['arm']=='moe_kin']:
        for control in ('moe_static','moe_kin_shuf'):
            other=next(r for r in repetitions if (r['scenario'],r['repetition'],r['arm'])==
                       (real['scenario'],real['repetition'],control))
            row=dict(scenario=real['scenario'],repetition=real['repetition'],real_arm='moe_kin',control=control)
            for metric in ('original_mean_energy',*METRICS):
                row['real_minus_control_'+metric]=(real[metric]-other[metric]
                    if real[metric] is not None and other[metric] is not None else None)
            comparisons.append(row)
    summaries=[]
    for scenario in SCENARIOS:
        for arm in ARMS:
            selected=[r for r in repetitions if r['scenario']==scenario and r['arm']==arm]
            result=dict(scenario=scenario,arm=arm,repetitions=len(selected))
            for metric in ('original_mean_energy',*METRICS):
                values=[r[metric] for r in selected if r[metric] is not None]
                result[metric+'_mean']=float(np.mean(values)) if values else None
                result[metric+'_median']=float(np.median(values)) if values else None
                result[metric+'_defined_repetitions']=len(values)
            summaries.append(result)
    return repetitions,comparisons,summaries


def analyze(directory,output):
    directory=Path(directory)
    require_complete(directory)
    # Importing the original module only verifies its plan and regenerates toy
    # truth after the complete forecast index. It never invokes run_plan/fit.
    from w15a_cellflow_moe_synthetic import verify_plan,generate_scenario
    plan=verify_plan(directory)
    original=json.loads((directory/'summary.json').read_text())
    records,fits=verify_completed_grid(directory,plan,original)
    run=Artifacts(output,seed=plan['repetition_seeds'],inputs=[Path(__file__),directory/'plan.json',
        directory/'plan_manifest.json',directory/'summary.json'])
    run.json('frozen_candidate_grid.json',dict(candidates=records,fits=fits,
        plan_sha256=sha256(directory/'plan.json'),original_summary_sha256=sha256(directory/'summary.json'),
        completed_fits=120,candidate_files=480,complete_index_before_truth_regeneration=True))
    rows=[]
    for scenario in SCENARIOS:
        for seed in plan['repetition_seeds']:
            truth=generate_scenario(plan,scenario,seed)['test']
            for record in [r for r in records if r['scenario']==scenario and r['repetition']==seed]:
                if sha256(record['path'])!=record['sha256']:raise ValueError('Candidate changed after full-grid freeze')
                cohort=truth[record['group_index']]
                high=float(np.mean(cohort['truth']['target_component_high']))
                with np.load(record['path'],allow_pickle=False) as f:
                    metrics=mechanism_metrics(f['candidates'],f['q'],
                        mode_amplitude=plan['target_mode_amplitude'],observed_high_probability=high)
                rows.append(dict(scenario=scenario,repetition=seed,arm=record['arm'],group=record['group'],
                    nominal_source_high_probability=cohort['truth']['nominal_source_high_probability'],
                    nominal_target_high_probability=cohort['truth']['nominal_target_high_probability'],**metrics))
            run.json('status.json',dict(status='descriptive_mechanism_scoring',completed_cohorts=len(rows),total=480))
    repetitions,comparisons,summaries=aggregate(rows,fits)
    write_csv(run.directory/'cohort_mechanisms.csv',rows)
    write_csv(run.directory/'repetition_mechanisms.csv',repetitions)
    write_csv(run.directory/'real_minus_controls.csv',comparisons)
    write_csv(run.directory/'scenario_arm_summary.csv',summaries)
    result=dict(status='COMPLETE_L3_SYNTHETIC_MECHANISM_DESCRIBED',completed_fits=120,candidate_files=480,
        original_population_gate_passed=original['population_composition_learning_gate_passed'],
        original_summary_sha256=sha256(directory/'summary.json'),scenario_arm_summary=summaries,
        gate_decision_unchanged=True,new_pass_thresholds=False,automatic_retraining=False,
        truth_used_for_descriptive_scoring_only=True,partial_results_read=False,real_data_read=False,
        density_estimation=False,experts_not_declared_separated_from_usage_entropy_alone=True,
        interpretation=('Compare q-weighted and uniform expert geometry separately from H(qbar)-mean H(qi); '
            'known +/-mode distance and mass errors separate endpoint collapse from routing allocation. '
            'Repetition means weight the four predeclared test cohorts equally; no new hypothesis decision.'))
    run.json('summary.json',result)
    (run.directory/'RESULTS.md').write_text('# L3 synthetic: routing versus expert geometry\n\n'+json.dumps(result,indent=2)+'\n')
    run.json('status.json',dict(status='complete'));return result


def wait_then_analyze(directory,output,timeout_hours=24):
    deadline=time.monotonic()+timeout_hours*3600
    while time.monotonic()<deadline:
        path=Path(directory)/'status.json'
        if path.exists():
            try:status=json.loads(path.read_text()).get('status')
            except json.JSONDecodeError:
                # The original runner writes status non-atomically. Empty/truncated
                # bytes during that brief write are not a partial result to inspect.
                time.sleep(20);continue
            if status=='failed':raise RuntimeError('Original synthetic execution failed; no partial scoring')
            if status=='complete':return analyze(directory,output)
        time.sleep(20)
    raise TimeoutError('Complete synthetic grid not available before bounded wait deadline')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--synthetic',required=True);p.add_argument('--output',required=True)
    p.add_argument('--wait',action='store_true');p.add_argument('--timeout-hours',type=float,default=24.)
    a=p.parse_args()
    result=(wait_then_analyze(a.synthetic,a.output,a.timeout_hours) if a.wait else analyze(a.synthetic,a.output))
    print(json.dumps({k:result[k] for k in ('status','completed_fits','candidate_files','original_population_gate_passed')},indent=2))
