"""Small RENGE direction-router pilot; existing full-grid code is untouched.

All arms use the same learned kinetic router and official expert architecture.
Only the cosine routing bias changes, except the explicitly labelled direct-U
representation control. Absolute, pre-standardization PCA velocity is used for
the cosine, and is compared to the total expert field at its source state.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from w4_artifacts import Artifacts, sha256
from w15c_renge_data import checked, load_fold, read_npz
from w15c_renge_l3 import create_model, kinetic_input, source_state_conditions
from w15a_cellflow_moe import align_ot_indices_jit, flow_matching_loss
from w15a_cellflow_moe_real_predict import make_candidate_integrator

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training.train_state import TrainState
from cellflow.utils import match_linear
from ott.solvers import utils as solver_utils

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('original', 'direction', 'direction_shuffled', 'directU')


def unit(value, epsilon=1e-8):
    value = jnp.asarray(value)
    return value/jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), epsilon)


def directional_weights(base_q, absolute_velocity, expert_start, strength):
    """A soft physical-direction prior; gradients cannot rotate experts to game it."""
    scores = jnp.einsum('nd,nkd->nk', unit(absolute_velocity),
                        unit(jax.lax.stop_gradient(expert_start)))
    biased = jax.nn.softmax(jnp.log(jnp.maximum(base_q, 1e-30))+strength*scores, axis=-1)
    q = jnp.where(strength == 0, base_q, biased)
    return q, scores


def shuffle_directions(value, label, permutation_seed=0):
    """Keep each cell's norm and the cohort's directions, permute their pairing."""
    value = np.asarray(value, dtype='float32')
    lengths = np.linalg.norm(value, axis=1, keepdims=True)
    valid = np.flatnonzero(lengths[:,0] > 1e-8)
    seed = int.from_bytes(hashlib.sha256(str(label).encode()).digest()[:4], 'little')
    rng = np.random.default_rng(np.random.SeedSequence([seed,permutation_seed]))
    result = value.copy()
    if len(valid)>1:
        order = rng.permutation(valid)
        result[valid] = value[order]/lengths[order]*lengths[valid]
    return result


def load_direction_fold(pack, fold, arm):
    fm, groups, source, norm = load_fold(pack,fold)
    original = fm['original_manifest']
    old_groups = {g['label']:g for g in original['training_groups']}
    prepared = []
    for group in groups+[source]:
        old = (old_groups[group['label']] if group['label'] in old_groups else
               dict(path=original['source_path'],sha256=original['source_sha256']))
        saved = read_npz(checked(old['path'],old['sha256']))
        np.testing.assert_array_equal(saved['cell_ids'],group['cell_ids'])
        np.testing.assert_array_equal(saved['z'],group['z'])
        # The parent L1 pack used a zero reference: this is absolute PCA velocity.
        absolute = np.asarray(saved['velocity'],dtype='float32')
        if arm == 'direction_shuffled':
            absolute = shuffle_directions(absolute,fold+'|'+group['label'])
        feature_arm = 'moe_directU' if arm=='directU' else 'moe_kin'
        prepared.append(dict(group,kinetic_input=kinetic_input(group,norm,feature_arm),
                             absolute_velocity=absolute))
    return fm,prepared[:-1],prepared[-1]


def make_train_step(model, configuration):
    match_fn = functools.partial(match_linear,epsilon=configuration['ot_epsilon'],
        scale_cost='mean',max_iterations=configuration['ot_max_iterations'])

    @jax.jit
    def step(state,key,source,target,kappa,absolute,condition,strength):
        pair_key,tau_key = jax.random.split(key)
        coupling = match_fn(source,target)
        si,ti = solver_utils.sample_joint(pair_key,coupling)
        paired = align_ot_indices_jit(source,kappa,target,si,ti,condition=condition)
        tau = jax.random.uniform(tau_key,(len(si),1))
        rate = paired.target_x-paired.source_x

        def objective(params):
            variables = dict(params=params)
            base = model.apply(variables,paired.source_x,paired.kinetic_input,method=model.route)
            start = model.apply(variables,paired.source_x,jnp.zeros((len(si),1)),
                                paired.condition,method=model.velocities)
            q,scores = directional_weights(base,absolute[si],start,strength)
            velocities = model.apply(variables,paired.source_x+tau*rate,tau,
                                      paired.condition,method=model.velocities)
            loss,metrics = flow_matching_loss(q,velocities,rate,model.config.balance_coefficient)
            return loss,dict(metrics,mean_direction_cosine=jnp.mean(jnp.sum(q*scores,axis=1)),
                routing_change=jnp.mean(jnp.abs(q-base)))

        (loss,metrics),gradients = jax.value_and_grad(objective,has_aux=True)(state.params)
        return state.apply_gradients(grads=gradients),dict(metrics,loss=loss)
    return step


def make_route(model):
    @jax.jit
    def route(params,z,kappa,velocity,condition,strength):
        base = model.apply(dict(params=params),z,kappa,method=model.route)
        start = model.apply(dict(params=params),z,jnp.zeros((len(z),1)),condition,method=model.velocities)
        return directional_weights(base,velocity,start,strength)
    return route


def fit(model,step,groups,c,seed,arm,callback):
    first = groups[0]
    initial = model.init(jax.random.PRNGKey(seed),jnp.asarray(first['z'][:1]),
        jnp.asarray(first['kinetic_input'][:1]),jnp.asarray(first['z'][:1]),.5,
        source_state_conditions(1,first['time_hours'],first['embedding']))['params']
    state = TrainState.create(apply_fn=model.apply,params=initial,tx=optax.adam(c['learning_rate']))
    rng = np.random.default_rng(seed); history=[]; started=time.monotonic()
    for update in range(1,c['pilot_steps']+1):
        group = groups[int(rng.integers(len(groups)))]; n=c['batch_size']
        si=rng.choice(len(group['z']),n,replace=len(group['z'])<n)
        ti=rng.choice(len(group['target']),n,replace=len(group['target'])<n)
        strength=(c['direction_strength']*min(update/c['direction_warmup_steps'],1.)
                  if arm in ('direction','direction_shuffled') else 0.)
        state,metrics=step(state,jax.random.fold_in(jax.random.PRNGKey(seed),update),
            jnp.asarray(group['z'][si]),jnp.asarray(group['target'][ti]),
            jnp.asarray(group['kinetic_input'][si]),jnp.asarray(group['absolute_velocity'][si]),
            source_state_conditions(n,group['time_hours'],group['embedding']),jnp.asarray(strength))
        if update==1 or update%c['log_interval']==0 or update==c['pilot_steps']:
            row=dict(step=update,elapsed_seconds=time.monotonic()-started,
                direction_strength=strength,**{k:float(v) for k,v in metrics.items()})
            if not np.isfinite(list(row.values())).all():
                raise FloatingPointError('Nonfinite directional pilot loss')
            history.append(row); callback(row)
    return state.params,initial,history


def predict(model,route,integrate,params,source,c,arm):
    if 'target' in source:
        raise ValueError('Source-only prediction required')
    batch=c['prediction_batch']; q_parts=[]; angle_parts=[]
    strength=c['direction_strength'] if arm in ('direction','direction_shuffled') else 0.
    for begin in range(0,len(source['z']),batch):
        end=min(begin+batch,len(source['z'])); n=end-begin
        pad=lambda x: np.pad(x[begin:end],((0,batch-n),(0,0)),mode='edge')
        q,scores=route(params,jnp.asarray(pad(source['z'])),jnp.asarray(pad(source['kinetic_input'])),
            jnp.asarray(pad(source['absolute_velocity'])),
            source_state_conditions(batch,source['time_hours'],source['embedding']),jnp.asarray(strength))
        q_parts.append(np.asarray(q)[:n]); angle_parts.append(np.asarray(scores)[:n])
    q=np.concatenate(q_parts); scores=np.concatenate(angle_parts)
    indices=source['score_indices'][:c['evaluation_source_cells']]
    pieces=[]
    for begin in range(0,len(indices),batch):
        ix=indices[begin:begin+batch]; n=len(ix)
        z=np.pad(source['z'][ix],((0,batch-n),(0,0)),mode='edge')
        result=integrate(params,jnp.asarray(z),
            source_state_conditions(batch,source['time_hours'],source['embedding']),1.)
        pieces.append(np.asarray(result)[:n])
    candidates=np.concatenate(pieces)
    return dict(candidates=candidates,q=q[indices],source_ids=source['cell_ids'][indices],
        direction_cosines=scores[indices],full_source_mean_q=q.mean(0))


def evaluate(root,c,frozen):
    from w15c_renge_program import load_target
    from w14_loro import energy
    # Every fit must be frozen before opening any held-out destination counts.
    for row in frozen:
        checked(row['path'],row['sha256']); checked(row['model'],row['model_sha256'])
    rows=[]
    for fold in c['folds']:
        pack=json.loads((Path(c['pack'])/'study_manifest.json').read_text())
        record=next(r for r in pack['folds'] if r['name']==fold)
        target,metadata=load_target(record); target=target[:c['evaluation_target_cells']]
        for arm in ARMS:
            selected=[r for r in frozen if r['fold']==fold and r['arm']==arm]
            if len(selected)!=len(c['seeds']):raise ValueError('Incomplete pilot seed mixture')
            packs=[read_npz(r['path']) for r in selected]
            for p in packs[1:]:np.testing.assert_array_equal(p['source_ids'],packs[0]['source_ids'])
            x=np.concatenate([p['candidates'].reshape(-1,50) for p in packs])
            weights=np.concatenate([p['q'].ravel()/len(p['q'])/len(packs) for p in packs])
            rows.append(dict(fold=fold,arm=arm,energy=energy(x,weights,target,'cuda'),
                source_cells=len(packs[0]['q']),target_cells=len(target),**metadata))
    comparisons=[]
    for fold in c['folds']:
        e={r['arm']:r['energy'] for r in rows if r['fold']==fold}
        comparisons.append(dict(fold=fold,**{f'gain_vs_{control}':1-e['direction']/e[control]
            for control in ('original','direction_shuffled','directU')}))
    from w7_router_scaling import write_csv
    write_csv(root/'metrics.csv',rows); write_csv(root/'comparisons.csv',comparisons)
    return dict(status='DIRECTION_PILOT_COMPLETE',fits=len(frozen),comparisons=comparisons,
        mean_gains={key:float(np.mean([r[key] for r in comparisons])) for key in comparisons[0] if key!='fold'},
        scope='exploratory_two_intervals_one_TF_no_significance_claim',
        velocity_semantics='absolute_64NN_pre_standardization_PCA_velocity',
        test_dataset='RENGE',new_dataset_targets_read=False,
        expert_direction_gradients='stop_gradient_in_cosine_bias',automatic_retuning=False)


def program(config_path,output):
    from veloroute.gpu_policy import enforce_gpu_policy
    enforce_gpu_policy('cuda')
    c=json.loads(Path(config_path).read_text())
    run=Artifacts(output,seed=c['seeds'],inputs=[Path(config_path),Path(__file__),
        ROOT/'scripts/phase0/w15c_renge_l3.py',ROOT/'scripts/phase0/w15a_cellflow_moe.py',
        ROOT/'scripts/phase0/w15a_cellflow_moe_official.py'])
    run.json('study_manifest.json',dict(config=c,arms=list(ARMS),heldout_targets_read=False,
        selection='first_TF_alphabetically_in_frozen_grid_both_intervals',planned_fits=16))
    deadline=time.monotonic()+c['maximum_hours']*3600; frozen=[]
    try:
        _,first,_=load_direction_fold(c['pack'],c['folds'][0],'original')
        model=create_model(c,50,0,first[0]['embedding'])
        step=make_train_step(model,c); route=make_route(model); integrate=make_candidate_integrator(model)
        for fold in c['folds']:
            for seed in c['seeds']:
                for arm in ARMS:
                    if time.monotonic()>deadline:raise TimeoutError('Small pilot deadline exceeded')
                    _,groups,source=load_direction_fold(c['pack'],fold,arm)
                    directory=run.directory/'fits'/fold/f'seed{seed}'/arm
                    fitrun=Artifacts(directory,seed=seed,inputs=[Path(config_path),Path(__file__)])
                    progress=lambda row:fitrun.json('status.json',dict(status='training',**row))
                    params,initial,history=fit(model,step,groups,c,seed,arm,progress)
                    model_path=directory/'model.msgpack'; model_path.write_bytes(serialization.to_bytes(params))
                    params=serialization.from_bytes(initial,model_path.read_bytes())
                    fitrun.json('training_history.json',dict(history=history))
                    prediction=predict(model,route,integrate,params,source,c,arm)
                    path=directory/'predictions.npz'; np.savez_compressed(path,**prediction)
                    record=dict(fold=fold,seed=seed,arm=arm,path=str(path.resolve()),sha256=sha256(path),
                        model=str(model_path.resolve()),model_sha256=sha256(model_path),heldout_future_used=False)
                    fitrun.json('prediction_manifest.json',record)
                    fitrun.json('status.json',dict(status='complete'))
                    frozen.append(record)
                    run.json('status.json',dict(status='training',completed=len(frozen),total=16,last=record))
                    print(json.dumps(dict(status='fit_complete',completed=len(frozen),fold=fold,seed=seed,arm=arm)),flush=True)
        run.json('frozen_grid.json',dict(predictions=frozen,heldout_targets_read=False))
        result=evaluate(run.directory,c,frozen)
        run.json('summary.json',result)
        (run.directory/'RESULTS.md').write_text('# Direction router pilot\n\n'+json.dumps(result,indent=2)+'\n')
        run.json('status.json',dict(status='complete',fits=len(frozen)))
        return result
    except Exception as exc:
        run.json('status.json',dict(status='failed',error=repr(exc))); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--configuration',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(program(args.configuration,args.output),indent=2))
