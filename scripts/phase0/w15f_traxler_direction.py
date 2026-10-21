"""Short 0→6h perturbation pilot using real mouse ESM and frozen KO splits.

Train on 106 KO backgrounds, evaluate the 13 validation KOs only. The 13 test
KOs remain unopened. Both endpoints have the same KO, before/after infection;
this is genetic-background-conditioned stimulus response, not KO from controls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np

from w4_artifacts import Artifacts,sha256
from w15e_direction_pilot import (create_model,make_train_step,make_route,
    make_candidate_integrator,shuffle_directions,source_state_conditions,read_npz)
from veloroute.latent import load_pack

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training.train_state import TrainState

ARMS=('static','original','direction','direction_shuffled','directU')


def condition(group,count,delta):
    return source_state_conditions(count,group['time_hours'],group['embedding'],delta_hours=delta)


def prepare_groups(c):
    fold=Path(c['fold']);split=json.loads(Path(c['split']).read_text())
    source,_=load_pack(fold/'train_source.npz',expected_side='source')
    target,_=load_pack(fold/'train_target.npz',expected_side='target')
    validation,_=load_pack(fold/'validation_source.npz',expected_side='source')
    esm=read_npz(c['conditions']); metadata=json.loads(str(esm['metadata_json']))
    if not metadata.get('frozen') or not metadata.get('no_cell_expression_used'):
        raise ValueError('Frozen public mouse protein conditions required')
    vectors={str(k):v for k,v in zip(esm['conditions'],esm['embeddings'])}
    for arrays,role in ((source,'train'),(target,'train'),(validation,'val')):
        if set(map(str,arrays['conditions']))!=set(split[role]):
            raise ValueError('Frozen KO assignment changed')
    norm={kind:dict(mean=source[key].mean(0),scale=np.maximum(source[key].std(0),1e-5))
        for kind,key in (('kin','velocity'),('U','u_features'))}

    def groups(arrays,training):
        result=[]
        for label in sorted(set(map(str,arrays['conditions']))):
            mask=arrays['conditions']==label
            g=dict(z=arrays['z'][mask],absolute_velocity=arrays['velocity'][mask],
                cell_kin=arrays['velocity'][mask],cell_directU=arrays['u_features'][mask],
                cell_ids=arrays['cell_ids'][mask],label=label,time_hours=0.,embedding=vectors[label])
            if training:
                g['target']=target['z'][target['conditions']==label]
            else:
                g['score_indices']=np.random.default_rng(0).choice(len(g['z']),
                    min(len(g['z']),c['evaluation_cells']),replace=False)
            result.append(g)
        return result
    return groups(source,True),groups(validation,False),norm,split


def apply_arm(group,norm,arm):
    if arm not in ARMS:raise ValueError('Unknown short perturbation arm')
    kind='U' if arm=='directU' else 'kin'
    key='cell_directU' if kind=='U' else 'cell_kin'
    feature=(group[key]-norm[kind]['mean'])/norm[kind]['scale']
    if arm=='static':feature=np.zeros_like(feature)
    velocity=group['absolute_velocity']
    if arm=='direction_shuffled':velocity=shuffle_directions(velocity,'Traxler_0to6|'+group['label'])
    return dict(group,kinetic_input=np.asarray(feature,dtype='float32'),absolute_velocity=velocity)


def train(model,step,groups,c,seed,arm,callback):
    g=groups[0]
    initial=model.init(jax.random.PRNGKey(seed),jnp.asarray(g['z'][:1]),
        jnp.asarray(g['kinetic_input'][:1]),jnp.asarray(g['z'][:1]),.5,condition(g,1,c['delta_hours']))['params']
    state=TrainState.create(apply_fn=model.apply,params=initial,tx=optax.adam(c['learning_rate']))
    rng=np.random.default_rng(seed);history=[];started=time.monotonic()
    for update in range(1,c['pilot_steps']+1):
        g=groups[int(rng.integers(len(groups)))];n=c['batch_size']
        si=rng.choice(len(g['z']),n,replace=len(g['z'])<n)
        ti=rng.choice(len(g['target']),n,replace=len(g['target'])<n)
        strength=(c['direction_strength']*min(update/c['direction_warmup_steps'],1.)
                  if arm in ('direction','direction_shuffled') else 0.)
        state,metrics=step(state,jax.random.fold_in(jax.random.PRNGKey(seed),update),
            jnp.asarray(g['z'][si]),jnp.asarray(g['target'][ti]),jnp.asarray(g['kinetic_input'][si]),
            jnp.asarray(g['absolute_velocity'][si]),condition(g,n,c['delta_hours']),jnp.asarray(strength))
        if update==1 or update%c['log_interval']==0 or update==c['pilot_steps']:
            row=dict(step=update,elapsed_seconds=time.monotonic()-started,**{k:float(v) for k,v in metrics.items()})
            if not np.isfinite(list(row.values())).all():raise FloatingPointError('Nonfinite perturbation loss')
            history.append(row);callback(row)
    return state.params,initial,history


def predict(route,integrate,params,source,c,arm):
    if 'target' in source:raise ValueError('Prediction source contains future')
    indices=source['score_indices'];batch=c['prediction_batch'];qq=[];xx=[]
    strength=c['direction_strength'] if arm in ('direction','direction_shuffled') else 0.
    for begin in range(0,len(indices),batch):
        ix=indices[begin:begin+batch];n=len(ix)
        pad=lambda v:np.pad(v[ix],((0,batch-n),(0,0)),mode='edge')
        z=jnp.asarray(pad(source['z']));cond=condition(source,batch,c['delta_hours'])
        q,_=route(params,z,jnp.asarray(pad(source['kinetic_input'])),
            jnp.asarray(pad(source['absolute_velocity'])),cond,jnp.asarray(strength))
        x=integrate(params,z,cond,1.)
        xx.append(np.asarray(x)[:n]);qq.append(np.asarray(q)[:n])
    return dict(candidates=np.concatenate(xx),q=np.concatenate(qq),source_ids=source['cell_ids'][indices])


def evaluate(root,c,frozen,split):
    from w14_loro import energy
    from w7_router_scaling import write_csv
    for r in frozen:
        if sha256(r['path'])!=r['sha256']:raise ValueError('Frozen prediction changed')
    # All models and validation predictions exist before any validation future is opened.
    target,_=load_pack(Path(c['fold'])/'validation_target.npz',expected_side='target')
    if set(map(str,target['conditions']))!=set(split['val']):raise ValueError('Validation target KO mismatch')
    rows=[]
    for label in sorted(split['val']):
        y=target['z'][target['conditions']==label]
        y=y[np.random.default_rng(0).choice(len(y),min(len(y),c['evaluation_cells']),replace=False)]
        for arm in ARMS:
            records=[r for r in frozen if r['KO']==label and r['arm']==arm]
            if len(records)!=len(c['seeds']):raise ValueError('Incomplete seed mixture')
            packs=[read_npz(r['path']) for r in records]
            for p in packs[1:]:np.testing.assert_array_equal(p['source_ids'],packs[0]['source_ids'])
            x=np.concatenate([p['candidates'].reshape(-1,50) for p in packs])
            w=np.concatenate([p['q'].ravel()/len(p['q'])/len(packs) for p in packs])
            rows.append(dict(KO=label,arm=arm,energy=energy(x,w,y,'cuda'),
                source_cells=len(packs[0]['q']),target_cells=len(y)))
    comparisons=[]
    for label in sorted(split['val']):
        e={r['arm']:r['energy'] for r in rows if r['KO']==label}
        comparisons.append(dict(KO=label,**{f'gain_vs_{a}':1-e['direction']/e[a]
            for a in ARMS if a!='direction'}))
    write_csv(root/'metrics.csv',rows);write_csv(root/'comparisons.csv',comparisons)
    return dict(status='TRAXLER_DIRECTION_DEVELOPMENT_COMPLETE',task='KO_background_conditioned_0to6h_infection_response',
        train_KOs=len(split['train']),validation_KOs=len(split['val']),test_KOs=len(split['test']),test_targets_read=False,
        fits=len(ARMS)*len(c['seeds']),mean_gains={k:float(np.mean([r[k] for r in comparisons]))
            for k in comparisons[0] if k!='KO'},frozen_predictions=len(frozen),
        velocity='source_only_beta1_tail_OLS_no_neighborhood_pooling',independent_confirmation=False,
        conditions='frozen_mouse_ESM2_3B',delta_hours=6,tau_end=1.,automatic_retuning=False)


def program(config_path,output):
    from veloroute.gpu_policy import enforce_gpu_policy
    enforce_gpu_policy('cuda')
    c=json.loads(Path(config_path).read_text());deadline=time.monotonic()+c['maximum_hours']*3600
    # Wait for protein-only preprocessing; the test outcome remains unopened.
    while not Path(c['conditions']).is_file():
        s=Path(c['conditions']).parent/'status.json'
        if s.exists() and json.loads(s.read_text())['status']=='failed':raise RuntimeError('Protein embedding failed')
        if time.monotonic()>deadline:raise TimeoutError('Protein-condition wait exceeded deadline')
        time.sleep(15)
    fold=Path(c['fold'])
    run=Artifacts(output,seed=c['seeds'],inputs=[Path(config_path),Path(__file__),
        Path(__file__).with_name('w15e_direction_pilot.py'),Path(c['conditions']),Path(c['split']),
        fold/'train_source.npz',fold/'train_target.npz',fold/'validation_source.npz',fold/'transform.npz'])
    try:
        training,validation,norm,split=prepare_groups(c)
        run.json('study_manifest.json',dict(config=c,arms=list(ARMS),train_KOs=split['train'],validation_KOs=split['val'],
            test_KOs=split['test'],test_targets_read=False,validation_targets_read=False))
        model=create_model(c,50,0,training[0]['embedding'])
        step=make_train_step(model,c);route=make_route(model);integrate=make_candidate_integrator(model)
        frozen=[]
        for seed in c['seeds']:
            for arm in ARMS:
                if time.monotonic()>deadline:raise TimeoutError('Bounded perturbation pilot deadline exceeded')
                groups=[apply_arm(g,norm,arm) for g in training]
                directory=run.directory/'fits'/f'seed{seed}'/arm
                fitrun=Artifacts(directory,seed=seed,inputs=[Path(config_path),Path(__file__)])
                params,initial,history=train(model,step,groups,c,seed,arm,
                    lambda r:fitrun.json('status.json',dict(status='training',**r)))
                checkpoint=directory/'model.msgpack';checkpoint.write_bytes(serialization.to_bytes(params))
                params=serialization.from_bytes(initial,checkpoint.read_bytes())
                fitrun.json('training_history.json',dict(history=history))
                for source in validation:
                    source=apply_arm(source,norm,arm)
                    prediction=predict(route,integrate,params,source,c,arm)
                    path=directory/(source['label']+'.npz');np.savez_compressed(path,**prediction)
                    frozen.append(dict(KO=source['label'],seed=seed,arm=arm,path=str(path.resolve()),sha256=sha256(path),
                        model_sha256=sha256(checkpoint),heldout_future_used=False))
                fitrun.json('status.json',dict(status='complete'))
                run.json('status.json',dict(status='training',completed_models=len(frozen)//len(validation),
                    total_models=len(c['seeds'])*len(ARMS),last_seed=seed,last_arm=arm))
                print(json.dumps(dict(status='model_complete',seed=seed,arm=arm)),flush=True)
        run.json('frozen_grid.json',dict(predictions=frozen,validation_targets_read=False,test_targets_read=False))
        result=evaluate(run.directory,c,frozen,split)
        run.json('summary.json',result);run.json('status.json',dict(status='complete'))
        (run.directory/'RESULTS.md').write_text('# Traxler short perturbation direction pilot\n\n'+json.dumps(result,indent=2)+'\n')
        return result
    except Exception as exc:
        run.json('status.json',dict(status='failed',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--configuration',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();print(json.dumps(program(args.configuration,args.output),indent=2))
