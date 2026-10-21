"""Unmodified official CellFlow OTFM training with fixed source-side conditions."""
from __future__ import annotations

import argparse
import csv
import functools
import json
from pathlib import Path
import time

import numpy as np

from w4_artifacts import Artifacts, sha256
from w4_shorttime import read_count_rows, read_metadata
from w4_streams import SourceTransform
from w15a_cellflow_adapter import (ROOT, TokenNormalizer, SourceCohort, add_cellflow_paths,
    align_conditions_to_source, array_hash, condition_inputs, plugin_network_kwargs, source_tokens)


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def time_from_label(label):
    """Pre-training decision: D3=0h; author's r4 nominal hour labels retained."""
    if label.startswith('D'):
        return 24.*(int(label[1:])-3)
    if label.endswith('h'):
        return float(label[:-1])
    raise ValueError('Unknown preregistered time label')


def prepare(config_path, output):
    config = json.loads(Path(config_path).read_text()); output = Path(output)
    source_data = ROOT/config['data_path']; original = ROOT/config['original_cell_pack']
    transform_path = ROOT/config['transform_path']; transform = SourceTransform.load(transform_path)
    run = Artifacts(output, seed=0, inputs=[Path(config_path), transform_path, original/'manifest.json',
        Path(__file__), Path(__file__).with_name('w15a_cellflow_adapter.py')])
    obs, genes = read_metadata(source_data)
    if not np.array_equal(np.asarray(genes, dtype=str), transform.gene_names):
        raise ValueError('CellFlow gene order differs from frozen L1 transform')
    lookup = {str(key):i for i,key in enumerate(obs.index)}
    cohorts = {}; packs = {}; records=[]
    def load(rep, label, expected_ids):
        key = (rep,label)
        if key in cohorts:
            return cohorts[key]
        indices = np.array([lookup[str(identity)] for identity in expected_ids])
        selected = obs.iloc[indices]
        if not (selected['rep'].astype(str) == rep).all() or not (selected['timepoint'].astype(str) == label).all():
            raise ValueError('Frozen source identities do not match replicate/timepoint')
        s,u = [read_count_rows(source_data,layer,indices) for layer in ('spliced','unspliced')]
        cohorts[key] = SourceCohort('neural_tube_differentiation',time_from_label(label),rep,s,u,expected_ids)
        return cohorts[key]
    for rep in config['training_replicates']:
        initial = load_npz(original/f'train_{rep}_D3.npz')
        baseline = load(rep,'D3',initial['cell_ids'])
        for day in config['training_days'][:-1]:
            old_path = original/f'train_{rep}_D{day}.npz'; old = load_npz(old_path)
            cohort = load(rep,f'D{day}',old['cell_ids'])
            pack = source_tokens(transform,cohort,baseline,neighbors=config['neighbors'])
            np.testing.assert_allclose(pack['z'],old['z'],atol=2e-5,rtol=1e-6)
            packs[(rep,day)] = pack
            path = output/f'train_{rep}_D{day}.npz'
            np.savez_compressed(path,z=pack['z'].astype('float32'),target=old['target'],
                cell_ids=pack['source_ids'],cell_kin=pack['cell_kin'].astype('float32'),
                **{'population_'+k:v.astype('float32') for k,v in pack['population'].items()})
            records.append(dict(replicate=rep,label=f'D{day}',time_hours=cohort.time_hours,
                delta_hours=24.,path=str(path.resolve()),sha256=sha256(path),source_key=cohort.key,
                source_hash=pack['source_hash'],original_pack_sha256=sha256(old_path)))
            run.json('status.json',dict(status='preparing',training_groups=len(records)))
    normalizer = TokenNormalizer.fit(list(packs.values()),
        allowed_training_source_keys=[p['source_key'] for p in packs.values()])
    np.savez_compressed(output/'token_normalizer.npz',
        **{'mean_'+k:v for k,v in normalizer.mean.items()},
        **{'scale_'+k:v for k,v in normalizer.scale.items()},training_source_keys=normalizer.training_source_keys)
    sources=[]
    baseline_raw = load_npz(original/'source_05_to_10.npz')
    baseline = load('r4','05h',baseline_raw['cell_ids'])
    for start,stop in config['development_intervals_hours']:
        old_path=original/f'source_{start:02d}_to_{start+5:02d}.npz';old=load_npz(old_path)
        cohort=load('r4',f'{start:02d}h',old['cell_ids'])
        pack=source_tokens(transform,cohort,baseline,neighbors=config['neighbors'])
        np.testing.assert_allclose(pack['z'],old['z'],atol=2e-5,rtol=1e-6)
        path=output/f'source_{start:02d}h.npz'
        if not path.exists():
            np.savez_compressed(path,z=pack['z'].astype('float32'),cell_ids=pack['source_ids'],
                cell_kin=pack['cell_kin'].astype('float32'),
                **{'population_'+k:v.astype('float32') for k,v in pack['population'].items()})
        sources.append(dict(replicate='r4',label=f'{start:02d}h',time_hours=float(start),
            stop_hours=float(stop),delta_hours=float(stop-start),path=str(path.resolve()),sha256=sha256(path),
            source_key=cohort.key,source_hash=pack['source_hash'],population_reference_is_self=start==5))
    # The clusters are fit ONLY on training targets, as required by CF-H4.
    from sklearn.cluster import KMeans
    train_target=np.concatenate([load_npz(r['path'])['target'] for r in records])
    centers=KMeans(n_clusters=8,n_init=10,random_state=0).fit(train_target).cluster_centers_
    np.save(output/'training_target_mode_centers.npy',centers)
    result=dict(status='CELLFLOW_SOURCE_PACK_READY',config=config,training_groups=records,
        development_sources=sources,token_normalizer_sha256=sha256(output/'token_normalizer.npz'),
        training_target_centers_sha256=sha256(output/'training_target_mode_centers.npy'),
        source_transform_sha256=sha256(transform_path),r4_targets_read=False,
        mean_anchor=False,biological_time_origin='RA_SAG_start_D3',
        r4_time_policy='author_nominal_05_10_15_20_hours_not_GEO_rounded_decimal_days')
    run.json('manifest.json',result);run.json('status.json',dict(status='complete'))
    return result


def load_normalizer(directory):
    a=load_npz(Path(directory)/'token_normalizer.npz')
    return TokenNormalizer({k[5:]:v for k,v in a.items() if k.startswith('mean_')},
        {k[6:]:v for k,v in a.items() if k.startswith('scale_')},tuple(a['training_source_keys'].tolist()))


def read_pack(path, source_key):
    a=load_npz(path)
    return dict(z=a['z'],cell_kin=a['cell_kin'],source_ids=a['cell_ids'],source_key=source_key,
        population={k:a['population_'+k] for k in ('kin','Ssum','directU')},
        **({'target':a['target']} if 'target' in a else {}))


def create_solver(config, conditions, dimension, seed):
    add_cellflow_paths()
    import jax
    import optax
    from cellflow.networks import ConditionalVelocityField
    from cellflow.solvers import OTFlowMatching
    from cellflow._compat import ConstantNoiseFlow
    from cellflow.utils import match_linear
    width,depth=config['width'],config['depth']
    vf=ConditionalVelocityField(output_dim=dimension,max_combination_length=1,
        condition_embedding_dim=config['condition_dim'],condition_mode='deterministic',regularization=0.,
        pooling=config.get('pooling','mean'),time_freqs=config.get('time_freqs',64),
        time_encoder_dims=(config.get('time_width',width),)*config.get('time_depth',2),
        hidden_dims=(width,)*depth,decoder_dims=(width,)*depth,
        **plugin_network_kwargs(dimension,config['condition_dim']))
    match=functools.partial(match_linear,epsilon=config.get('ot_epsilon',.1),
        scale_cost='mean',max_iterations=config.get('ot_max_iterations',200))
    solver=OTFlowMatching(vf,ConstantNoiseFlow(0.),match_fn=match,
        rng=jax.random.PRNGKey(seed),optimizer=optax.adam(config['learning_rate']),
        conditions=conditions,ema=config.get('ema',1.))
    return solver


def source_labeled_step(solver,rng,batch):
    """Official OT/path/objective/update; particle conditions follow src_ixs.

    The original step does not reindex conditions after OT. Group tokens are
    unaffected, but local tokens would otherwise become attached to wrong cells.
    """
    import jax
    from ott.solvers import utils as solver_utils
    from cellflow.solvers.utils import ema_update
    src,tgt=batch['src_cell_data'],batch['tgt_cell_data'];condition=batch.get('condition')
    rng_resample,rng_time,rng_step,rng_noise=jax.random.split(rng,4)
    n=src.shape[0];tau=solver.time_sampler(rng_time,n)
    noise=jax.random.normal(rng_noise,(n,solver.vf.condition_embedding_dim))
    if solver.match_fn is not None:
        transport=solver.match_fn(src,tgt)
        src_indices,tgt_indices=solver_utils.sample_joint(rng_resample,transport)
        src,tgt=src[src_indices],tgt[tgt_indices]
        condition=align_conditions_to_source(condition,src_indices,source_size=n)
    solver.vf_state,loss=solver.vf_step_fn(rng_step,solver.vf_state,tau,src,tgt,condition,noise)
    if solver.ema==1.:
        solver.vf_state_inference=solver.vf_state
    else:
        solver.vf_state_inference=solver.vf_state_inference.replace(
            params=ema_update(solver.vf_state_inference.params,solver.vf_state.params,solver.ema))
    return loss


def fit_groups(solver,groups,config,seed,*,callback=None):
    import jax
    import jax.numpy as jnp
    rng=np.random.default_rng(seed);history=[];started=time.monotonic()
    for step in range(1,config['steps']+1):
        group=groups[int(rng.integers(len(groups)))]
        si=rng.choice(len(group['z']),config['batch_size'],replace=len(group['z'])<config['batch_size'])
        ti=rng.choice(len(group['target']),config['batch_size'],replace=len(group['target'])<config['batch_size'])
        cond={k:jnp.asarray(v[si] if len(v)>1 else v) for k,v in group['condition'].items()}
        loss=source_labeled_step(solver,jax.random.fold_in(jax.random.PRNGKey(seed),step),
            dict(src_cell_data=jnp.asarray(group['z'][si]),tgt_cell_data=jnp.asarray(group['target'][ti]),condition=cond))
        if step==1 or step%config.get('log_interval',250)==0 or step==config['steps']:
            value=float(loss)
            if not np.isfinite(value):raise FloatingPointError('Nonfinite official CellFlow objective')
            row=dict(step=step,loss=value,fit_runtime_seconds=time.monotonic()-started)
            history.append(row)
            if callback:callback(row)
    return history


def train(pack_directory,config_path,output,*,arm='cf_static',seed=0):
    add_cellflow_paths()
    import jax
    from flax import serialization
    config=json.loads(Path(config_path).read_text());pack_directory=Path(pack_directory)
    manifest=json.loads((pack_directory/'manifest.json').read_text())
    if arm=='cf_kin_shuf':raise ValueError('sci-FATE2 has no cross-condition population shuffle')
    run=Artifacts(output,seed=seed,inputs=[Path(config_path),pack_directory/'manifest.json',
        pack_directory/'token_normalizer.npz',Path(__file__),Path(__file__).with_name('w15a_cellflow_adapter.py')])
    norm=load_normalizer(pack_directory);groups=[]
    for row in manifest['training_groups']:
        if sha256(row['path'])!=row['sha256']:raise ValueError('Frozen training pack changed')
        p=read_pack(row['path'],row['source_key'])
        inputs=condition_inputs(p,norm,arm,time_hours=row['time_hours'],condition_delta_hours=24.,
            cell_permutation=np.random.default_rng(0).permutation(len(p['z'])))
        groups.append(dict(z=p['z'],target=p['target'],condition=inputs))
    solver=create_solver(config,{k:v[:1] for k,v in groups[0]['condition'].items()},groups[0]['z'].shape[1],seed)
    def update(row):run.json('status.json',dict(status='training',arm=arm,seed=seed,**row))
    history=fit_groups(solver,groups,config,seed,callback=update)
    (run.directory/'model.msgpack').write_bytes(serialization.to_bytes(solver.vf_state_inference))
    result=dict(status='CELLFLOW_OFFICIAL_TRAINED',arm=arm,seed=seed,config=config,
        pack_directory=str(pack_directory.resolve()),training_groups=len(groups),mean_anchor=False,
        parameters=sum(p.size for p in jax.tree.leaves(solver.vf_state.params)),
        model_sha256=sha256(run.directory/'model.msgpack'),r4_targets_read=False,
        backend=str(jax.devices()),history=history)
    run.json('summary.json',result);run.json('status.json',dict(status='complete'))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','train'])
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--pack');parser.add_argument('--arm',default='cf_static');parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    result=(prepare(args.config,args.output) if args.action=='prepare' else
        train(args.pack,args.config,args.output,arm=args.arm,seed=args.seed))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
