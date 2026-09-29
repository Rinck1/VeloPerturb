"""Fixed-budget Kang runs: shared expert geometry isolates the velocity router.

Target distributions are training supervision only. Held-out predictions consume
control S/U only; evaluation is a separate all-grid-frozen operation.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from .artifacts import Run, load_config, object_hash, save_csv, save_json, sha256
from .full_experiments import condition_token_sets
from .full_model import load_full_checkpoint, save_full_checkpoint
from .gfg_experiments import read_gene_input, train_gfg, training_gene_us
from .gpu_policy import enforce_gpu_policy
from .latent import FrozenSplicingTransform, load_pack
from .mixture_energy import candidate_endpoints, mixture_energy


ROUTER_ARMS = ('static_router','gfg_joint_router','gfg_joint_shuffled_U','gfg_frozen_router','raw_SU_router')


def completed(directory):
    path = Path(directory)/'provenance.json'
    return path.exists() and json.loads(path.read_text())['status'] == 'complete'


def base_config(config, donor, seed, *, experts=2, full=False):
    base = load_config('configs/veloroute_gfg_joint_20260914.yaml')
    t = config['training']
    fold = Path(config['root'])/'folds'/donor
    if sha256(t['gfg_checkpoint']) != t['gfg_checkpoint_sha256']:
        raise ValueError('Registered GFG checkpoint changed')
    base.update(fold=str(fold), source_counts=None, conditions=str(fold/'conditions.npz'),
        gfg_checkpoint=t['gfg_checkpoint'], research_status='Kang_fixed_protocol_donor_holdout', cpu_threads=4,
        device=config.get('device','cuda'))
    base['model'].update(esm_dim=1, hidden_dim=768 if full else t['model_width'],
        residual_blocks=6 if full else t['residual_blocks'], max_experts=experts, top_k=min(2,experts),
        initial_active=1 if full else experts, use_adaptive_modes=full, use_intrinsic=full,
        use_gate=full, use_noise=full, support_start=0., support_end=1.,
        gfg_genes=config['representation']['selected_genes'],state_dim=config['representation']['pca_components'])
    base['training'].update(stage_a_steps=t['full_stage_a_steps'] if full else t['geometry_stage_a_steps'],
        stage_b_steps=t['full_stage_b_steps'] if full else t['geometry_stage_b_steps'],
        stage_c_steps=t['full_stage_c_steps'] if full else t['geometry_stage_c_steps'],
        batch_size=t['source_batch'],time_start=0.,time_end=1.,seed=seed,
        lambda_dynamic=.1 if full else 0., lambda_field=.01 if full else 0.,
        checkpoint_interval=2000 if full else 250, reference_interval=2000 if full else 1000,
        e_interval=300 if full else 50, usage_interval=2000 if full else 1000)
    base['kang_protocol_hash'] = object_hash(config)
    return base


def prepare_geometry(config, donor, seed, checkpoint, output):
    device = config.get('device','cuda')
    enforce_gpu_policy(device)
    torch.set_num_threads(4)
    model, meta = load_full_checkpoint(checkpoint, map_location=device)
    fold = Path(config['root'])/'folds'/donor
    if meta['task'] != config['task']: raise ValueError('Not a Kang model')
    with Run(output, stage='Kang_common_expert_geometry', kind='engineering',
        config=dict(donor=donor,seed=seed,time_coordinate=[0.,1.],geometry_selection='fixed_static_seed_only'),
        inputs=[checkpoint,fold/'train_source.npz',fold/'validation_source.npz'],seed=seed) as run:
        for role in ('train','validation'):
            source, _ = load_pack(fold/f'{role}_source.npz',expected_side='source')
            tokens,mask,_ = condition_token_sets(fold/'conditions.npz',source['conditions'])
            state,embedding,points = [],[],[]
            with torch.no_grad():
                for start in range(0,len(source['z']),256):
                    ix=slice(start,start+256)
                    z=model.encode_state(torch.tensor(source['z'][ix],device=device))
                    e=model.condition_encoder(torch.tensor(tokens[ix],device=device),torch.tensor(mask[ix],device=device))
                    state.append(z.cpu().numpy());embedding.append(e.cpu().numpy())
                    points.append(candidate_endpoints(model,z,e,start=0.,end=1.,step=.25).cpu().numpy())
            with (run.directory/f'{role}.npz').open('xb') as f:
                np.savez_compressed(f,state=np.concatenate(state),embedding=np.concatenate(embedding),
                    candidates=np.concatenate(points),cell_ids=source['cell_ids'],conditions=source['conditions'])
        result=dict(status='KANG_COMMON_GEOMETRY_READY',checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
            heldout_donor=donor,seed=seed,heldout_target_read=False,transform_hash=meta['transform_hash'])
        save_json(run.directory/'summary.json',result)
        (run.directory/'RESULTS.md').write_text('# Kang common candidate geometry\n\n'+json.dumps(result,indent=2))
    return result


def load_geometry(path, role, source):
    path=Path(path)
    provenance=json.loads((path/'provenance.json').read_text())
    file=path/f'{role}.npz'
    if provenance['status']!='complete' or sha256(file)!=provenance['outputs'][file.name]:
        raise ValueError('Unfrozen/modified candidate geometry')
    with np.load(file,allow_pickle=False) as f: values={k:f[k] for k in f.files}
    for k in ('cell_ids','conditions'):
        if not np.array_equal(values[k],source[k]): raise ValueError('Candidate/source alignment mismatch')
    return values


def representation(model,z,us,arm,raw_mean,raw_scale):
    r,_,_=model.dynamics(z,us)
    if arm=='static_router': return torch.zeros_like(r)
    if arm=='raw_SU_router':
        u=us[:,:model.dynamics.genes]
        features=(torch.log1p(u)@model.dynamics.components.T-raw_mean)/raw_scale
        return model.dynamics.representation(features)
    return r


def train_router(config,donor,seed,arm,geometry,output,*,resume=None):
    if arm not in ROUTER_ARMS: raise ValueError('Unregistered router arm')
    device=config.get('device','cuda');enforce_gpu_policy(device);torch.set_num_threads(4)
    fold=Path(config['root'])/'folds'/donor
    us,source,sm=read_gene_input(fold/'train_gene_source.npz',fold/'train_source.npz')
    target,tm=load_pack(fold/'train_target.npz',expected_side='target')
    if sm['role']!='train' or tm['role']!='train' or set(source['conditions'])!=set(target['conditions']):
        raise ValueError('Training donor/type source-target groups differ')
    data=load_geometry(geometry,'train',source)
    gm=json.loads((Path(geometry)/'summary.json').read_text())
    checkpoint=gm['checkpoint']
    if sha256(checkpoint)!=gm['checkpoint_sha256']: raise ValueError('Common checkpoint changed')
    model,_=load_full_checkpoint(Path(resume)/'model.pt' if resume else checkpoint,map_location=device)
    model.config=replace(model.config,use_dynamics=arm!='static_router',joint_dynamics=True)
    model.requires_grad_(False);model.router.requires_grad_(True);model.dynamics.requires_grad_(True)
    if arm=='gfg_frozen_router': model.dynamics.core.requires_grad_(False)
    for e in (model.dynamics.core.manifold_encoder,model.dynamics.core.velocity_encoder):
        e.attention_layer.requires_grad_(False)
    model.eval()
    tensor=lambda x:torch.tensor(x,dtype=torch.float32,device=device)
    if arm=='gfg_joint_shuffled_U': us=training_gene_us(us,source,'gfg_joint_shuffled',seed)
    z,us,state,e,points,y=map(tensor,(source['z'],us,data['state'],data['embedding'],data['candidates'],target['z']))
    raw_features=torch.log1p(us[:,:model.dynamics.genes])@model.dynamics.components.T
    raw_mean=raw_features.mean(0);raw_scale=raw_features.std(0,unbiased=False).clamp_min(1e-3)
    t=config['training']
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW([dict(params=model.router.parameters(),lr=t['router_lr']),
        dict(params=[p for p in model.dynamics.parameters() if p.requires_grad],lr=t['gfg_lr'])])
    labels=sorted(set(source['conditions']))
    sg={c:np.flatnonzero(source['conditions']==c) for c in labels}
    tg={c:np.flatnonzero(target['conditions']==c) for c in labels}
    rng=np.random.default_rng(seed);trace=[];start_step=0
    effective=dict(protocol_hash=object_hash(config),arm=arm,seed=seed,donor=donor,training=t,
                   geometry_sha256=sha256(Path(geometry)/'train.npz'))
    inputs=[fold/'train_gene_source.npz',fold/'train_target.npz',checkpoint,Path(geometry)/'train.npz']
    binding={str(p):sha256(p) for p in inputs}
    meta=dict(task=config['task'],architecture='Kang_common_geometry_velocity_router',arm=arm,seed=seed,donor=donor,
        transform_hash=sm['transform_hash'],protocol_hash=object_hash(config),configuration=effective,
        GFG_checkpoint_sha256=t['gfg_checkpoint_sha256'],GFG_joint=arm in ('gfg_joint_router','gfg_joint_shuffled_U'),
        raw_mean=raw_mean.detach().cpu().tolist(),raw_scale=raw_scale.detach().cpu().tolist(),
        source_only=True,fields_frozen=True,condition_enters_GFG=False,individual_fate_truth_used=False,
        native_GFG_training_control='same_budget_except_explicit_frozen_ablation',geometry=str(geometry))
    if resume:
        resume=Path(resume);manifest=json.loads((resume/'manifest.json').read_text())
        if manifest['inputs']!=binding or manifest['config_hash']!=object_hash(effective):
            raise ValueError('Resume config/data mismatch')
        for name in ('model.pt','optimizer.pt'):
            if sha256(resume/name)!=manifest['files'][name]:raise ValueError('Resume checkpoint changed')
        saved=torch.load(resume/'optimizer.pt',map_location=device,weights_only=True)
        optimizer.load_state_dict(saved['optimizer']);rng.bit_generator.state=saved['rng']
        trace=saved['trace'];start_step=saved['step']
    with Run(output,stage='Kang_joint_router_training',kind='engineering',config=effective,inputs=inputs,seed=seed) as run:
        for step in range(start_step,t['router_steps']):
            label=labels[step%len(labels)]
            si=torch.tensor(rng.choice(sg[label],t['source_batch'],replace=True),device=device)
            ti=torch.tensor(rng.choice(tg[label],t['target_batch'],replace=True),device=device)
            r=representation(model,z[si],us[si],arm,raw_mean,raw_scale)
            q=model.router.logits(state[si],r,e[si]).softmax(-1)
            task=mixture_energy(points[si],q,y[ti]);native=model.dynamics.last_native_loss
            loss=task+t['native_weight']*native
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; no successful checkpoint emitted')
            gradient=0.
            if step%50==0 and arm in ('gfg_joint_router','gfg_joint_shuffled_U'):
                v=torch.autograd.grad(task,model.dynamics.core.velocity_encoder.net[-1].weight,retain_graph=True)[0]
                gradient=float(v.detach().norm())
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
            row=dict(step=step,task_loss=float(task.detach()),native_loss=float(native.detach()),
                     gfg_task_gradient=gradient,router_entropy=float((-(q*q.clamp_min(1e-12).log()).sum(-1)).mean().detach()))
            trace.append(row)
            if step%10==0:save_json(run.directory/'status.json',dict(status='running',arm=arm,seed=seed,**row))
            if (step+1)%250==0:
                cp=run.directory/'checkpoints'/f'step_{step+1:07d}';cp.mkdir(parents=True)
                save_full_checkpoint(cp/'model.pt',model,metadata=meta)
                with (cp/'optimizer.pt').open('xb') as f:
                    torch.save(dict(optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,trace=trace,step=step+1),f)
                save_json(cp/'manifest.json',dict(inputs=binding,config_hash=object_hash(effective),
                    files={n:sha256(cp/n) for n in ('model.pt','optimizer.pt')}))
        save_full_checkpoint(run.directory/'model.pt',model,metadata=meta)
        save_csv(run.directory/'training_trace.csv',trace)
        result=dict(status='KANG_ROUTER_TRAINED',arm=arm,donor=donor,seed=seed,steps=len(trace),
                    maximum_GFG_task_gradient=max(r['gfg_task_gradient'] for r in trace),fields_frozen=True)
        save_json(run.directory/'summary.json',result)
        (run.directory/'RESULTS.md').write_text('# Kang velocity/router experiment\n\n'+json.dumps(result,indent=2))
    return result


def predict_router(config,donor,seed,arm,geometry,checkpoint,output):
    device=config.get('device','cuda');enforce_gpu_policy(device);torch.set_num_threads(4)
    fold=Path(config['root'])/'folds'/donor
    us,source,sm=read_gene_input(fold/'validation_gene_source.npz',fold/'validation_source.npz')
    data=load_geometry(geometry,'validation',source)
    model,meta=load_full_checkpoint(checkpoint,map_location=device)
    if (meta['arm'],meta['donor'],meta['seed'],meta['transform_hash'])!=(arm,donor,seed,sm['transform_hash']):
        raise ValueError('Kang prediction checkpoint identity mismatch')
    if meta['protocol_hash']!=object_hash(config):raise ValueError('Protocol changed after training')
    if arm=='gfg_joint_shuffled_U':us=training_gene_us(us,source,'gfg_joint_shuffled',seed)
    model.eval();tensor=lambda x:torch.tensor(x,dtype=torch.float32,device=device)
    with Run(output,stage='Kang_source_only_prediction',kind='engineering',
        config=dict(donor=donor,seed=seed,arm=arm,protocol_hash=object_hash(config)),
        inputs=[checkpoint,fold/'validation_gene_source.npz',Path(geometry)/'validation.npz'],seed=seed) as run:
        qs=[]
        with torch.no_grad():
            for start in range(0,len(us),8):
                ix=slice(start,start+8)
                r=representation(model,tensor(source['z'][ix]),tensor(us[ix]),arm,tensor(meta['raw_mean']),tensor(meta['raw_scale']))
                qs.append(model.router.logits(tensor(data['state'][ix]),r,tensor(data['embedding'][ix])).softmax(-1).cpu().numpy())
        q=np.concatenate(qs)
        uniforms=np.random.default_rng(seed).random(len(q))
        modes=np.minimum((uniforms[:,None]>q.cumsum(1)).sum(1),q.shape[1]-1)
        endpoint=data['candidates'][np.arange(len(q)),modes]
        freeze_prediction(run.directory,dict(candidates=data['candidates'],q=q,z=endpoint,
            conditions=source['conditions'],source_ids=source['cell_ids']),config,donor,seed,arm,sm['transform_hash'])


def freeze_prediction(directory,payload,config,donor,seed,arm,transform_hash):
    with (directory/'predictions.npz').open('xb') as f:np.savez_compressed(f,**payload)
    manifest=dict(prediction_sha256=sha256(directory/'predictions.npz'),transform_hash=transform_hash,
        donor=donor,seed=seed,arm=arm,protocol_hash=object_hash(config),task=config['task'],
        future_target_read=False,role='outer_heldout')
    save_json(directory/'prediction_manifest.json',manifest)
    (directory/'RESULTS.md').write_text('# Kang frozen source-only prediction\n\n'+json.dumps(manifest,indent=2))


def predict_simple(config,donor,seed,arm,output,*,geometry=None):
    fold=Path(config['root'])/'folds'/donor
    source,sm=load_pack(fold/'validation_source.npz',expected_side='source')
    inputs=[fold/'validation_source.npz']
    if arm=='static_K1':
        data=load_geometry(geometry,'validation',source);points=data['candidates'][:,0]
        inputs.append(Path(geometry)/'validation.npz')
    elif arm=='identity':points=source['z'].copy()
    elif arm in config['experiments']['simple_baselines']:
        train,_=load_pack(fold/'train_source.npz',expected_side='source')
        target,tm=load_pack(fold/'train_target.npz',expected_side='target')
        if tm['role']!='train':raise ValueError('Resampling baseline must not read heldout target')
        inputs.extend([fold/'train_source.npz',fold/'train_target.npz'])
        points=source['z'].copy();rng=np.random.default_rng(seed)
        for label in np.unique(source['conditions']):
            ct=label.split('|',1)[1];ix=source['conditions']==label
            groups=sorted(c for c in set(train['conditions'])&set(target['conditions']) if c.split('|',1)[1]==ct)
            if not groups:raise ValueError('Missing training cell-type baseline')
            if arm=='training_celltype_mean_shift':
                delta=np.mean([target['z'][target['conditions']==g].mean(0)-train['z'][train['conditions']==g].mean(0) for g in groups],0)
                points[ix]+=delta
            else:
                selected=rng.integers(len(groups),size=int(ix.sum()));rows=[]
                for k in selected:
                    y=target['z'][target['conditions']==groups[k]];rows.append(y[rng.integers(len(y))])
                points[ix]=np.array(rows)
    else:raise ValueError('Unknown baseline')
    with Run(output,stage='Kang_registered_baseline',kind='engineering',config=dict(arm=arm,donor=donor,seed=seed),
             inputs=inputs,seed=seed) as run:
        freeze_prediction(run.directory,dict(candidates=points[:,None],q=np.ones((len(points),1),dtype='float32'),
            z=points,conditions=source['conditions'],source_ids=source['cell_ids']),config,donor,seed,arm,sm['transform_hash'])
