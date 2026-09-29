"""Full EM/experts/gate and source-only corruption evaluation for Kang."""
import json
from pathlib import Path

import numpy as np
import torch

from .artifacts import Run,object_hash,save_csv,save_json,sha256
from .full_experiments import condition_token_sets
from .full_model import load_full_checkpoint
from .gfg_experiments import read_gene_input,training_gene_us
from .gpu_policy import enforce_gpu_policy
from .kang_experiments import base_config,freeze_prediction


def variant_config(config,donor,seed,variant):
    result=base_config(config,donor,seed,experts=variant['experts'],full=True)
    result['model'].update(use_adaptive_modes=variant['adaptive'],use_gate=variant['gate'],
                          use_intrinsic=variant['intrinsic'],initial_active=1 if variant['adaptive'] else variant['experts'])
    result['prediction']=dict(force_gate=variant.get('force_gate'))
    result['kang_variant']=variant
    return result


def predict_full(config,donor,seed,variant,checkpoint,output):
    device='cuda';enforce_gpu_policy(device);torch.set_num_threads(4)
    fold=Path(config['root'])/'folds'/donor
    values,source,sm=read_gene_input(fold/'validation_gene_source.npz',fold/'validation_source.npz')
    model,metadata=load_full_checkpoint(checkpoint,map_location=device)
    if metadata['task']!=config['task'] or metadata['transform_hash']!=sm['transform_hash'] or metadata['seed']!=seed:
        raise ValueError('Full Kang checkpoint/fold mismatch')
    if metadata['configuration']['kang_protocol_hash']!=object_hash(config):raise ValueError('Protocol changed')
    if metadata['configuration']['kang_variant']!=variant:raise ValueError('Full ablation identity mismatch')
    values=training_gene_us(values,source,variant['arm'],seed)
    tokens,mask,_=condition_token_sets(fold/'conditions.npz',source['conditions'])
    generator=torch.Generator(device=device).manual_seed(seed)
    results={name:[] for name in ('z','q','g','modes')};calibration=[]
    with Run(output,stage='Kang_full_source_only_prediction',kind='engineering',
        config=dict(donor=donor,seed=seed,variant=variant,protocol_hash=object_hash(config)),
        inputs=[checkpoint,fold/'validation_gene_source.npz',fold/'validation_source.npz'],seed=seed) as run:
        for start in range(0,len(values),8):
            ix=slice(start,start+8)
            z,us,token=[torch.tensor(x[ix],dtype=torch.float32,device=device) for x in (source['z'],values,tokens)]
            padding=torch.tensor(mask[ix],dtype=torch.bool,device=device)
            prediction=model.predict(z,us,token,padding,t0=0.,t1=1.,max_step=.25,generator=generator,
                                     force_gate=variant.get('force_gate'),stochastic=True)
            for k,v in zip(results,(prediction.endpoint,prediction.probabilities,prediction.gate,prediction.modes)):
                results[k].append(v.cpu().numpy())
        if variant['name']=='full_GFG_EM_gate':
            # Within each source donor/type, synthetic corruption only. No target U/S.
            for label in sorted(set(source['conditions'])):
                indices=np.flatnonzero(source['conditions']==label)[:64]
                for start in range(0,len(indices),8):
                    ix=indices[start:start+8]
                    z,us,token=[torch.tensor(x[ix],dtype=torch.float32,device=device) for x in (source['z'],values,tokens)]
                    padding=torch.tensor(mask[ix],dtype=torch.bool,device=device)
                    for method in ('clean','shift','gaussian','zero_u'):
                        corrupted=us if method=='clean' else model.dynamics.corrupt(us,method,generator)
                        with torch.no_grad():
                            context=model.make_context(z,corrupted,token,padding,time_start=0.)
                            g=model.gate_value(context,0.).cpu().numpy()
                        calibration.extend(dict(group=label,method=method,gate=float(p),label=int(method=='clean')) for p in g)
        payload={k:np.concatenate(v) for k,v in results.items()}
        n=len(values)
        freeze_prediction(run.directory,dict(z=payload['z'],candidates=payload['z'][:,None],q=np.ones((n,1),dtype='float32'),
            routing_probabilities=payload['q'],gate=payload['g'],modes=payload['modes'],
            conditions=source['conditions'],source_ids=source['cell_ids']),config,donor,seed,variant['name'],sm['transform_hash'])
        if calibration:save_csv(run.directory/'gate_corruption.csv',calibration)
        save_json(run.directory/'diagnostics.json',dict(gate_mean=float(payload['g'].mean()),
            active_modes=int(model.active_modes.sum()),sampled_mode_usage=np.bincount(payload['modes'],minlength=variant['experts']).tolist(),
            latent_time_is_counterfactual=True,sampled_distribution_not_exact_noise_marginal=True))
