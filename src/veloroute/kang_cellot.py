"""Historical CellOT adapter; new training is disabled by the reuse-only policy."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .artifacts import Run,object_hash,save_csv,save_json,sha256
from .gpu_policy import enforce_gpu_policy
from .kang_experiments import freeze_prediction
from .latent import load_pack


def official_modules(root):
    source=Path(root)/'external/cellot'
    sys.path.insert(0,str(source))
    # Import the reviewed objective file without its unrelated AE/data loader.
    spec=importlib.util.spec_from_file_location('veloroute_official_cellot',source/'cellot/models/cellot.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,source


def run_cellot(config,extended,donor,seed,output,*,smoke_steps=None):
    # Check before data access, GPU allocation, imports of official code, or output creation.
    # Synthetic smoke must not bypass the user's no-retraining instruction either.
    policy=extended.get('cellot',{})
    if policy.get('execution')!='train' or policy.get('training_enabled') is not True:
        raise RuntimeError('CellOT training disabled: reuse existing reproduction results; '
                           'new training requires explicit user authorization and a revised policy.')
    device=config.get('device','cuda');enforce_gpu_policy(device);torch.set_num_threads(4)
    module,official=official_modules(config['root'])
    fold=Path(config['root'])/'folds'/donor
    x,xm=load_pack(fold/'train_source.npz',expected_side='source')
    y,ym=load_pack(fold/'train_target.npz',expected_side='target')
    if xm['role']!='train' or ym['role']!='train':raise ValueError('CellOT fit must be training-only')
    cfg=extended['cellot'];torch.manual_seed(seed);rng=np.random.default_rng(seed)
    recipe=load_recipe=dict(model=dict(name='cellot',hidden_units=cfg['hidden_units'],softplus_W_kernels=False,
        g=dict(fnorm_penalty=cfg['g_fnorm_penalty']),kernel_init_fxn=dict(name='uniform',b=.1)),
        optim=dict(optimizer='Adam',lr=cfg['learning_rate'],beta1=cfg['beta1'],beta2=cfg['beta2'],weight_decay=0))
    f,g=module.load_networks(recipe,input_dim=x['z'].shape[1]);f,g=f.to(device),g.to(device)
    opts=module.load_opts(recipe,f,g)
    sx,ty=[torch.tensor(a['z'],dtype=torch.float32,device=device) for a in (x,y)]
    labels=sorted(set(x['conditions'])&set(y['conditions']))
    sg={c:np.flatnonzero(x['conditions']==c) for c in labels};tg={c:np.flatnonzero(y['conditions']==c) for c in labels}
    count=smoke_steps or cfg['iterations'];trace=[]
    inputs=[fold/'train_source.npz',fold/'train_target.npz',official/'cellot/models/cellot.py',official/'cellot/networks/icnns.py']
    with Run(output,stage='Kang_CellOT_official_network_and_objective',kind='engineering',
        config=dict(recipe=cfg,donor=donor,seed=seed,smoke_steps=smoke_steps,protocol_hash=object_hash(config)),
        inputs=inputs,seed=seed) as run:
        for step in range(count):
            label=labels[step%len(labels)]
            target=ty[torch.tensor(rng.choice(tg[label],cfg['batch_size']),device=device)]
            for _ in range(cfg['inner_iterations']):
                source=sx[torch.tensor(rng.choice(sg[label],cfg['batch_size']),device=device)].detach().requires_grad_(True)
                opts.g.zero_grad();gl=module.compute_loss_g(f,g,source).mean()+g.penalize_w();gl.backward();opts.g.step()
            source=sx[torch.tensor(rng.choice(sg[label],cfg['batch_size']),device=device)].detach().requires_grad_(True)
            opts.f.zero_grad();fl=module.compute_loss_f(f,g,source,target).mean();fl.backward();opts.f.step();f.clamp_w()
            if not torch.isfinite(gl+fl):raise RuntimeError('CellOT numerical failure; no successful result')
            if step%50==0:
                row=dict(step=step,f_loss=float(fl.detach()),g_loss=float(gl.detach()));trace.append(row)
                save_json(run.directory/'status.json',dict(status='running',**row))
            if (step+1)%10000==0:
                path=run.directory/'checkpoints'/f'step_{step+1:07d}.pt';path.parent.mkdir(exist_ok=True)
                with path.open('xb') as stream:torch.save(dict(f=f.state_dict(),g=g.state_dict(),
                    optimizer_f=opts.f.state_dict(),optimizer_g=opts.g.state_dict(),rng=rng.bit_generator.state,step=step+1),stream)
        with (run.directory/'model.pt').open('xb') as stream:torch.save(dict(f=f.state_dict(),g=g.state_dict(),recipe=recipe),stream)
        save_csv(run.directory/'training_trace.csv',trace)
        # A separate prediction sub-run; no held-out target is ever loaded.
        source,sm=load_pack(fold/'validation_source.npz',expected_side='source')
        points=[];g.eval()
        for start in range(0,len(source['z']),256):
            a=torch.tensor(source['z'][start:start+256],device=device).requires_grad_(True)
            points.append(g.transport(a).detach().cpu().numpy())
        points=np.concatenate(points)
        with Run(run.directory/'predict',stage='Kang_CellOT_source_only_prediction',kind='engineering',
            config=dict(donor=donor,seed=seed),inputs=[run.directory/'model.pt',fold/'validation_source.npz'],seed=seed) as prediction:
            freeze_prediction(prediction.directory,dict(z=points,candidates=points[:,None],q=np.ones((len(points),1),dtype='float32'),
                conditions=source['conditions'],source_ids=source['cell_ids']),config,donor,seed,'CellOT_official',sm['transform_hash'])
        result=dict(status='KANG_CELLOT_COMPLETE',iterations=count,seed=seed,donor=donor,official_network_and_loss=True,
            paper_identical_preprocessing=False,input_space='training_fold_PCA',early_stopping=False,heldout_target_read=False)
        save_json(run.directory/'summary.json',result)
        (run.directory/'RESULTS.md').write_text('# CellOT Kang adaptation\n\n'+json.dumps(result,indent=2))
    return result
