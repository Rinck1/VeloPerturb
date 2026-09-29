"""Native official scDFM gene/PAD model, training-only graph, common PCA output."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse,stats
import torch

from .artifacts import Run,object_hash,save_csv,save_json,sha256
from .gfg_experiments import read_gene_input
from .gpu_policy import enforce_gpu_policy
from .kang_experiments import freeze_prediction
from .latent import FrozenSplicingTransform,load_pack


def official_definitions(path,names):
    """Load reviewed standalone definitions unchanged, without unrelated CLI side effects."""
    tree=ast.parse(Path(path).read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('Official helper source changed')
    namespace=dict(np=np,pd=pd,torch=torch,sparse=sparse,stats=stats)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def official_model(root):
    path=Path(root)/'external/scDFM/src/models/origin/model.py'
    name='veloroute_official_scdfm_origin'
    spec=importlib.util.spec_from_file_location(name,path,submodule_search_locations=[str(path.parent)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module.model,path


def run_scdfm(config,extended,donor,seed,output,*,smoke_steps=None):
    device=config.get('device','cuda');enforce_gpu_policy(device);torch.set_num_threads(4)
    root=Path(config['root']);fold=root/'folds'/donor;cfg=extended['scdfm']
    us,source,sm=read_gene_input(fold/'train_gene_source.npz',fold/'train_source.npz')
    target,tm=load_pack(fold/'train_target.npz',expected_side='target')
    transform=FrozenSplicingTransform.load(fold/'transform.npz');genes=len(transform.selected)
    with np.load(fold/'train_target_genes.npz',allow_pickle=False) as f:
        if not np.array_equal(f['cell_ids'],target['cell_ids']):raise ValueError('Training gene target ID mismatch')
        sy=f['gene_logspliced'].copy()
    sx=np.log1p(us[:,genes:])  # S only. U is never sent to this static baseline.
    Model,model_path=official_model(root)
    graph_path=root/'external/scDFM/src/utils/utils.py';loss_path=root/'external/scDFM/src/script/run.py'
    helpers=official_definitions(graph_path,['preprocess_expression','safe_correlation_matrix','soft_threshold_weights',
        'sparsify_topk','sparsify_threshold','adjacency_to_mha_mask','build_gene_coexpression_graph'])
    losses=official_definitions(loss_path,['pairwise_sq_dists','median_sigmas','mmd2_unbiased_multi_sigma'])
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    inputs=[fold/'train_gene_source.npz',fold/'train_target_genes.npz',fold/'transform.npz',model_path,graph_path,loss_path]
    with Run(output,stage='Kang_scDFM_native_gene_adaptation',kind='engineering',
        config=dict(recipe=cfg,donor=donor,seed=seed,smoke_steps=smoke_steps,protocol_hash=object_hash(config)),inputs=inputs,seed=seed) as run:
        graph=helpers['build_gene_coexpression_graph'](np.concatenate((sx,sy)),method='pearson',k=cfg['graph_topk'],
            use_negative_edge=cfg['graph_negative_edges'],log1p=False)
        mask_path=run.directory/'training_graph.pt'
        with mask_path.open('xb') as f:torch.save(graph,f)
        model=Model(ntoken=genes,d_model=32 if smoke_steps else cfg['d_model'],nhead=8,
            nlayers=1 if smoke_steps else cfg['layers'],fusion_method='differential_perceiver',
            perturbation_function='cytokine',use_perturbation_interaction=True,mask_path=str(mask_path)).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        steps=smoke_steps or cfg['iterations'];scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=1e-6)
        x,y=[torch.tensor(a,dtype=torch.float32,device=device) for a in (sx,sy)]
        labels=sorted(set(source['conditions'])&set(target['conditions']))
        sg={c:np.flatnonzero(source['conditions']==c) for c in labels};tg={c:np.flatnonzero(target['conditions']==c) for c in labels}
        trace=[];batch=4 if smoke_steps else cfg['batch_size']
        for step in range(steps):
            group=labels[step%len(labels)]
            si=torch.tensor(rng.choice(sg[group],batch),device=device);ti=torch.tensor(rng.choice(tg[group],batch),device=device)
            selected=torch.randperm(genes,device=device)[:min(genes,cfg['gene_batch'])]
            xs,yt=x[si][:,selected],y[ti][:,selected];noise=torch.randn_like(xs);time=torch.rand(batch,device=device)
            xt=(1-time[:,None])*noise+time[:,None]*yt
            ids=selected[None].expand(batch,-1);pert=torch.zeros((batch,1),dtype=torch.long,device=device)
            velocity=model(ids,xt,time,xs,pert,ids,mode='predict_y')
            fm=(velocity-(yt-noise)).square().mean()
            sigmas=losses['median_sigmas'](yt,scales=(.5,1.,2.,4.))
            mmd=losses['mmd2_unbiased_multi_sigma'](xt+velocity*(1-time[:,None]),yt,sigmas)
            loss=fm+cfg['mmd_weight']*mmd
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite official scDFM loss')
            optimizer.zero_grad();loss.backward();optimizer.step();scheduler.step()
            if step%10==0:
                row=dict(step=step,flow_loss=float(fm.detach()),mmd_loss=float(mmd.detach()));trace.append(row)
                save_json(run.directory/'status.json',dict(status='running',**row))
            if (step+1)%2000==0:
                p=run.directory/'checkpoints'/f'step_{step+1:07d}.pt';p.parent.mkdir(exist_ok=True)
                with p.open('xb') as f:torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),step=step+1),f)
        with (run.directory/'model.pt').open('xb') as f:torch.save(dict(model=model.state_dict(),recipe=cfg,genes=genes),f)
        save_csv(run.directory/'training_trace.csv',trace)
        # Source-only deployment, constant IFN token, never a stimulated-cell input.
        vus,validation,vm=read_gene_input(fold/'validation_gene_source.npz',fold/'validation_source.npz')
        vs=np.log1p(vus[:,genes:]);gene_pred=np.empty_like(vs);model.eval()
        with torch.no_grad():
            for start in range(0,len(vs),cfg['inference_batch']):
                sx=torch.tensor(vs[start:start+cfg['inference_batch']],device=device);n=len(sx)
                for gene_start in range(0,genes,cfg['gene_batch']):
                    selected=torch.arange(gene_start,min(genes,gene_start+cfg['gene_batch']),device=device)
                    ids=selected[None].expand(n,-1);condition=torch.zeros((n,1),dtype=torch.long,device=device)
                    source_gene=sx[:,selected];z=torch.randn_like(source_gene);h=cfg['inference_rk4_step']
                    for step in range(round(1/h)):
                        t=step*h
                        def field(value,time):return model(ids,value,torch.full((n,),time,device=device),source_gene,condition,ids)
                        a=field(z,t);b=field(z+h*a/2,t+h/2);cc=field(z+h*b/2,t+h/2);d=field(z+h*cc,t+h)
                        z=z+h*(a+2*b+2*cc+d)/6
                    if not torch.isfinite(z).all():raise RuntimeError('scDFM inference diverged')
                    gene_pred[start:start+n,gene_start:gene_start+len(selected)]=z.cpu().numpy()
        z=((gene_pred-transform.mean)@transform.components.T).astype('float32')
        with Run(run.directory/'predict',stage='Kang_scDFM_source_only_prediction',kind='engineering',
            config=dict(seed=seed,donor=donor),inputs=[run.directory/'model.pt',fold/'validation_gene_source.npz'],seed=seed) as pred:
            freeze_prediction(pred.directory,dict(z=z,candidates=z[:,None],q=np.ones((len(z),1),dtype='float32'),
                gene_logspliced=gene_pred,conditions=validation['conditions'],source_ids=validation['cell_ids']),
                config,donor,seed,'scDFM_official',vm['transform_hash'])
        result=dict(status='KANG_SCDFM_COMPLETE',steps=steps,donor=donor,seed=seed,native_gene_architecture=True,
            source_U_used=False,graph_fit='training_source_target_S_only',common_PCA_evaluation=True,
            paper_identical_recipe=False,actual_deployment_parameters=sum(p.numel() for p in model.parameters()))
        save_json(run.directory/'summary.json',result)
        (run.directory/'RESULTS.md').write_text('# scDFM native Kang adaptation\n\n'+json.dumps(result,indent=2))
    return result
