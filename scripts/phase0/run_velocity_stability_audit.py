#!/usr/bin/env python3
"""Source-only GFG stability audit on the frozen RENGE day4 gene source.

The prepared gene source is library-normalized, not raw UMI counts. Therefore
this script does not claim a true count-split test: it reports U dropout,
local pre-GFG U permutation, and cell bootstrap as measurement stress tests.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import torch
from veloroute.contracts import FitScope
from veloroute.full_model import FullConfig
from veloroute.gfg import GFGDynamics
from veloroute.gfg_experiments import read_gene_input
from veloroute.latent import FrozenSplicingTransform
from veloroute.probes import permute_local
import anndata as ad

def sha256(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def cosine(a,b):
    return np.sum(a*b,1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)

def metric(ref, alt, label):
    nr=np.linalg.norm(ref,axis=1); na=np.linalg.norm(alt,axis=1)
    c=cosine(ref,alt)
    return dict(perturbation=label,cells=len(ref),cosine_mean=float(c.mean()),cosine_median=float(np.median(c)),
                cosine_q10=float(np.quantile(c,.1)),cosine_q90=float(np.quantile(c,.9)),
                norm_ratio_median=float(np.median(na/np.maximum(nr,1e-12))),
                relative_rms=float(np.sqrt(np.mean((alt-ref)**2))/max(np.sqrt(np.mean(ref**2)),1e-12)),
                norm_rank_corr=float(np.corrcoef(np.argsort(np.argsort(nr)),np.argsort(np.argsort(na)))[0,1]))

def infer(dyn, z, values, device):
    out=[]
    with torch.no_grad():
        for i in range(0,len(values),64):
            _,v,_=dyn(torch.as_tensor(z[i:i+64],dtype=torch.float32,device=device),torch.as_tensor(values[i:i+64],dtype=torch.float32,device=device))
            out.append(v.cpu().numpy())
    return np.concatenate(out)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--device',default='cuda'); p.add_argument('--seed',type=int,default=20260923); p.add_argument('--max-cells',type=int,default=1200); a=p.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    device=torch.device(a.device); rng=np.random.default_rng(a.seed)
    fold=Path('outputs/veloroute_real_pipeline_20260912_v2/fold'); gene=Path('outputs/veloroute_gfg_inputs_20260914/train_gene_source.npz')
    values, source, sm=read_gene_input(gene,fold/'train_source.npz'); transform=FrozenSplicingTransform.load(fold/'transform.npz')
    sel=np.arange(len(values)) if len(values)<=a.max_cells else np.sort(rng.choice(len(values),a.max_cells,replace=False))
    values=values[sel]; z=source['z'][sel]; ids=source['cell_ids'][sel]; source={k:v[sel] if hasattr(v,'__len__') and len(v)==len(source['z']) else v for k,v in source.items()}
    cfg=FullConfig(dynamics_backend='gfg',gfg_genes=values.shape[1]//2,state_dim=50,representation_dim=64,joint_dynamics=False)
    dyn=GFGDynamics(cfg).to(device).eval(); ck='/data/yuchang/GFG/results/mousebrain_graphbatch_directed_v1_seed0/final.pth'; dyn.core.load_pretrained(ck)
    dyn.prepare(torch.as_tensor(values,dtype=torch.float32,device=device),torch.as_tensor(transform.components,dtype=torch.float32,device=device),cell_ids=ids.tolist(),scope=FitScope(frozenset(ids.tolist())))
    ref=infer(dyn,z,values,device); g=values.shape[1]//2; rows=[]
    # True count split from the processed RENGE h5ad. The split is performed
    # before library normalization and uses the same frozen PCA gene order.
    h5=Path('data/renge/processed_release_v1/day4/day4.h5ad')
    raw_split_status='complete'
    try:
        adata=ad.read_h5ad(h5,backed='r')
        raw_ix=adata.obs_names.get_indexer(ids)
        selected_gene_ids=np.asarray(transform.gene_ids)[np.asarray(transform.selected)]
        gene_ix=adata.var_names.get_indexer(selected_gene_ids)
        if (raw_ix<0).any() or (gene_ix<0).any(): raise ValueError('raw barcode/gene alignment failed')
        sraw=adata.layers['spliced'][raw_ix][:,gene_ix]; uraw=adata.layers['unspliced'][raw_ix][:,gene_ix]
        sraw=sraw.toarray() if hasattr(sraw,'toarray') else np.asarray(sraw); uraw=uraw.toarray() if hasattr(uraw,'toarray') else np.asarray(uraw)
        sraw=np.rint(sraw).astype('int64'); uraw=np.rint(uraw).astype('int64');
        s1=rng.binomial(sraw,0.5); u1=rng.binomial(uraw,0.5); s2=sraw-s1; u2=uraw-u1
        split_vectors=[]
        for ss,uu in ((s1,u1),(s2,u2)):
            depth=np.maximum(ss.sum(1),1); factor=10000.0/depth[:,None]
            split=np.concatenate((uu*factor,ss*factor),1).astype('float32')
            split_vectors.append(infer(dyn,z,split,device))
        n0=np.linalg.norm(split_vectors[0],axis=1); n1=np.linalg.norm(split_vectors[1],axis=1)
        csplit=cosine(split_vectors[0],split_vectors[1])
        rows.append(dict(perturbation='true_raw_UMI_count_split',cells=len(csplit),
            cosine_mean=float(csplit.mean()),cosine_median=float(np.median(csplit)),
            cosine_q10=float(np.quantile(csplit,.1)),cosine_q90=float(np.quantile(csplit,.9)),
            norm_ratio_median=float(np.median(n1/np.maximum(n0,1e-12))),
            relative_rms=float(np.sqrt(np.mean((split_vectors[1]-split_vectors[0])**2))/max(np.sqrt(np.mean(split_vectors[0]**2)),1e-12)),
            norm_rank_corr=float(np.corrcoef(np.argsort(np.argsort(n0)),np.argsort(np.argsort(n1)))[0,1]),
            comparison='true_count_split',fraction=.5))
        adata.file.close()
    except Exception as exc:
        raw_split_status=f'blocked:{type(exc).__name__}:{exc}'
    # The prepared matrix is normalized U/S; these are explicit stress tests,
    # not raw-UMI Poisson thinning.
    for frac in (.25,.5,.75):
        x=values.copy(); mask=rng.random((len(x),g))>frac; x[:,:g][mask]=0
        rows.append({**metric(ref,infer(dyn,z,x,device),f'U_dropout_{frac}'),'comparison':'real_dropout','fraction':frac})
    for seed in range(3):
        x=values.copy(); x[:,:g]=permute_local(values[:,:g],source,seed=seed,neighbors=10)[0]
        rows.append({**metric(ref,infer(dyn,z,x,device),f'local_preGFG_U_shuffle_{seed}'),'comparison':'shuffled_U','fraction':1.0})
    for b in range(5):
        ix=rng.integers(len(values),size=len(values)); vb=infer(dyn,z[ix],values[ix],device)
        rows.append({**metric(ref[ix],vb,f'cell_bootstrap_{b}'),'comparison':'cell_bootstrap','fraction':1.0})
    with open(out/'metrics.csv','w',newline='') as f:
        fields=sorted({k for r in rows for k in r}); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary={'status':'SOURCE_ONLY_GFG_STABILITY_COMPLETE','dataset':'RENGE_train_day4','cells':len(values),'genes':g,'target_read':False,'heldout_target_read':False,'checkpoint':ck,'checkpoint_sha256':sha256(ck),'normalization':'library_normalized_gene_US','true_raw_UMI_count_split':raw_split_status,'interpretation':'measurement_stress_test_only; no task utility claim','limitations':['U dropout is not Poisson count thinning','count-split uses processed integer S/U layers and frozen full-source normalization statistics','single frozen GFG checkpoint','cell bootstrap is resampling, not replicate sequencing','local shuffled-U tests cell correspondence, not all velocity nulls']}
    json.dump(summary,open(out/'summary.json','w'),indent=2); (out/'RESULTS.md').write_text('# GFG source-only stability audit\n\n'+json.dumps(summary,indent=2)+'\n')
    json.dump({'status':'complete'},open(out/'status.json','w'))

if __name__=='__main__': main()
