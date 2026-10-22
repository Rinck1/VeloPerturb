"""Technical source-count sensitivity, no future data and no model training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from w4_artifacts import Artifacts


def integer_counts(value):
    value=np.asarray(value)
    if not np.isfinite(value).all() or (value<0).any() or np.any(np.abs(value-np.rint(value))>1e-8):
        raise ValueError('Binomial splitting requires nonnegative integer counts')
    return value.astype('int64')


def split_counts(s,u,depth,seed):
    s,u,depth=map(integer_counts,(s,u,depth));remainder=depth-s.sum(1)
    if (remainder<0).any():raise ValueError('Selected counts exceed total S depth')
    rng=np.random.default_rng(seed)
    sa=rng.binomial(s,.5);ua=rng.binomial(u,.5)
    da=sa.sum(1)+rng.binomial(remainder,.5);db=depth-da
    if (da<=0).any() or (db<=0).any():raise ValueError('Zero half-depth; report without inventing counts')
    return (sa,ua,da),(s-sa,u-ua,db)


def normalized(s,u,depth,target_sum):
    factor=target_sum/np.asarray(depth,dtype='float64')[:,None]
    return np.asarray(s,dtype='float64')*factor,np.asarray(u,dtype='float64')*factor


def latent(s,u,transform,norm):
    z=(np.log1p(s)-transform.mean)@transform.components.T
    U=np.log1p(u)@transform.components.T-transform.u_mean
    vg=np.where(transform.supported,(u-s*transform.gamma)/(1+s),0.)
    velocity=(vg@transform.components.T)/transform.velocity_scale
    return dict(S_state50=z,U50=U,velocity_arrow50=velocity,
        velocity_program5=(velocity-norm['velocity_reference'])@norm['axes'].T,
        U_program5=(U-norm['U_reference'])@norm['axes'].T)


def cosine(a,b):
    a,b=map(lambda x:np.asarray(x,dtype='float64'),(a,b))
    denominator=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    if (denominator<=1e-15).any():raise ValueError('Zero norm direction is undefined')
    return np.clip(np.sum(a*b,axis=1)/denominator,-1,1)


def read_selected(source,transform,raw_directory):
    import h5py
    from w4_shorttime import read_count_rows
    ids=np.asarray(source['cell_ids'],dtype=str);n=len(ids);ng=len(transform.selected)
    s=np.zeros((n,ng));u=np.zeros_like(s);depth=np.zeros(n)
    for library in sorted({x.split(':',1)[0] for x in ids}):
        path=Path(raw_directory)/(library+'.h5ad')
        with h5py.File(path,'r') as f:
            names=f['obs']['_index'].asstr()[:];genes=f['var']['_index'].asstr()[:]
        if tuple(genes)!=transform.gene_ids:raise ValueError('Gene order differs from frozen transform')
        lookup={x:i for i,x in enumerate(names)};positions=np.flatnonzero(np.char.startswith(ids,library+':'))
        rows=np.array([lookup[ids[i]] for i in positions]);order=np.argsort(rows);rows=rows[order];positions=positions[order]
        sm=read_count_rows(path,'spliced',rows);um=read_count_rows(path,'unspliced',rows)
        s[positions]=sm[:,transform.selected].toarray();u[positions]=um[:,transform.selected].toarray()
        depth[positions]=np.asarray(sm.sum(1)).ravel()
    np.testing.assert_allclose(depth,source['depth'],rtol=0,atol=1e-8)
    return integer_counts(s),integer_counts(u),integer_counts(depth)


def program(configuration,output):
    from veloroute.latent import load_pack,FrozenSplicingTransform
    from w7_router_scaling import write_csv
    from scipy.stats import spearmanr
    c=json.loads(Path(configuration).read_text());fold=Path(c['fold'])
    source,metadata=load_pack(fold/'validation_source.npz',expected_side='source')
    if metadata['day']!=6:raise ValueError('Expected 6h source cells only')
    transform=FrozenSplicingTransform.load(fold/'transform.npz')
    with np.load(c['programs'],allow_pickle=False) as f:norm={key:f[key] for key in f.files}
    libraries=sorted({x.split(':',1)[0] for x in source['cell_ids']})
    root=Path(__file__).resolve().parents[2]
    inputs=[Path(configuration),Path(__file__),root/'PLANS/w15q_velocity_countsplit_20261022.md',
        Path(__file__).with_name('w4_shorttime.py'),Path(__file__).with_name('w4_artifacts.py'),
        root/'src/veloroute/latent.py',fold/'validation_source.npz',fold/'transform.npz',Path(c['programs'])]
    inputs.extend(Path(c['raw_directory'])/(library+'.h5ad') for library in libraries)
    run=Artifacts(output,seed=c['seeds'],inputs=inputs)
    try:
        s,u,depth=read_selected(source,transform,c['raw_directory']);sn,un=normalized(s,u,depth,transform.target_sum)
        original=latent(sn,un,transform,norm);reproduction={}
        for key,pack_key in [('S_state50','z'),('U50','u_features'),('velocity_arrow50','velocity')]:
            np.testing.assert_allclose(original[key],source[pack_key],rtol=1e-5,atol=2e-5)
            reproduction[key]=float(np.max(np.abs(original[key]-source[pack_key])))
        rows=[];summaries=[];cell_cosines={}
        for seed in c['seeds']:
            a,b=split_counts(s,u,depth,seed)
            sa,ua=normalized(*a,transform.target_sum);sb,ub=normalized(*b,transform.target_sum)
            U_a=normalized(s,a[1]*2,depth,transform.target_sum)[1]
            U_b=normalized(s,b[1]*2,depth,transform.target_sum)[1]
            for regime,first,second in [('both_split',(sa,ua),(sb,ub)),
                ('U_only_split',(sn,U_a),(sn,U_b)),('S_only_split',(sa,un),(sb,un))]:
                la,lb=latent(*first,transform,norm),latent(*second,transform,norm)
                for name in c['modalities']:
                    value=cosine(la[name],lb[name]);cell_cosines.setdefault((regime,name),[]).append(value)
                    for i,v in enumerate(value):
                        rows.append(dict(seed=seed,regime=regime,modality=name,cell_id=source['cell_ids'][i],
                            KO=source['conditions'][i],S_depth=int(depth[i]),cosine=float(v),squared_cosine=float(v*v)))
                    for ko in sorted(set(source['conditions'])):
                        selected=value[source['conditions']==ko]
                        summaries.append(dict(seed=seed,regime=regime,modality=name,KO=ko,cells=len(selected),
                            mean_cosine=float(selected.mean()),median_cosine=float(np.median(selected)),
                            mean_squared_cosine=float(np.mean(selected**2)),negative_fraction=float(np.mean(selected<0))))
        write_csv(run.directory/'cell_reliability.csv',rows);write_csv(run.directory/'KO_reliability.csv',summaries)
        aggregate=[]
        for (regime,name),values in cell_cosines.items():
            mean=np.mean(values,axis=0);coef=float(spearmanr(depth,mean).statistic) if np.std(mean)>1e-10 else None
            selected=[r for r in summaries if r['regime']==regime and r['modality']==name]
            aggregate.append(dict(regime=regime,modality=name,cells=len(mean),
                mean_cosine_equal_KO=float(np.mean([r['mean_cosine'] for r in selected])),
                mean_squared_cosine_equal_KO=float(np.mean([r['mean_squared_cosine'] for r in selected])),
                negative_fraction_equal_KO=float(np.mean([r['negative_fraction'] for r in selected])),
                depth_spearman=coef))
        write_csv(run.directory/'aggregate_reliability.csv',aggregate)
        result=dict(status='VELOCITY_SOURCE_COUNT_SPLIT_COMPLETE',source_cells=len(s),KOs=len(set(source['conditions'])),
            seeds=c['seeds'],regimes=c['regimes'],raw_count_libraries=libraries,reproduction_max_abs=reproduction,
            technical_half_depth_sensitivity_only=True,biological_accuracy_claim=False,full_depth_uncertainty_bound=False,
            future_targets_read=False,GFG=False,JVP=False,aggregate=aggregate)
        run.json('summary.json',result)
        (run.directory/'RESULTS.md').write_text('# Source-count split sensitivity\n\n'+json.dumps(result,indent=2)+'\n')
        run.json('status.json',dict(status='complete'))
        return result
    except Exception as exc:
        run.json('status.json',dict(status='failed',error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--configuration',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(program(args.configuration,args.output),indent=2),flush=True)
