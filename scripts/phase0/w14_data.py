"""Fold-local W14 geometry, kinetic/raw-U features, and W5 mean training packs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from w4_artifacts import Artifacts,sha256
from w4_shorttime import read_count_rows,read_metadata
from w4_streams import SourceTransform
from w5_joint_scifate2 import population
from w5_scntseq_repair import regularized_gamma


ROOT=Path(__file__).resolve().parents[2]


class CountReader:
    def __init__(self,path,config):
        self.path=Path(path);self.config=config;self.obs,self.genes=read_metadata(path);self.cache={}
        self.reps=self.obs['rep'].astype(str).to_numpy();self.times=self.obs['timepoint'].astype(str).to_numpy()

    def load(self,rep,day):
        key=(rep,day)
        if key not in self.cache:
            rows=np.flatnonzero((self.reps==rep)&(self.times==f'D{day}'))
            s,u=[read_count_rows(self.path,layer,rows) for layer in ('spliced','unspliced')]
            keep=(np.asarray(s.sum(1)).ravel()>=self.config['minimum_spliced'])&(
                np.asarray(u.sum(1)).ravel()>=self.config['minimum_unspliced'])
            if keep.sum()<self.config['neighbors']:raise ValueError(f'Insufficient source cells in {rep}/D{day}')
            self.cache[key]=dict(s=s[keep],u=u[keep],ids=self.obs.index[rows[keep]].to_numpy(dtype=str))
        return self.cache[key]


def fit_geometry(training_baselines,genes,config):
    """Only caller-provided TRAINING D3 cohorts exist on this fit boundary."""
    s,u=[sparse.vstack([g[k] for g in training_baselines],format='csr') for k in ('s','u')]
    ids=np.concatenate([g['ids'] for g in training_baselines])
    transform=SourceTransform.fit(s,u,cell_ids=ids,allowed_ids=set(ids),gene_names=genes,
        n_hvg=config['n_hvg'],dimensions=config['dimensions'],clusters=config['clusters'])
    sn,un=transform.normalized(s.sum(0),u.sum(0))
    transform.gamma,_=regularized_gamma(sn[0],un[0])
    return transform


def pooled_log_features(components,sn,un):
    """Raw S/U controls have no gamma argument or kinetic dependency."""
    sp=np.log1p(sn)@components.T;up=np.log1p(un)@components.T
    return up,np.c_[sp,up]


def source_features(transform,group,baseline,neighbors):
    s,u=group['s'],group['u'];z=transform.state(s);k=min(neighbors,len(z))
    ids=NearestNeighbors(n_neighbors=k,n_jobs=4).fit(z).kneighbors(z,return_distance=False)
    adjacency=sparse.csr_matrix((np.ones(ids.size),(np.repeat(np.arange(len(z)),k),ids.ravel())),shape=(len(z),len(z)))
    factor=transform.target_sum/np.asarray(adjacency@np.asarray(s.sum(1)).ravel()).reshape(-1,1)
    sn=(adjacency@s[:,transform.genes]).toarray()*factor
    un=(adjacency@u[:,transform.genes]).toarray()*factor
    raw_u,su=pooled_log_features(transform.components,sn,un)
    velocity=transform.projected_rate(sn,un-transform.gamma*sn)-baseline['velocity']
    current=population(transform,s,u)
    return dict(z=z.astype('float32'),velocity=velocity.astype('float32'),direct_U=raw_u.astype('float32'),
        pooled_SU=su.astype('float32'),cell_ids=group['ids'],
        base=np.r_[current['state'],current['state']-baseline['state']],
        population_velocity=current['velocity']-baseline['velocity'])


def prepare_fold(config,heldout,output):
    output=Path(output)
    if (output/'manifest.json').exists():
        saved=json.loads((output/'manifest.json').read_text())
        if saved['config']!=config or saved['heldout']!=heldout:raise ValueError('Existing fold identity differs')
        for row in saved['training_groups']+saved['source_groups']:
            if sha256(row['path'])!=row['sha256']:raise ValueError('Existing fold artifact changed')
        return saved
    data=ROOT/config['data_path'];reader=CountReader(data,config)
    training=[r for r in config['replicates'] if r!=heldout]
    transform=fit_geometry([reader.load(r,3) for r in training],reader.genes,config)
    code=[Path(__file__),ROOT/'scripts/phase0/w4_streams.py',ROOT/'scripts/phase0/w5_scntseq_repair.py']
    run=Artifacts(output,seed=0,inputs=[ROOT/'configs/w14_loro_v1.json']+code)
    transform.save(output/'source_transform.npz')
    training_groups=[];source_groups=[];features=[];means={k:[] for k in ('base','velocity','delta')}
    bank={k:[] for k in means};rng=np.random.default_rng(0)
    mc=json.loads((ROOT/config['mean_config']).read_text())
    for rep in training:
        first=reader.load(rep,3);baseline=population(transform,first['s'],first['u'])
        for day in config['training_days'][:-1]:
            group=reader.load(rep,day);future=reader.load(rep,day+1)
            f=source_features(transform,group,baseline,config['neighbors'])
            target=transform.state(future['s']).astype('float32')
            delta=(target.astype(float).mean(0)-f['z'].astype(float).mean(0))/config['calibration_hours']
            means['base'].append(f['base']);means['velocity'].append(f['population_velocity']);means['delta'].append(delta)
            sh,uh=group['s'][:,transform.genes],group['u'][:,transform.genes]
            depth=np.asarray(group['s'].sum(1)).ravel()
            from w4_kinetic_head import pooled_feature_values
            for _ in range(mc['training_partitions_per_condition']):
                for half in np.array_split(rng.permutation(len(f['z'])),2):
                    mu=f['z'][half].astype(float).mean(0)
                    v=pooled_feature_values(transform,np.asarray(sh[half].sum(0)),np.asarray(uh[half].sum(0)),
                        [depth[half].sum()])['vel_pool'][0]
                    bank['base'].append(np.r_[mu,mu-baseline['state']]);bank['velocity'].append(v-baseline['velocity'])
                    bank['delta'].append((target.astype(float).mean(0)-mu)/config['calibration_hours'])
            path=output/f'train_{rep}_D{day}.npz';np.savez_compressed(path,**f,target=target)
            training_groups.append(dict(rep=rep,source_day=day,target_day=day+1,path=str(path.resolve()),sha256=sha256(path)))
            features.append({k:f[k] for k in ('z','velocity','direct_U','pooled_SU')})
            run.json('status.json',dict(status='preparing_training',heldout=heldout,groups=len(training_groups)))
    np.savez_compressed(output/'mean_training.npz',**{k:np.stack(v) for k,v in means.items()},
        **{'bank_'+k:np.stack(v) for k,v in bank.items()})
    norm={}
    for k in ('z','velocity','direct_U','pooled_SU'):
        values=np.vstack([g[k] for g in features]);norm[k+'_mean']=values.mean(0)
        norm[k+'_scale']=np.maximum(values.std(0),1e-3 if k=='z' else 1e-5)
    np.savez_compressed(output/'normalization.npz',**norm)
    first=reader.load(heldout,3);baseline=population(transform,first['s'],first['u'])
    for day in config['training_days'][:-1]:
        group=reader.load(heldout,day);f=source_features(transform,group,baseline,config['neighbors'])
        choice=np.random.default_rng(config['evaluation_seed']).choice(len(f['z']),
            min(len(f['z']),config['evaluation_source_cells']),replace=False)
        path=output/f'source_{heldout}_D{day}.npz';np.savez_compressed(path,**f,score_indices=choice)
        source_groups.append(dict(rep=heldout,source_day=day,target_day=day+1,path=str(path.resolve()),sha256=sha256(path)))
    manifest=dict(status='FOLD_SOURCE_PACK_READY',config=config,heldout=heldout,training_replicates=training,
        training_groups=training_groups,source_groups=source_groups,transform_sha256=sha256(output/'source_transform.npz'),
        normalization_sha256=sha256(output/'normalization.npz'),mean_training_sha256=sha256(output/'mean_training.npz'),
        source_fit_ids=transform.source_ids.tolist(),geometry_fingerprint=transform.fingerprint(),
        heldout_targets_used_for_fitting=False,r4_read=False)
    run.json('manifest.json',manifest);run.json('status.json',dict(status='complete'))
    return manifest


def load_npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
