"""Cheap KO-held-out probe of direction in training-defined response programs.

Use the observed 6h source to predict 24h population means. Target-defined
response axes, source normalizers, protein PCA and coefficients use training KOs
only. Validation predictions are frozen before reading validation destinations.
No test cells, artificial cell pairing, full router or biological fate is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from w4_artifacts import Artifacts,sha256
from veloroute.latent import load_pack

ROOT=Path(__file__).resolve().parents[2]


def read_npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def unit(x):
    x=np.asarray(x,dtype='float64')
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def derangement(n,seed):
    if n<2:raise ValueError('At least two KO cohorts are needed for direction shuffling')
    rng=np.random.default_rng(seed)
    for _ in range(10000):
        order=rng.permutation(n)
        if np.all(order!=np.arange(n)):return order
    raise ValueError('Cannot construct direction derangement')


def response_basis(displacement,programs):
    mean=np.asarray(displacement,dtype=float).mean(0)
    _,singular,vt=np.linalg.svd(displacement-mean,full_matrices=False)
    k=min(programs,len(displacement)-1,displacement.shape[1])
    # Deterministic signs; direction agreement itself is invariant to axis signs.
    axes=vt[:k].copy()
    for axis in axes:
        if axis[np.argmax(np.abs(axis))]<0:axis*=-1
    return mean,axes,singular


def ridge(train,test,target,alpha):
    """Same source-only scaling and fixed regularization in every arm."""
    mean=train.mean(0);scale=np.maximum(train.std(0),1e-5)
    x=(train-mean)/scale;z=(test-mean)/scale;ymean=target.mean(0)
    coef=np.linalg.solve(x@x.T+alpha*np.eye(len(x)),target-ymean)
    return ymean+z@x.T@coef,ymean+x@x.T@coef


def cohort_arrays(pack,labels,vectors):
    if set(map(str,pack['conditions']))!=set(labels):raise ValueError('Frozen KO cohort mismatch')
    result=dict(labels=np.array(sorted(labels)),state=[],U=[],velocity=[],embedding=[],cells=[])
    for label in result['labels']:
        mask=pack['conditions']==label
        result['state'].append(pack['z'][mask].astype(float).mean(0))
        result['U'].append(pack['u_features'][mask].astype(float).mean(0))
        result['velocity'].append(pack['velocity'][mask].astype(float).mean(0))
        result['embedding'].append(vectors[str(label)])
        result['cells'].append(int(mask.sum()))
    for k in ('state','U','velocity','embedding'):result[k]=np.stack(result[k])
    result['cells']=np.array(result['cells'])
    return result


def future_means(pack,labels):
    if set(map(str,pack['conditions']))!=set(labels):raise ValueError('Future KO identities mismatch')
    return np.stack([pack['z'][pack['conditions']==label].astype(float).mean(0) for label in labels])


def prepare(configuration):
    """No validation target argument exists on the fit boundary."""
    c=configuration;fold=Path(c['fold']);split=json.loads(Path(c['split']).read_text())
    raw,sm=load_pack(fold/'train_source.npz',expected_side='source')
    future,tm=load_pack(fold/'train_target.npz',expected_side='target')
    heldout,hm=load_pack(fold/'validation_source.npz',expected_side='source')
    if (sm.get('day'),tm.get('day'),hm.get('day'))!=(c['source_hours'],c['target_hours'],c['source_hours']):
        raise ValueError('Source/target biological times differ from the planned early-response task')
    esm=read_npz(c['conditions']);meta=json.loads(str(esm['metadata_json']))
    if not meta.get('frozen') or not meta.get('no_cell_expression_used'):
        raise ValueError('Frozen public protein conditions required')
    vectors={str(k):v.astype(float) for k,v in zip(esm['conditions'],esm['embeddings'])}
    train=cohort_arrays(raw,split['train'],vectors)
    source=cohort_arrays(heldout,split['val'],vectors)
    target=future_means(future,train['labels'])-train['state']
    mean,axes,singular=response_basis(target,c['programs'])
    emean=train['embedding'].mean(0)
    _,_,epc=np.linalg.svd(train['embedding']-emean,full_matrices=False)
    epc=epc[:min(c['protein_dimensions'],len(train['labels'])-1)]
    base=lambda g:np.c_[g['state'],g['U'],(g['embedding']-emean)@epc.T]
    x,xv=base(train),base(source)
    vmean=train['velocity'].mean(0);umean=train['U'].mean(0);smean=train['state'].mean(0)
    def representations(g):
        centered=(g['velocity']-vmean)@axes.T
        magnitude=np.linalg.norm(centered,axis=1,keepdims=True)
        return dict(direction=unit(centered),absolute_direction=unit(g['velocity']@axes.T),
            U_direction=unit((g['U']-umean)@axes.T),
            S_direction=unit((g['state']-smean)@axes.T),
            magnitude=np.pad(magnitude,((0,0),(0,len(axes)-1))))
    return dict(train=train,source=source,target=target,mean=mean,axes=axes,singular=singular,
        x=x,xv=xv,features=representations(train),validation_features=representations(source),
        split=split,source_metadata=sm,target_metadata=tm,validation_metadata=hm,
        source_velocity_reference=vmean,protein_mean=emean,protein_components=epc)


def fit_predictions(p,c):
    x,xv,y=p['x'],p['xv'],p['target'];k=len(p['axes']);rows=[];predictions={}
    features=p['features'];valid=p['validation_features']
    for alpha in c['ridge_alphas']:
        key=str(alpha);a={};train_fits={}
        zeros=np.zeros((len(x),k));vzeros=np.zeros((len(xv),k))
        a['S_U'],train_fits['S_U']=ridge(np.c_[x,zeros],np.c_[xv,vzeros],y,alpha)
        for name,feature in [('velocity_direction','direction'),('absolute_velocity_direction','absolute_direction'),
                             ('U_program_direction','U_direction'),('S_program_direction','S_direction'),
                             ('velocity_magnitude','magnitude')]:
            a[name],_=ridge(np.c_[x,features[feature]],np.c_[xv,valid[feature]],y,alpha)
        for seed in c['shuffle_seeds']:
            t=features['direction'][derangement(len(x),seed)]
            v=valid['direction'][derangement(len(xv),seed+100000)]
            a[f'shuffled_direction_perm{seed}'],_=ridge(np.c_[x,t],np.c_[xv,v],y,alpha)
        # Preserve physical sign: one nonnegative coefficient, never learn a reversal.
        direction_train=features['direction']@p['axes']
        direction_valid=valid['direction']@p['axes']
        residual=y-train_fits['S_U']
        eta=max(0.,float((residual*direction_train).sum()/max((direction_train**2).sum(),1e-12)))
        a['positive_direction_correction']=a['S_U']+eta*direction_valid
        a['shared_response']=np.broadcast_to(p['mean'],(len(xv),len(p['mean']))).copy()
        predictions[key]=a
        rows.append(dict(alpha=alpha,positive_direction_coefficient=eta,
                         estimated_coefficients=(x.shape[1]+k+1)*y.shape[1]))
    return predictions,rows


def evaluate(p,predictions,heldout_target):
    """First held-out future read occurs only after all prediction artifacts freeze."""
    truth=future_means(heldout_target,p['source']['labels'])-p['source']['state']
    actual=(truth-p['mean'])@p['axes'].T
    raw_direction=p['validation_features']['direction']
    cosine=(raw_direction*unit(actual)).sum(1)
    alignment=dict(mean_cosine=float(cosine.mean()),positive_KOs=int((cosine>0).sum()),
        KOs=len(cosine),program_sign_accuracy=float(np.mean(np.sign(raw_direction)==np.sign(actual))))
    rows=[];gains=[]
    for alpha,arms in predictions.items():
        error={};program_error={}
        for name,pred in arms.items():
            mse=((pred-truth)**2).mean(1)
            pmse=(((pred-p['mean'])@p['axes'].T-actual)**2).mean(1)
            signs=(np.sign((pred-p['mean'])@p['axes'].T)==np.sign(actual)).mean(1)
            error[name]=float(mse.mean());program_error[name]=float(pmse.mean())
            for i,label in enumerate(p['source']['labels']):
                rows.append(dict(alpha=float(alpha),arm=name,KO=str(label),MSE=float(mse[i]),
                    program_MSE=float(pmse[i]),program_sign_accuracy=float(signs[i]),
                    source_cells=int(p['source']['cells'][i])))
        baseline=error['S_U'];v=error['velocity_direction']
        shuffled=float(np.mean([error[n] for n in error if n.startswith('shuffled_direction')]))
        gains.append(dict(alpha=float(alpha),S_U_MSE=baseline,velocity_direction_MSE=v,
            gain_vs_S_U=1-v/baseline,gain_vs_mean_shuffle=1-v/shuffled,
            program_gain_vs_S_U=1-program_error['velocity_direction']/program_error['S_U'],
            positive_direction_correction_gain=1-error['positive_direction_correction']/baseline,
            gain_vs_U_program_direction=1-v/error['U_program_direction'],
            U_baseline_gain_vs_shared_response=1-baseline/error['shared_response'],
            direction_gain_vs_shared_response=1-v/error['shared_response']))
    return rows,gains,alignment


def program(config_path,output):
    c=json.loads(Path(config_path).read_text());fold=Path(c['fold'])
    run=Artifacts(output,seed=c['shuffle_seeds'],inputs=[Path(config_path),Path(__file__),
        Path(c['split']),Path(c['conditions']),fold/'train_source.npz',fold/'train_target.npz',
        fold/'validation_source.npz',fold/'transform.npz'])
    try:
        p=prepare(c);predictions,coefficients=fit_predictions(p,c)
        np.savez_compressed(run.directory/'training_programs.npz',axes=p['axes'],response_mean=p['mean'],
            singular_values=p['singular'],source_velocity_reference=p['source_velocity_reference'],
            protein_mean=p['protein_mean'],protein_components=p['protein_components'])
        frozen=[]
        for alpha,arms in predictions.items():
            for arm,pred in arms.items():
                path=run.directory/f'pred_alpha{alpha}_{arm}.npz'
                np.savez_compressed(path,delta=pred,KO=p['source']['labels'])
                frozen.append(dict(alpha=float(alpha),arm=arm,path=str(path.resolve()),sha256=sha256(path)))
        run.json('frozen_predictions.json',dict(predictions=frozen,validation_target_read=False,test_target_read=False))
        # Re-read the frozen prediction bytes so evaluation cannot use changed in-memory values.
        restored={}
        for r in frozen:
            if sha256(r['path'])!=r['sha256']:raise ValueError('Frozen prediction changed')
            saved=read_npz(r['path']);np.testing.assert_array_equal(saved['KO'],p['source']['labels'])
            restored.setdefault(format(r['alpha'],'g'),{})[r['arm']]=saved['delta']
        target,_=load_pack(fold/'validation_target.npz',expected_side='target')
        rows,gains,alignment=evaluate(p,restored,target)
        from w7_router_scaling import write_csv
        write_csv(run.directory/'metrics.csv',rows);write_csv(run.directory/'comparisons.csv',gains)
        write_csv(run.directory/'coefficients.csv',coefficients)
        all_positive=all(g['gain_vs_S_U']>0 and g['gain_vs_mean_shuffle']>0 for g in gains)
        result=dict(status='RESPONSE_PROGRAM_DIRECTION_PROBE_COMPLETE',task='KO_background_conditioned_6to24h_response',
            train_KOs=len(p['train']['labels']),validation_KOs=len(p['source']['labels']),
            test_KOs=len(p['split']['test']),test_targets_read=False,programs=len(p['axes']),
            ridge_alphas=c['ridge_alphas'],comparisons=gains,unlearned_direction_alignment=alignment,
            consistent_descriptive_gain_vs_U_and_shuffle=all_positive,
            scope='exploratory_population_mean_probe_not_full_distribution_or_fate',
            physical_sign_preserved_in_positive_correction=True,alpha_selected_on_validation=False,
            program_variance_explained=float((p['singular'][:len(p['axes'])]**2).sum()/(p['singular']**2).sum()))
        run.json('summary.json',result)
        (run.directory/'RESULTS.md').write_text('# Response program direction probe\n\n'+json.dumps(result,indent=2)+'\n')
        run.json('status.json',dict(status='complete'))
        return result
    except Exception as exc:
        run.json('status.json',dict(status='failed',error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--configuration',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(program(args.configuration,args.output),indent=2))
