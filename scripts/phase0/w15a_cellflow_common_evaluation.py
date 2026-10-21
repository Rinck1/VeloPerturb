"""New common-ID inference/evaluation; does not alter the active CF-2 grid.

Uses W13's frozen source IDs and exact target cache, not a new random sample.
Point endpoints (L2) and weighted candidate endpoints (L1/L3) share one scorer.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from w4_artifacts import Artifacts,sha256
from w15a_cellflow_adapter import (array_hash,add_cellflow_paths,condition_inputs,
    predict_cellflow_tau,time_schedule)
from w15a_cellflow_train import create_solver,load_npz,load_normalizer,read_pack


WINDOWS=((5,10),(10,15),(15,20),(5,15),(10,20),(5,20))
CLOCKS=('decay','linear')
KEYS={f'{a:02d}_{b:02d}_{clock}':(a,b,clock) for a,b in WINDOWS for clock in CLOCKS}
DEFAULT_REFERENCE=Path(__file__).resolve().parents[2]/'outputs/w13_scifate2_direct_continuation_v1'


def align_source_ids(expected_ids,actual_ids):
    expected=np.asarray(expected_ids,dtype=str);actual=np.asarray(actual_ids,dtype=str)
    if len(set(actual))!=len(actual) or len(set(expected))!=len(expected):
        raise ValueError('Duplicate source IDs')
    lookup={identity:i for i,identity in enumerate(actual)}
    if not set(expected)<=set(lookup):raise ValueError('Required common source ID is missing')
    return np.array([lookup[identity] for identity in expected],dtype=int)


def reference_sources(reference_root=DEFAULT_REFERENCE):
    """Read source packs only. Target paths/hashes are recorded but not opened."""
    root=Path(reference_root);study_path=root/'study_manifest.json';samples_path=root/'source_samples.json'
    study=json.loads(study_path.read_text());samples=json.loads(samples_path.read_text())
    pack=Path(study['pack']);pm_path=pack/'manifest.json'
    if sha256(pm_path)!=study['pack_manifest_sha256']:raise ValueError('W13 source pack manifest changed')
    pm=json.loads(pm_path.read_text());sources={}
    for start in (5,10,15):
        matches=[r for r in pm['prediction_groups'] if int(r['source_hour'])==start]
        if len(matches)!=1:raise ValueError('W13 requires exactly one source pack per origin')
        record=matches[0]
        if sha256(record['path'])!=record['sha256']:raise ValueError('W13 frozen source pack changed')
        data=load_npz(record['path']);indices=np.asarray(samples[f'{start}_to_{start+5}'],dtype=int)
        if len(set(indices))!=len(indices) or indices.min()<0 or indices.max()>=len(data['z']):
            raise ValueError('Invalid W13 source sample indices')
        ids=np.asarray(data['cell_ids'],dtype=str)[indices]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate W13 source IDs')
        sources[start]=dict(cell_ids=ids,z=data['z'][indices],all_cell_ids=np.asarray(data['cell_ids'],dtype=str),
            source_pack_sha256=record['sha256'],
            source_ids_sha256=array_hash(ids),indices=indices)
    targets={int(r['target_hour']):dict(r) for r in study['evaluation_caches']}
    if set(targets)!={10,15,20} or len(study['evaluation_caches'])!=3:
        raise ValueError('Expected exact W13 target cache grid')
    return dict(sources=sources,target_caches=targets,
        hashes=dict(study_manifest=sha256(study_path),source_samples=sha256(samples_path),
                    source_pack_manifest=sha256(pm_path)),reference_root=str(root.resolve()))


def save_common_predictions(output,forecasts,*,reference_root,arm,seed,
                            model_family='CellFlow_L2',extra=None):
    """Persist source-only model outputs, with exact twelve-window coverage.

    forecasts[key] contains source_ids and x[N,D] or x[N,K,D], optionally q[N,K].
    q is a source-conditional mixing probability, not target-derived membership.
    """
    reference=reference_sources(reference_root)
    if set(forecasts)!=set(KEYS):raise ValueError('All twelve common window keys are required')
    run=Artifacts(output,seed=seed,inputs=[Path(__file__),Path(reference_root)/'study_manifest.json',
                                       Path(reference_root)/'source_samples.json'])
    arrays={};records=[]
    for key,(start,stop,clock) in KEYS.items():
        f=forecasts[key];ids=np.asarray(f['source_ids'],dtype=str);expected=reference['sources'][start]['cell_ids']
        if not np.array_equal(ids,expected):raise ValueError('Prediction/source IDs differ from W13 order')
        x=np.asarray(f['x'])
        if x.ndim not in (2,3) or len(x)!=len(ids) or x.shape[-1]!=reference['sources'][start]['z'].shape[-1]:
            raise ValueError('Candidate endpoint/source shape mismatch')
        if not np.isfinite(x).all():raise ValueError('Nonfinite candidate endpoint')
        arrays[key]=x;arrays[key+'_source_ids']=ids
        if x.ndim==3:
            q=np.asarray(f.get('q'))
            if q.shape!=x.shape[:2] or not np.isfinite(q).all() or (q<0).any() or not np.allclose(q.sum(1),1.,atol=1e-6):
                raise ValueError('Invalid source-conditional candidate mixture')
            arrays[key+'_q']=q
        records.append(dict(key=key,source_hours=start,target_hours=stop,kernel=clock,source_cells=len(ids),
            source_ids_sha256=array_hash(ids),representation='weighted_candidates' if x.ndim==3 else 'points'))
    np.savez_compressed(run.directory/'predictions.npz',**arrays)
    result=dict(status='COMMON_SOURCE_PREDICTIONS_FROZEN',arm=arm,seed=int(seed),model_family=model_family,
        windows=records,predictions_sha256=sha256(run.directory/'predictions.npz'),
        reference_root=reference['reference_root'],reference_hashes=reference['hashes'],
        target_read=False,source_side_context='full_source_tokens_before_terminal_common_ID_subsample',extra=extra or {})
    run.json('prediction_manifest.json',result);run.json('status.json',dict(status='complete'))
    return result


def predict_common(training,output,reference_root=DEFAULT_REFERENCE):
    """Re-predict existing L2 weights on common IDs; never retrain or re-fit token stats."""
    add_cellflow_paths()
    from flax import serialization
    training=Path(training);tm=json.loads((training/'summary.json').read_text())
    if sha256(training/'model.msgpack')!=tm['model_sha256']:raise ValueError('CellFlow checkpoint changed')
    directory=Path(tm['pack_directory']);pm=json.loads((directory/'manifest.json').read_text())
    norm=load_normalizer(directory);reference=reference_sources(reference_root);solver=None;forecasts={}
    for key,(start,stop,clock) in KEYS.items():
        matches=[r for r in pm['development_sources'] if int(r['time_hours'])==start]
        paths={(r['path'],r['sha256'],r['source_key']) for r in matches}
        if len(paths)!=1:raise ValueError('Ambiguous source cohort for the common origin')
        path,digest,source_key=paths.pop()
        if sha256(path)!=digest:raise ValueError('CellFlow source pack changed')
        pack=read_pack(path,source_key);common=reference['sources'][start]
        if set(pack['source_ids'])!=set(common['all_cell_ids']):
            raise ValueError('Tokens must use the same complete source cohort, not a sampled or augmented one')
        indices=align_source_ids(common['cell_ids'],pack['source_ids'])
        np.testing.assert_allclose(pack['z'][indices],common['z'],rtol=1e-6,atol=2e-5)
        clock_spec=time_schedule(stop-start,kernel=clock)
        # Compute κ/group context on all observed source cells BEFORE selecting IDs.
        cond=condition_inputs(pack,norm,tm['arm'],time_hours=float(start),
            condition_delta_hours=clock_spec['condition_delta_hours'],
            cell_permutation=np.random.default_rng(0).permutation(len(pack['z'])))
        if solver is None:
            solver=create_solver(tm['config'],{k:v[:1] for k,v in cond.items()},pack['z'].shape[1],tm['seed'])
            solver.vf_state_inference=serialization.from_bytes(solver.vf_state_inference,
                (training/'model.msgpack').read_bytes())
        # Use exactly W13's x0, not a numerically reprojected alternative.
        x=predict_cellflow_tau(solver,common['z'],{k:v[indices] for k,v in cond.items()},
            tau_end=clock_spec['tau_end'],per_particle=True)
        forecasts[key]=dict(x=x,source_ids=common['cell_ids'])
    return save_common_predictions(output,forecasts,reference_root=reference_root,arm=tm['arm'],seed=tm['seed'],
        extra=dict(training=str(training.resolve()),model_sha256=tm['model_sha256'],
            pack_directory=str(directory.resolve()),source_transform_sha256=pm['source_transform_sha256']))


def freeze_common_grid(prediction_directories,*,reference_root=DEFAULT_REFERENCE,
                       expected_seeds=(0,1,2),expected_arms=None):
    """Check whole arm×seed×12-window grid before opening any target NPZ."""
    reference=reference_sources(reference_root);rows=[];identities=set()
    for directory in prediction_directories:
        directory=Path(directory);manifest_path=directory/'prediction_manifest.json'
        m=json.loads(manifest_path.read_text());identity=(m['arm'],int(m['seed']))
        if identity in identities:raise ValueError('Duplicate arm/seed prediction')
        identities.add(identity)
        if m['status']!='COMMON_SOURCE_PREDICTIONS_FROZEN' or m['target_read']:
            raise ValueError('Only frozen source-only predictions can enter common evaluation')
        if m['reference_hashes']!=reference['hashes']:raise ValueError('Common reference hashes differ')
        path=directory/'predictions.npz'
        if sha256(path)!=m['predictions_sha256']:raise ValueError('Frozen predictions changed')
        if len(m['windows'])!=12 or {r['key'] for r in m['windows']}!=set(KEYS):
            raise ValueError('Every seed requires the same complete twelve windows')
        with np.load(path,allow_pickle=False) as f:
            for record in m['windows']:
                key=record['key'];start,stop,clock=KEYS[key]
                if (record['source_hours'],record['target_hours'],record['kernel'])!=(start,stop,clock):
                    raise ValueError('Window metadata does not match its fixed key')
                ids=f[key+'_source_ids'];expected=reference['sources'][start]['cell_ids']
                if not np.array_equal(ids,expected) or array_hash(ids)!=record['source_ids_sha256']:
                    raise ValueError('Every seed must have identical common source IDs in order')
                if len(f[key])!=len(ids):raise ValueError('Prediction/source size changed')
                x=f[key]
                if x.ndim not in (2,3) or x.shape[-1]!=reference['sources'][start]['z'].shape[-1] or not np.isfinite(x).all():
                    raise ValueError('Invalid candidate endpoint dimensions or values')
                if x.ndim==3:
                    q=f[key+'_q']
                    if q.shape!=x.shape[:2] or not np.isfinite(q).all() or (q<0).any() or not np.allclose(q.sum(1),1.,atol=1e-6):
                        raise ValueError('Invalid candidate probabilities before target read')
        rows.append(dict(path=str(path.resolve()),manifest=str(manifest_path.resolve()),
            sha256=m['predictions_sha256'],manifest_sha256=sha256(manifest_path),
            arm=m['arm'],seed=int(m['seed']),model_family=m['model_family'],windows=m['windows']))
    arms=set(expected_arms) if expected_arms is not None else {r['arm'] for r in rows}
    required={(arm,int(seed)) for arm in arms for seed in expected_seeds}
    if not arms or identities!=required:raise ValueError('Incomplete expected arm by seed grid')
    return reference,rows


def weighted_energy(x,w,y,block=256):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float);w=np.asarray(w,dtype=float);w=w/w.sum()
    cross=0.;xx=0.;yy=0.
    for i in range(0,len(x),block):
        a=x[i:i+block];wa=w[i:i+block]
        cross+=float(wa@cdist(a,y).mean(1))
        for j in range(0,len(x),block):xx+=float(wa@cdist(a,x[j:j+block])@w[j:j+block])
    for i in range(0,len(y),block):yy+=float(cdist(y[i:i+block],y).sum())/len(y)**2
    return 2*cross-xx-yy


def evaluate_common(prediction_directories,output,*,reference_root=DEFAULT_REFERENCE,
                    expected_seeds=(0,1,2),expected_arms=None,mode_centers=None):
    reference,frozen=freeze_common_grid(prediction_directories,reference_root=reference_root,
        expected_seeds=expected_seeds,expected_arms=expected_arms)
    # Unique index, not same-named manifests passed to Artifacts (three-seed safe).
    inputs=[Path(__file__),Path(reference_root)/'study_manifest.json',Path(reference_root)/'source_samples.json']
    if mode_centers is not None:inputs.append(Path(mode_centers))
    run=Artifacts(output,seed=list(expected_seeds),inputs=inputs)
    run.json('frozen_grid.json',dict(reference_hashes=reference['hashes'],predictions=frozen,
        all_expected_seeds_and_twelve_windows_complete=True,target_caches_opened=False))
    centers=np.load(mode_centers,allow_pickle=False) if mode_centers is not None else None
    targets={}
    # First semantic read of a target happens AFTER immutable whole-grid index.
    for stop,cache in reference['target_caches'].items():
        if sha256(cache['path'])!=cache['sha256']:raise ValueError('W13 target cache changed')
        with np.load(cache['path'],allow_pickle=False) as f:
            targets[stop]=f['target'].copy()
    rows=[];saved={r['path']:load_npz(r['path']) for r in frozen}
    for arm in sorted({r['arm'] for r in frozen}):
        records=[r for r in frozen if r['arm']==arm]
        for key,(start,stop,clock) in KEYS.items():
            parts=[];weights=[]
            for record in records:
                p=saved[record['path']];x=np.asarray(p[key],dtype=float)
                if x.ndim==3:
                    q=np.asarray(p[key+'_q'],dtype=float)
                    if q.shape!=x.shape[:2] or (q<0).any() or not np.allclose(q.sum(1),1.,atol=1e-6):
                        raise ValueError('Invalid candidate probabilities')
                    q=q/q.sum(1,keepdims=True)
                    w=q.reshape(-1)/len(q);x=x.reshape(-1,x.shape[-1])
                else:w=np.full(len(x),1/len(x))
                parts.append(x);weights.append(w/len(records))
            x=np.concatenate(parts);w=np.concatenate(weights);y=targets[stop]
            mean=w@x;target_mean=y.mean(0)
            row=dict(arm=arm,key=key,source_hours=start,target_hours=stop,kernel=clock,seeds=len(records),
                source_cells_per_seed=len(reference['sources'][start]['cell_ids']),target_cells=len(y),
                energy=weighted_energy(x,w,y),sampled_prediction_mean_mse=float(np.mean((mean-target_mean)**2)),
                target_mean_semantics='same_frozen_512_target_empirical_mean_not_full_population_mean')
            if centers is not None:
                px=np.bincount(cdist(x,centers).argmin(1),weights=w,minlength=len(centers))
                py=np.bincount(cdist(y,centers).argmin(1),minlength=len(centers))/len(y)
                row['mode_proportion_TV']=float(.5*np.abs(px-py).sum())
            rows.append(row)
    with (run.directory/'metrics.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(status='COMMON_SAMPLE_DEVELOPMENT_EVALUATED',expected_seeds=list(expected_seeds),
        arms=sorted({r['arm'] for r in frozen}),windows_per_arm=12,
        reference_hashes=reference['hashes'],target_caches=reference['target_caches'],metrics=rows,
        no_retraining=True,Traxler_read=False,old_grid_unchanged=True,
        comparison_scope='common_empirical_distribution_only_not_virgin_confirmation')
    run.json('summary.json',result);run.json('status.json',dict(status='complete'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['predict','evaluate'])
    p.add_argument('--reference',default=str(DEFAULT_REFERENCE));p.add_argument('--training')
    p.add_argument('--predictions',nargs='+');p.add_argument('--output',required=True)
    p.add_argument('--expected-seeds',default='0,1,2');p.add_argument('--expected-arms',default=None)
    p.add_argument('--mode-centers',default=None);args=p.parse_args()
    result=(predict_common(args.training,args.output,args.reference) if args.action=='predict' else
        evaluate_common(args.predictions,args.output,reference_root=args.reference,
            expected_seeds=tuple(int(x) for x in args.expected_seeds.split(',')),
            expected_arms=None if args.expected_arms is None else args.expected_arms.split(','),mode_centers=args.mode_centers))
    print(json.dumps(result,indent=2))
