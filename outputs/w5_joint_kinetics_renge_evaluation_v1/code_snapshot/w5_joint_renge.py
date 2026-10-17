"""Three-version task-joint development run; existing held-out TF split retained."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from w4_artifacts import Artifacts,sha256
from w4_kinetic_head import pooled_feature_values
from w4_shorttime import CONTROL_LABELS,condition_embeddings,qc_mask,read_count_rows,read_metadata
from w4_streams import SourceTransform,group_mean,pooled_counts
from w5_joint_kinetics import fit_task_model,predict_delta


ROOT = Path(__file__).resolve().parents[2]
CODE = [Path(__file__).with_name(n) for n in ('w5_joint_renge.py','w5_joint_kinetics.py',
    'w4_kinetic_head.py','w4_streams.py','w4_shorttime.py','w4_arms.py','w4_artifacts.py')]


def features(transform,s,u,groups,labels):
    state = group_mean(transform.state(s),groups,labels)
    ps,pu = pooled_counts(s,groups,labels),pooled_counts(u,groups,labels)
    sn,un = transform.normalized(ps,pu)
    return state,transform.projected_rate(sn,un-transform.gamma*sn)


def source_bank(transform,s,u,groups,labels,embeddings,targets,baseline,*,partitions,seed):
    rng = np.random.default_rng(seed)
    z = transform.state(s)
    sh,uh = s[:,transform.genes],u[:,transform.genes]
    depth = np.asarray(s.sum(1)).ravel()
    rows = []
    for i,label in enumerate(labels):
        population = np.flatnonzero(groups == label)
        if len(population) < 2:raise ValueError('Two source cells per training group required')
        for _ in range(partitions):
            for half in np.array_split(rng.permutation(population),2):
                state = z[half].mean(0)
                velocity = pooled_feature_values(transform,np.asarray(sh[half].sum(0)),
                    np.asarray(uh[half].sum(0)),[depth[half].sum()])['vel_pool'][0]-baseline
                rows.append((np.r_[state,embeddings[i]],velocity,targets[i]-state))
    return {key:np.stack([row[i] for row in rows]) for i,key in enumerate(('base','velocity','delta'))}


def prepare(config_path,output):
    config = json.loads(Path(config_path).read_text())
    if config['router_enabled'] or config['gfg_jvp_enabled'] or config['primary_replication_changed']:
        raise ValueError('This is the no-JVP task-head development extension')
    old_root = ROOT/config['baseline_predictions']
    old = json.loads((old_root/'prediction_manifest.json').read_text())
    if sha256(old_root/'predictions.npz') != old['prediction_sha256']:raise ValueError('Baseline changed')
    source,target = Path(old['source_path']),Path(old['target_path'])
    transform_path = old_root/'transform_seed0.npz'
    if sha256(transform_path) != old['transform_sha256']['0']:raise ValueError('Geometry changed')
    embeddings_path = ROOT/config['embeddings']
    run = Artifacts(output,seed=config['seeds'],inputs=[Path(config_path),old_root/'prediction_manifest.json',
        old_root/'predictions.npz',source,target,transform_path,embeddings_path]+CODE)
    transform = SourceTransform.load(transform_path)
    obs,genes = read_metadata(source);future,fg = read_metadata(target)
    np.testing.assert_array_equal(genes,fg);np.testing.assert_array_equal(genes,transform.gene_names)
    labels = sorted(old['training_TFs']+old['controls']);development = old['heldout_TFs']
    sl,tl = obs['condition'].astype(str).to_numpy(),future['condition'].astype(str).to_numpy()
    si = np.flatnonzero(qc_mask(obs)&np.isin(sl,labels))
    vi = np.flatnonzero(qc_mask(obs)&np.isin(sl,development))
    ti = np.flatnonzero(qc_mask(future)&np.isin(tl,labels))
    if set(labels)&set(development) or set(tl[ti]) != set(labels):raise ValueError('Training group contract failed')
    np.testing.assert_array_equal(np.array([f'day4:{i}' for i in obs.index[si]]),transform.source_ids)
    s,u = [read_count_rows(source,key,si) for key in ('spliced','unspliced')]
    sv,uv = [read_count_rows(source,key,vi) for key in ('spliced','unspliced')]
    state,v = features(transform,s,u,sl[si],labels)
    test_state,test_v = features(transform,sv,uv,sl[vi],development)
    controls = np.isin(labels,CONTROL_LABELS)
    baseline = v[controls].mean(0)
    v,test_v = v-baseline,test_v-baseline
    train_target = group_mean(transform.state(read_count_rows(target,'spliced',ti)),tl[ti],labels)
    et,ev = condition_embeddings(embeddings_path,labels),condition_embeddings(embeddings_path,development)
    bank = source_bank(transform,s,u,sl[si],labels,et,train_target,baseline,
        partitions=config['training_partitions_per_condition'],seed=0)
    np.savez_compressed(run.directory/'training_pack.npz',base=np.c_[state,et],velocity=v,delta=train_target-state,
        controls=controls,**{f'bank_{k}':value for k,value in bank.items()})
    with np.load(old_root/'predictions.npz',allow_pickle=False) as saved:
        original_static,original_analytic = saved['static'].copy(),saved['vel_pool'].copy()
    np.savez_compressed(run.directory/'source_pack.npz',base=np.c_[test_state,ev],velocity=test_v,state=test_state,
        labels=np.array(development),original_static=original_static,original_analytic=original_analytic)
    run.json('manifest.json',dict(status='JOINT_DEVELOPMENT_PACK_READY',config=config,
        target_path=str(target),transform_path=str(transform_path),heldout_TFs=development,
        training_TFs=old['training_TFs'],training_groups=len(labels),augmented_observations=len(bank['base']),
        heldout_target_expression_read=False,source_velocity_centering='mean_of_source_controls',
        training_pack_sha256=sha256(run.directory/'training_pack.npz'),
        source_pack_sha256=sha256(run.directory/'source_pack.npz'),transform_sha256=sha256(transform_path)))
    run.json('status.json',dict(status='complete'))
    print(json.dumps(dict(status='JOINT_PACK_READY',training_groups=len(labels),source_perturbations=len(development))),flush=True)


def train(pack,output,seed,device):
    pack = Path(pack);manifest = json.loads((pack/'manifest.json').read_text());config = manifest['config']
    if seed not in config['seeds']:raise ValueError('Unregistered seed')
    if device != 'cpu':
        from veloroute.gpu_policy import enforce_gpu_policy
        enforce_gpu_policy(device)
    for filename,key in (('training_pack.npz','training_pack_sha256'),('source_pack.npz','source_pack_sha256')):
        if sha256(pack/filename) != manifest[key]:raise ValueError('Prepared data changed')
    run = Artifacts(output,seed=seed,inputs=[pack/'manifest.json',pack/'training_pack.npz',pack/'source_pack.npz']+CODE)
    with np.load(pack/'training_pack.npz',allow_pickle=False) as saved:a = {k:saved[k] for k in saved.files}
    with np.load(pack/'source_pack.npz',allow_pickle=False) as saved:b = {k:saved[k] for k in saved.files}
    bank = {k:a[f'bank_{k}'] for k in ('base','velocity','delta')}
    predictions,histories = {},{}
    for arm in config['arms']:
        model,history = fit_task_model(a['base'],a['velocity'],a['delta'],bank,config,
            controls=a['controls'],arm=arm,seed=seed,device=device)
        predictions[arm] = b['state']+predict_delta(model,b['base'],b['velocity'])
        torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},arm=arm,config=config,
            base_dimensions=model.base_dimensions,selection=model.selection),run.directory/f'{arm}.pt')
        histories[arm] = history
        run.json('status.json',dict(status='training',completed_arm=arm,seed=seed))
        print(json.dumps(dict(status='ARM_TRAINED',arm=arm,seed=seed,head_task_gradient=history['head_task_gradient_max'])),flush=True)
    np.savez_compressed(run.directory/'predictions.npz',labels=b['labels'],**predictions)
    run.json('training.json',histories)
    run.json('prediction_manifest.json',dict(status='PREDICTIONS_READY',seed=seed,pack=str(pack.resolve()),
        heldout_target_expression_read=False,prediction_sha256=sha256(run.directory/'predictions.npz')))
    run.json('status.json',dict(status='complete'))


def evaluate(pack,training_root,output,wait):
    pack,training_root = Path(pack),Path(training_root)
    manifest = json.loads((pack/'manifest.json').read_text());config = manifest['config']
    deadline = time.monotonic()+7200
    paths = [training_root/f'seed{seed}' for seed in config['seeds']]
    while not all((p/'prediction_manifest.json').is_file() for p in paths):
        if not wait or time.monotonic() > deadline:raise TimeoutError('Training predictions incomplete')
        time.sleep(15)
    predictions = {}
    for path in paths:
        pm = json.loads((path/'prediction_manifest.json').read_text())
        if pm['heldout_target_expression_read'] or sha256(path/'predictions.npz') != pm['prediction_sha256']:
            raise ValueError('Prediction integrity failed')
        with np.load(path/'predictions.npz',allow_pickle=False) as saved:
            for arm in config['arms']:predictions.setdefault(arm,[]).append(saved[arm].copy())
    run = Artifacts(output,seed=config['seeds'],inputs=[pack/'manifest.json',pack/'source_pack.npz',
        Path(manifest['target_path']),Path(manifest['transform_path'])]+[p/'predictions.npz' for p in paths]+CODE)
    with np.load(pack/'source_pack.npz',allow_pickle=False) as saved:
        labels = saved['labels'].tolist()
        predictions['original_static'] = list(saved['original_static'])
        predictions['original_analytic'] = list(saved['original_analytic'])
    transform = SourceTransform.load(manifest['transform_path'])
    obs,_ = read_metadata(manifest['target_path']);groups = obs['condition'].astype(str).to_numpy()
    rows = np.flatnonzero(qc_mask(obs)&np.isin(groups,labels))
    truth = group_mean(transform.state(read_count_rows(manifest['target_path'],'spliced',rows)),groups[rows],labels)
    errors,records = {},[]
    for arm,values in predictions.items():
        values = np.stack(values);error = np.square(values.mean(0)-truth).mean(1);errors[arm] = error
        for label,mse in zip(labels,error):records.append(dict(arm=arm,TF=label,mse=float(mse)))
    result = dict(status='TASK_JOINT_DEVELOPMENT_COMPLETE',development=True,
        mse={arm:float(e.mean()) for arm,e in errors.items()},
        gain_over_original_static={arm:float(1-e.mean()/errors['original_static'].mean()) for arm,e in errors.items()},
        joint_gain_over_analytic_task=float(1-errors['velocity_joint'].mean()/errors['analytic_task'].mean()),
        joint_gain_over_static_task=float(1-errors['velocity_joint'].mean()/errors['static_task'].mean()),
        seeds=config['seeds'],TFs=labels,router_enabled=False,gfg_jvp_enabled=False,
        task_supervision_in_velocity_head=True,original_results_changed=False)
    with (run.directory/'metrics.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['arm','TF','mse']);writer.writeheader();writer.writerows(records)
    np.savez_compressed(run.directory/'predictions_and_truth.npz',truth=truth,labels=np.array(labels),
        **{arm:np.stack(values) for arm,values in predictions.items()})
    run.json('summary.json',result)
    (run.directory/'RESULTS.md').write_text('# Task-joint kinetic expansion\n\n'+json.dumps(result,indent=2)+'\n')
    run.json('status.json',dict(status='complete'));print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();sub = parser.add_subparsers(dest='stage',required=True)
    p = sub.add_parser('prepare');p.add_argument('--config',default='configs/w5_joint_kinetics_renge_v1.json');p.add_argument('--output',required=True)
    t = sub.add_parser('train');t.add_argument('--pack',required=True);t.add_argument('--output',required=True)
    t.add_argument('--seed',type=int,required=True);t.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    e = sub.add_parser('evaluate');e.add_argument('--pack',required=True);e.add_argument('--training-root',required=True)
    e.add_argument('--output',required=True);e.add_argument('--wait',action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':prepare(args.config,args.output)
    elif args.stage == 'train':train(args.pack,args.output,args.seed,args.device)
    else:evaluate(args.pack,args.training_root,args.output,args.wait)
