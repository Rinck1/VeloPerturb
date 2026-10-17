"""Train on three daily series; predict the independent five-hour r4 series."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import torch

from w4_artifacts import Artifacts,sha256
from w4_kinetic_head import pooled_feature_values
from w4_shorttime import read_count_rows,read_metadata
from w4_streams import SourceTransform
from w5_joint_kinetics import fit_task_model,predict_delta
from w5_scntseq_repair import effective_minutes,regularized_gamma


ROOT = Path(__file__).resolve().parents[2]
CODE = [Path(__file__).with_name(n) for n in ('w5_joint_scifate2.py','w5_joint_kinetics.py',
    'w4_kinetic_head.py','w4_streams.py','w4_shorttime.py','w4_arms.py','w4_artifacts.py','w5_scntseq_repair.py')]


def population(transform,s,u):
    sn,un = transform.normalized(s.sum(0),u.sum(0))
    return dict(state=transform.state(s).mean(0),
        velocity=transform.projected_rate(sn,un-transform.gamma*sn)[0])


def prepare(config_path,output):
    config = json.loads(Path(config_path).read_text());data = ROOT/config['data_path']
    if config['gfg_jvp_enabled'] or config['router_enabled'] or sha256(data) != config['data_sha256']:
        raise ValueError('Use the fixed no-JVP input configuration')
    run = Artifacts(output,seed=config['seeds'],inputs=[Path(config_path),data]+CODE)
    obs,genes = read_metadata(data)
    cache,counts = {},[]
    def load(rep,tp,*,target_only=False):
        key = (rep,tp)
        if key in cache:return cache[key]
        rows = np.flatnonzero((obs['rep'].astype(str).to_numpy() == rep)&(obs['timepoint'].astype(str).to_numpy() == tp))
        s,u = [read_count_rows(data,layer,rows) for layer in ('spliced','unspliced')]
        ds,du = np.asarray(s.sum(1)).ravel(),np.asarray(u.sum(1)).ravel()
        keep = (ds >= config['minimum_spliced'])&(du >= config['minimum_unspliced'])
        if keep.sum() < 50:raise ValueError('Insufficient cells for the selected source/target group')
        counts.append(dict(rep=rep,timepoint=tp,cells_before=len(rows),cells=int(keep.sum())))
        result = (s[keep],u[keep],obs.index[rows[keep]].to_numpy(dtype=str))
        cache[key] = result
        return result
    baselines = [load(rep,'D3') for rep in config['training_replicates']]
    s,u = [sparse.vstack([p[i] for p in baselines],format='csr') for i in (0,1)]
    ids = np.concatenate([p[2] for p in baselines])
    transform = SourceTransform.fit(s,u,cell_ids=ids,allowed_ids=set(ids),gene_names=genes,
        n_hvg=config['n_hvg'],dimensions=config['dimensions'],clusters=config['clusters'])
    sn,un = transform.normalized(s.sum(0),u.sum(0))
    transform.gamma,_ = regularized_gamma(sn[0],un[0])
    transform.save(run.directory/'source_transform.npz')
    base,velocity,delta,bank = [],[],[],dict(base=[],velocity=[],delta=[])
    generator = np.random.default_rng(0)
    for rep in config['training_replicates']:
        first_s,first_u,_ = load(rep,'D3');baseline = population(transform,first_s,first_u)
        for day in config['training_days'][:-1]:
            s,u,_ = load(rep,f'D{day}')
            future_s,_,_ = load(rep,f'D{day+1}',target_only=True)
            z = transform.state(s);state = z.mean(0);target = transform.state(future_s).mean(0)
            current = population(transform,s,u)
            base.append(np.r_[state,state-baseline['state']])
            velocity.append(current['velocity']-baseline['velocity'])
            delta.append((target-state)/config['calibration_hours'])
            sh,uh = s[:,transform.genes],u[:,transform.genes];depth = np.asarray(s.sum(1)).ravel()
            for _ in range(config['training_partitions_per_condition']):
                for half in np.array_split(generator.permutation(s.shape[0]),2):
                    mu = z[half].mean(0)
                    v = pooled_feature_values(transform,np.asarray(sh[half].sum(0)),
                        np.asarray(uh[half].sum(0)),[depth[half].sum()])['vel_pool'][0]
                    bank['base'].append(np.r_[mu,mu-baseline['state']])
                    bank['velocity'].append(v-baseline['velocity'])
                    bank['delta'].append((target-mu)/config['calibration_hours'])
    np.savez_compressed(run.directory/'training_pack.npz',base=np.stack(base),velocity=np.stack(velocity),
        delta=np.stack(delta),controls=np.zeros(len(base),dtype=bool),
        **{f'bank_{k}':np.stack(v) for k,v in bank.items()})
    rep = config['prediction_replicate'];bs,bu,_ = load(rep,'05h');baseline = population(transform,bs,bu)
    states,test_base,test_v = [],[],[]
    for start,stop in config['prediction_intervals_hours']:
        s,u,_ = load(rep,f'{start:02d}h');current = population(transform,s,u)
        states.append(current['state']);test_base.append(np.r_[current['state'],current['state']-baseline['state']])
        test_v.append(current['velocity']-baseline['velocity'])
    intervals = np.array(config['prediction_intervals_hours'])
    np.savez_compressed(run.directory/'source_pack.npz',state=np.stack(states),base=np.stack(test_base),
        velocity=np.stack(test_v),intervals=intervals,
        duration=np.array([effective_minutes(b-a,config['calibration_hours'],config['time_kernel']) for a,b in intervals]))
    with (run.directory/'technical_cohorts.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(counts[0]));writer.writeheader();writer.writerows(counts)
    run.json('manifest.json',dict(status='SCIFATE2_JOINT_PACK_READY',config=config,data_path=str(data),
        transform_path=str(run.directory/'source_transform.npz'),
        training_transitions=len(base),augmented_training_observations=len(bank['base']),
        r4_used_for_fitting=False,per_origin_future_target_used=False,
        prediction_source_hours=[5,10,15],unread_last_target_hours=20,
        task='daily_training_to_independent_five_hour_series_transfer',
        training_pack_sha256=sha256(run.directory/'training_pack.npz'),
        source_pack_sha256=sha256(run.directory/'source_pack.npz')))
    run.json('status.json',dict(status='complete'))
    print(json.dumps(dict(status='SCIFATE2_PACK_READY',training_transitions=len(base))),flush=True)


def train(pack,output,seed,device):
    pack = Path(pack);manifest = json.loads((pack/'manifest.json').read_text());config = manifest['config']
    if seed not in config['seeds']:raise ValueError('Unknown seed')
    if device != 'cpu':
        from veloroute.gpu_policy import enforce_gpu_policy
        enforce_gpu_policy(device)
    run = Artifacts(output,seed=seed,inputs=[pack/'manifest.json',pack/'training_pack.npz',pack/'source_pack.npz']+CODE)
    with np.load(pack/'training_pack.npz',allow_pickle=False) as f:a = {k:f[k] for k in f.files}
    with np.load(pack/'source_pack.npz',allow_pickle=False) as f:b = {k:f[k] for k in f.files}
    predictions,histories = {},{}
    for arm in config['arms']:
        model,history = fit_task_model(a['base'],a['velocity'],a['delta'],
            {k:a[f'bank_{k}'] for k in ('base','velocity','delta')},config,
            controls=a['controls'],arm=arm,seed=seed,device=device)
        predictions[arm] = b['state']+b['duration'][:,None]*predict_delta(model,b['base'],b['velocity'])
        torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},config=config,arm=arm),run.directory/f'{arm}.pt')
        histories[arm] = history
        print(json.dumps(dict(status='ARM_TRAINED',arm=arm,seed=seed,head_task_gradient=history['head_task_gradient_max'])),flush=True)
    np.savez_compressed(run.directory/'predictions.npz',intervals=b['intervals'],**predictions)
    run.json('training.json',histories)
    run.json('prediction_manifest.json',dict(status='PREDICTIONS_READY',seed=seed,
        prediction_sha256=sha256(run.directory/'predictions.npz'),r4_used_for_fitting=False))
    run.json('status.json',dict(status='complete'))


def evaluate(pack,training_root,output,wait):
    pack,training_root = Path(pack),Path(training_root)
    manifest = json.loads((pack/'manifest.json').read_text());config = manifest['config']
    paths = [training_root/f'seed{seed}' for seed in config['seeds']];deadline = time.monotonic()+7200
    while not all((p/'prediction_manifest.json').is_file() for p in paths):
        if not wait or time.monotonic() > deadline:raise TimeoutError('Predictions incomplete')
        time.sleep(15)
    predictions = {arm:[] for arm in config['arms']}
    for path in paths:
        pm = json.loads((path/'prediction_manifest.json').read_text())
        if pm['r4_used_for_fitting'] or sha256(path/'predictions.npz') != pm['prediction_sha256']:
            raise ValueError('Invalid prediction artifact')
        with np.load(path/'predictions.npz',allow_pickle=False) as f:
            for arm in predictions:predictions[arm].append(f[arm].copy())
    data = Path(manifest['data_path']);transform_path = Path(manifest['transform_path'])
    run = Artifacts(output,seed=config['seeds'],inputs=[pack/'manifest.json',pack/'source_pack.npz',data,transform_path]
        +[p/'predictions.npz' for p in paths]+CODE)
    transform = SourceTransform.load(transform_path);obs,_ = read_metadata(data)
    truth,records = [],[]
    for start,stop in config['prediction_intervals_hours']:
        rows = np.flatnonzero((obs['rep'].astype(str).to_numpy() == config['prediction_replicate'])
            &(obs['timepoint'].astype(str).to_numpy() == f'{stop:02d}h'))
        s,u = [read_count_rows(data,key,rows) for key in ('spliced','unspliced')]
        ds,du = np.asarray(s.sum(1)).ravel(),np.asarray(u.sum(1)).ravel()
        keep = (ds >= config['minimum_spliced'])&(du >= config['minimum_unspliced'])
        truth.append(transform.state(s[keep]).mean(0))
    truth = np.stack(truth)
    with np.load(pack/'source_pack.npz',allow_pickle=False) as f:predictions['identity'] = [f['state'].copy()]
    errors = {arm:np.square(np.stack(values).mean(0)-truth).mean(1) for arm,values in predictions.items()}
    for arm,values in errors.items():
        for interval,mse in zip(config['prediction_intervals_hours'],values):
            records.append(dict(arm=arm,source_hour=interval[0],target_hour=interval[1],mse=float(mse),
                gain_over_identity=float(1-mse/errors['identity'][config['prediction_intervals_hours'].index(interval)])))
    result = dict(status='SCIFATE2_COARSE_TO_FINE_TRAINING_COMPLETE',
        task='train_r1_r2_r3_daily_transitions_predict_r4_five_hour_series',
        intervals_hours=config['prediction_intervals_hours'],mse_by_interval={k:v.tolist() for k,v in errors.items()},
        mean_mse={k:float(v.mean()) for k,v in errors.items()},
        joint_gain_over_static=(1-errors['velocity_joint']/errors['static_task']).tolist(),
        joint_gain_over_analytic=(1-errors['velocity_joint']/errors['analytic_task']).tolist(),
        r4_used_for_fitting=False,biological_p_values=False,seeds=config['seeds'],router_enabled=False,gfg_jvp_enabled=False)
    with (run.directory/'metrics.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    np.savez_compressed(run.directory/'predictions_and_truth.npz',truth=truth,**{arm:np.stack(v) for arm,v in predictions.items()})
    run.json('summary.json',result)
    (run.directory/'RESULTS.md').write_text('# sci-FATE2 coarse-to-fine training\n\n'+json.dumps(result,indent=2)+'\n')
    run.json('status.json',dict(status='complete'));print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();sub = parser.add_subparsers(dest='stage',required=True)
    p = sub.add_parser('prepare');p.add_argument('--config',default='configs/w5_joint_kinetics_scifate2_v1.json');p.add_argument('--output',required=True)
    t = sub.add_parser('train');t.add_argument('--pack',required=True);t.add_argument('--output',required=True)
    t.add_argument('--seed',type=int,required=True);t.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    e = sub.add_parser('evaluate');e.add_argument('--pack',required=True);e.add_argument('--training-root',required=True)
    e.add_argument('--output',required=True);e.add_argument('--wait',action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':prepare(args.config,args.output)
    elif args.stage == 'train':train(args.pack,args.output,args.seed,args.device)
    else:evaluate(args.pack,args.training_root,args.output,args.wait)
