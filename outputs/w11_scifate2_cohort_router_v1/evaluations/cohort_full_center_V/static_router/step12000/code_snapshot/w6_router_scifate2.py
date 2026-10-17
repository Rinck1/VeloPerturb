"""Local source dynamics -> response router/experts, with frozen W5 mean forecasts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from w4_artifacts import Artifacts,sha256
from w4_shorttime import read_count_rows,read_metadata
from w4_streams import SourceTransform
from w6_local_router import energy_from_weights,fit_local,local_pooled_velocity,predict_local


ROOT = Path(__file__).resolve().parents[2]
CODE = [Path(__file__).with_name(name) for name in ('w6_router_scifate2.py','w6_local_router.py',
    'w4_streams.py','w4_shorttime.py','w4_artifacts.py')]


def rate_from_checkpoint(path,base,velocity):
    saved = torch.load(path,map_location='cpu',weights_only=True)['state_dict']
    feature = torch.tensor(np.c_[base,velocity],dtype=torch.float32)
    with torch.no_grad():
        rate = torch.nn.functional.linear((feature-saved['input_mean'])/saved['input_scale'],
            saved['readout.weight'],saved['readout.bias'])
    return rate.numpy().astype(float)


def prepare(config_path,output):
    config = json.loads(Path(config_path).read_text());old_pack = ROOT/config['global_pack']
    old = json.loads((old_pack/'manifest.json').read_text());base_config = old['config']
    if (config['gfg_jvp_enabled'] or not config['preserve_global_mean']
            or config['velocity_consumers'] != ['router'] or config['seeds'] != base_config['seeds']):
        raise ValueError('Use the fixed mean-preserving analytic-velocity expansion')
    data = Path(old['data_path']);transform_path = Path(old['transform_path'])
    if sha256(data) != base_config['data_sha256']:raise ValueError('Source counts changed')
    global_root = ROOT/config['global_training']
    checkpoints = [global_root/f'seed{seed}'/'analytic_task.pt' for seed in config['seeds']]
    run = Artifacts(output,seed=config['seeds'],inputs=[Path(config_path),data,transform_path,
        old_pack/'manifest.json',old_pack/'training_pack.npz',old_pack/'source_pack.npz']+checkpoints+CODE)
    transform = SourceTransform.load(transform_path);obs,genes = read_metadata(data)
    np.testing.assert_array_equal(genes,transform.gene_names)
    with np.load(old_pack/'training_pack.npz',allow_pickle=False) as saved:
        rates = np.stack([rate_from_checkpoint(cp,saved['base'],saved['velocity']) for cp in checkpoints])
    with np.load(old_pack/'source_pack.npz',allow_pickle=False) as saved:
        origin_states = saved['state'].copy();durations = saved['duration'].copy()
        test_rates = np.stack([rate_from_checkpoint(cp,saved['base'],saved['velocity']) for cp in checkpoints])
    anchors = []
    for i,seed in enumerate(config['seeds']):
        path = global_root/f'seed{seed}'/'predictions.npz'
        pm = json.loads((path.parent/'prediction_manifest.json').read_text())
        if pm['r4_used_for_fitting'] or sha256(path) != pm['prediction_sha256']:
            raise ValueError('Global prediction changed')
        with np.load(path,allow_pickle=False) as saved:anchor = saved['analytic_task'].copy()
        np.testing.assert_allclose(origin_states+durations[:,None]*test_rates[i],anchor,atol=1e-6,rtol=1e-6)
        anchors.append(anchor)
    anchors = np.stack(anchors)
    cache = {}
    rep_values,tp_values = [obs[key].astype(str).to_numpy() for key in ('rep','timepoint')]
    def load(rep,tp):
        key = (rep,tp)
        if key in cache:return cache[key]
        rows = np.flatnonzero((rep_values == rep)&(tp_values == tp))
        s,u = [read_count_rows(data,layer,rows) for layer in ('spliced','unspliced')]
        ds,du = np.asarray(s.sum(1)).ravel(),np.asarray(u.sum(1)).ravel()
        keep = (ds >= base_config['minimum_spliced'])&(du >= base_config['minimum_unspliced'])
        if keep.sum() < config['neighbors']:raise ValueError('Source cohort is smaller than local neighborhood')
        s,u = s[keep],u[keep]
        value = dict(s=s,u=u,z=transform.state(s),ids=obs.index[rows[keep]].to_numpy(dtype=str))
        cache[key] = value
        return value
    def baseline(rep,tp):
        group = load(rep,tp)
        sn,un = transform.normalized(group['s'].sum(0),group['u'].sum(0))
        return transform.projected_rate(sn,un-transform.gamma*sn)[0]
    paths,all_z,all_v = [],[],[]
    index = 0
    for rep in base_config['training_replicates']:
        reference = baseline(rep,'D3')
        for day in base_config['training_days'][:-1]:
            current,future = load(rep,f'D{day}'),load(rep,f'D{day+1}')
            v = local_pooled_velocity(transform,current['s'],current['u'],current['z'],reference,config['neighbors'])
            name = f'train_{rep}_D{day}.npz';path = run.directory/name
            np.savez_compressed(path,z=current['z'].astype('float32'),velocity=v,target=future['z'].astype('float32'),
                global_rates=rates[:,index],cell_ids=current['ids'])
            paths.append(dict(rep=rep,source_day=day,target_day=day+1,path=str(path.resolve()),sha256=sha256(path)))
            all_z.append(current['z']);all_v.append(v);index += 1
            run.json('status.json',dict(status='preparing',completed_training_groups=index))
            print(json.dumps(dict(status='LOCAL_GROUP_READY',rep=rep,day=day,cells=len(v))),flush=True)
    z,v = np.vstack(all_z),np.vstack(all_v)
    normalization = dict(z_mean=z.mean(0),z_scale=np.maximum(z.std(0),1e-3),
        v_mean=v.mean(0),v_scale=np.maximum(v.std(0),1e-5))
    np.savez_compressed(run.directory/'normalization.npz',**normalization)
    del z,v,all_z,all_v
    tests = [];rep = base_config['prediction_replicate'];reference = baseline(rep,'05h')
    for i,(start,stop) in enumerate(base_config['prediction_intervals_hours']):
        current = load(rep,f'{start:02d}h')
        v = local_pooled_velocity(transform,current['s'],current['u'],current['z'],reference,config['neighbors'])
        np.testing.assert_allclose(current['z'].mean(0),origin_states[i],atol=1e-7)
        path = run.directory/f'source_{start:02d}_to_{stop:02d}.npz'
        np.savez_compressed(path,z=current['z'].astype('float32'),velocity=v,cell_ids=current['ids'],
            source_mean=origin_states[i],anchors=anchors[:,i],duration=durations[i])
        tests.append(dict(source_hour=start,target_hour=stop,path=str(path.resolve()),sha256=sha256(path)))
    run.json('manifest.json',dict(status='LOCAL_ROUTER_PACK_READY',config=config,base_config=base_config,
        data_path=str(data),transform_path=str(transform_path),training_groups=paths,prediction_groups=tests,
        normalization_sha256=sha256(run.directory/'normalization.npz'),
        r4_used_for_fitting=False,future_target_used_for_origin_features=False,
        dynamics='within_source_knn_count_pooling_minus_source_baseline_global_field',
        teacher_or_OT_labels=False,global_mean_frozen=True))
    run.json('status.json',dict(status='complete'))
    print(json.dumps(dict(status='LOCAL_ROUTER_PACK_READY',training_groups=len(paths))),flush=True)


def train(pack,output,seed,device):
    pack = Path(pack);manifest = json.loads((pack/'manifest.json').read_text());config = manifest['config']
    if seed not in config['seeds']:raise ValueError('Unknown seed')
    if device != 'cpu':
        from veloroute.gpu_policy import enforce_gpu_policy
        enforce_gpu_policy(device)
    seed_index = config['seeds'].index(seed)
    files = [Path(item['path']) for item in manifest['training_groups']+manifest['prediction_groups']]
    run = Artifacts(output,seed=seed,inputs=[pack/'manifest.json',pack/'normalization.npz']+files+CODE)
    groups = []
    for item in manifest['training_groups']:
        if sha256(item['path']) != item['sha256']:raise ValueError('Training pack changed')
        with np.load(item['path'],allow_pickle=False) as saved:
            groups.append(dict(z=saved['z'].copy(),velocity=saved['velocity'].copy(),target=saved['target'].copy(),
                global_rate=saved['global_rates'][seed_index].copy()))
    with np.load(pack/'normalization.npz',allow_pickle=False) as saved:normalization = {k:saved[k] for k in saved.files}
    histories,artifacts = {},[]
    for arm in config['arms']:
        def progress(row):
            run.json('status.json',dict(status='training',arm=arm,seed=seed,**row))
        model,history = fit_local(groups,normalization,config,arm=arm,seed=seed,device=device,progress=progress)
        torch.save(dict(state_dict={k:v.cpu() for k,v in model.state_dict().items()},config=config,arm=arm,seed=seed),
            run.directory/f'{arm}.pt')
        histories[arm] = history
        for item in manifest['prediction_groups']:
            if sha256(item['path']) != item['sha256']:raise ValueError('Source prediction group changed')
            with np.load(item['path'],allow_pickle=False) as saved:
                z,v = saved['z'],saved['velocity'];duration = float(saved['duration'])
                anchor = saved['anchors'][seed_index];source_mean = saved['source_mean']
                global_rate = (anchor-source_mean)/duration
            positions,q = predict_local(model,z,v,global_rate,duration)
            positions,q = positions.astype(float),q.astype(float)
            q /= q.sum(1,keepdims=True)
            # Source-only roundoff correction; does not consult a future target.
            numerical_shift = anchor-(positions*q[:,:,None]).sum(1).mean(0)
            positions += numerical_shift
            path = run.directory/f'{arm}_{item["source_hour"]:02d}_to_{item["target_hour"]:02d}.npz'
            np.savez_compressed(path,candidates=positions.astype('float32'),q=q,
                mean_roundoff_correction=numerical_shift,anchor=anchor)
            artifacts.append(dict(arm=arm,source_hour=item['source_hour'],target_hour=item['target_hour'],
                path=str(path.resolve()),sha256=sha256(path)))
        run.json('training.json',histories)
        print(json.dumps(dict(status='LOCAL_ARM_TRAINED',arm=arm,seed=seed,parameters=history['parameters'])),flush=True)
    run.json('prediction_manifest.json',dict(status='LOCAL_PREDICTIONS_READY',seed=seed,artifacts=artifacts,
        global_mean_preserved=True,r4_used_for_fitting=False,weighted_modes_not_averaged_endpoints=True))
    run.json('status.json',dict(status='complete'))


def moments(x,w):
    x,w = np.asarray(x,dtype=float),np.asarray(w,dtype=float)
    w = w/w.sum();mean = w@x;centered = x-mean
    return mean,(centered*w[:,None]).T@centered


def evaluate(pack,training_root,output,wait,device):
    if device != 'cpu':
        from veloroute.gpu_policy import enforce_gpu_policy
        enforce_gpu_policy(device)
    pack,training_root = Path(pack),Path(training_root)
    manifest = json.loads((pack/'manifest.json').read_text());config = manifest['config'];bc = manifest['base_config']
    paths = [training_root/f'seed{seed}' for seed in config['seeds']];deadline = time.monotonic()+7200
    while not all((p/'prediction_manifest.json').is_file() for p in paths):
        if not wait or time.monotonic() > deadline:raise TimeoutError('Local-router predictions incomplete')
        time.sleep(15)
    frozen = []
    for path in paths:
        pm = json.loads((path/'prediction_manifest.json').read_text())
        if pm['r4_used_for_fitting']:raise ValueError('Held-out series entered optimization')
        for item in pm['artifacts']:
            if sha256(item['path']) != item['sha256']:raise ValueError('Candidate predictions changed')
            frozen.append(dict(seed=pm['seed'],**item))
    run = Artifacts(output,seed=config['seeds'],inputs=[pack/'manifest.json',Path(manifest['data_path']),
        Path(manifest['transform_path'])]+[Path(item['path']) for item in frozen]+CODE)
    run.json('frozen_grid.json',dict(artifacts=frozen))
    transform = SourceTransform.load(manifest['transform_path']);obs,_ = read_metadata(manifest['data_path'])
    records,route_rows,sample_manifest = [],[],{}
    arms = ['identity','global_translation']+config['arms']
    generator = np.random.default_rng(config['evaluation_seed'])
    for item in manifest['prediction_groups']:
        start,stop = item['source_hour'],item['target_hour']
        with np.load(item['path'],allow_pickle=False) as saved:
            source_z = saved['z'].astype(float);source_mean = saved['source_mean'];anchors = saved['anchors']
        source_z += source_mean-source_z.mean(0)
        rows = np.flatnonzero((obs['rep'].astype(str).to_numpy() == bc['prediction_replicate'])
            &(obs['timepoint'].astype(str).to_numpy() == f'{stop:02d}h'))
        s,u = [read_count_rows(manifest['data_path'],name,rows) for name in ('spliced','unspliced')]
        keep = (np.asarray(s.sum(1)).ravel() >= bc['minimum_spliced'])&(np.asarray(u.sum(1)).ravel() >= bc['minimum_unspliced'])
        target_z = transform.state(s[keep]);target_mean = target_z.mean(0)
        target_cov = np.cov(target_z,rowvar=False,bias=True)
        si = generator.choice(len(source_z),min(len(source_z),config['evaluation_source_cells']),replace=False)
        ti = generator.choice(len(target_z),min(len(target_z),config['evaluation_target_cells']),replace=False)
        sample_manifest[f'{start}_to_{stop}'] = dict(source_rows=si.tolist(),target_rows=ti.tolist())
        for arm in arms:
            means,covariances,sub_x,sub_w,entropies = [],[],[],[],[]
            for seed_index,seed in enumerate(config['seeds']):
                if arm in ('identity','global_translation'):
                    shift = np.zeros_like(source_mean) if arm == 'identity' else anchors[seed_index]-source_mean
                    x = (source_z+shift)[:,None,:];q = np.ones((len(source_z),1))
                else:
                    prediction = next(r for r in frozen if r['seed'] == seed and r['arm'] == arm
                        and r['source_hour'] == start and r['target_hour'] == stop)
                    with np.load(prediction['path'],allow_pickle=False) as saved:
                        x,q = saved['candidates'].astype(float),saved['q']
                mean,covariance = moments(x.reshape(-1,x.shape[-1]),q.reshape(-1)/len(x))
                means.append(mean);covariances.append(covariance)
                sub_x.append(x[si].reshape(-1,x.shape[-1]));sub_w.append(q[si].reshape(-1)/(len(si)*len(config['seeds'])))
                entropy = float(-(q*np.log(np.maximum(q,1e-12))).sum(1).mean());entropies.append(entropy)
                for mode,usage in enumerate(q.mean(0)):
                    route_rows.append(dict(arm=arm,seed=seed,source_hour=start,target_hour=stop,mode=mode,
                        usage=float(usage),cell_entropy=entropy))
            mean = np.mean(means,axis=0)
            covariance = np.mean([c+np.outer(m-mean,m-mean) for m,c in zip(means,covariances)],axis=0)
            a,b,w = [torch.as_tensor(value,dtype=torch.float64,device=device) for value in
                (np.concatenate(sub_x),target_z[ti],np.concatenate(sub_w))]
            with torch.no_grad():energy = float(energy_from_weights(a,w,b))
            expected_mean = source_mean if arm == 'identity' else anchors.mean(0)
            record = dict(arm=arm,source_hour=start,target_hour=stop,energy_distance=energy,
                mean_mse=float(np.square(mean-target_mean).mean()),
                variance_ratio=float(np.trace(covariance)/max(np.trace(target_cov),1e-12)),
                covariance_relative_error=float(np.linalg.norm(covariance-target_cov)/max(np.linalg.norm(target_cov),1e-12)),
                mean_anchor_error=float(np.linalg.norm(mean-expected_mean)),route_entropy=float(np.mean(entropies)),
                source_cells=len(source_z),target_cells=len(target_z),energy_source_cells=len(si),energy_target_cells=len(ti))
            records.append(record)
        print(json.dumps(dict(status='LOCAL_INTERVAL_EVALUATED',source_hour=start,target_hour=stop)),flush=True)
    metrics = {arm:{metric:float(np.mean([r[metric] for r in records if r['arm'] == arm]))
        for metric in ('energy_distance','mean_mse','variance_ratio','covariance_relative_error')} for arm in arms}
    dynamic = metrics['velocity_router']['energy_distance']
    comparisons = {arm:float(1-dynamic/metrics[arm]['energy_distance']) for arm in arms if arm != 'velocity_router'}
    result = dict(status='LOCAL_VELOCITY_ROUTER_EXPANSION_COMPLETE',metrics=metrics,
        velocity_router_energy_gain=comparisons,
        velocity_router_energy_gain_by_interval={arm:[float(1-d['energy_distance']/r['energy_distance'])
            for d,r in zip([a for a in records if a['arm'] == 'velocity_router'],[a for a in records if a['arm'] == arm])]
            for arm in ('global_translation','k1','static_router')},
        mean_anchor_max_error=max(r['mean_anchor_error'] for r in records),
        seeds=config['seeds'],weighted_prediction_ensemble=True,
        r4_used_for_fitting=False,gfg_jvp_enabled=False,individual_fate_truth_used=False,
        global_mean_frozen=True,distribution_energy_uses_fixed_subsamples=True)
    for filename,values in (('metrics.csv',records),('mode_usage.csv',route_rows)):
        with (run.directory/filename).open('x',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    run.json('evaluation_samples.json',sample_manifest);run.json('summary.json',result)
    (run.directory/'RESULTS.md').write_text('# Local velocity router expansion\n\n'
        'The original analytic population mean is frozen; experts change only distribution structure. '
        'Energy uses fixed source/target subsamples and an equally weighted seed mixture.\n\n'
        +json.dumps(result,indent=2)+'\n')
    run.json('status.json',dict(status='complete'));print(json.dumps(result),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();sub = parser.add_subparsers(dest='stage',required=True)
    p = sub.add_parser('prepare');p.add_argument('--config',default='configs/w6_local_router_scifate2_v1.json');p.add_argument('--output',required=True)
    t = sub.add_parser('train');t.add_argument('--pack',required=True);t.add_argument('--output',required=True)
    t.add_argument('--seed',type=int,required=True);t.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    e = sub.add_parser('evaluate');e.add_argument('--pack',required=True);e.add_argument('--training-root',required=True)
    e.add_argument('--output',required=True);e.add_argument('--wait',action='store_true');e.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args = parser.parse_args()
    if args.stage == 'prepare':prepare(args.config,args.output)
    elif args.stage == 'train':train(args.pack,args.output,args.seed,args.device)
    else:evaluate(args.pack,args.training_root,args.output,args.wait,args.device)
