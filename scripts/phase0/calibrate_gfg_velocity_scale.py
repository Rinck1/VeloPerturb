"""Training-only offline GFG scale audit. No validation/confirmation target access.

The adjacent displacement is a condition-centroid shift, not a cell fate label.
Positive scalars can calibrate units but cannot improve cosine direction.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import torch

from veloroute.artifacts import object_hash, save_csv, save_json, sha256
from veloroute.contracts import FitScope
from veloroute.full_model import FullConfig, load_full_checkpoint
from veloroute.gfg import GFGDynamics
from veloroute.gfg_experiments import read_gene_input
from veloroute.gpu_policy import enforce_gpu_policy
from veloroute.latent import FrozenSplicingTransform, load_pack
from veloroute.probes import permute_local


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def bootstrap_median(values, seed, repetitions=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(len(values), size=(repetitions, len(values)))]
    lo, hi = np.quantile(np.median(sample, axis=1), [.025, .975])
    return float(np.median(values)), float(lo), float(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='outputs/veloroute_velocity_scale_calibration_20260923')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=20260923)
    args = p.parse_args()
    enforce_gpu_policy(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path('outputs/veloroute_real_pipeline_20260912_v2/fold')
    gene_path = Path('outputs/veloroute_gfg_inputs_20260914/train_gene_source.npz')
    native_path = Path('/data/yuchang/GFG/results/mousebrain_graphbatch_directed_v1_seed0/final.pth')
    joint_path = Path('/data/yuchang/veloroute_gainprobe_20260922/corrected_renge_v2/gfg_joint/train/model.pt')
    source_path, target_path = root/'train_source.npz', root/'train_target.npz'
    paths = [source_path, target_path, gene_path, root/'transform.npz', native_path, joint_path, Path(__file__), Path('src/veloroute/gfg.py')]
    config = vars(args) | {'source_pack': str(source_path), 'target_pack': str(target_path),
        'native_checkpoint': str(native_path), 'joint_checkpoint': str(joint_path),
        'scale_rule': 'median_condition_norm_centroid_displacement_over_norm_mean_velocity',
        'heldout_targets_allowed': False, 'adjacent_days': [4, 5]}
    save_json(output/'config.json', config)
    save_json(output/'status.json', {'status': 'running', 'stage': 'load_training_only'})
    started = time.time()
    values, source, sm = read_gene_input(gene_path, source_path)
    target, tm = load_pack(target_path, expected_side='target')
    assert sm['role'] == tm['role'] == 'train' and sm['day'] == 4 and tm['day'] == 5
    assert sm['transform_hash'] == tm['transform_hash']
    assert set(source['conditions']) == set(target['conditions'])
    transform = FrozenSplicingTransform.load(root/'transform.npz')
    device = torch.device(args.device)
    us = torch.as_tensor(values, dtype=torch.float32, device=device)
    z = torch.as_tensor(source['z'], dtype=torch.float32, device=device)
    shuffled = values.copy()
    genes = values.shape[1]//2
    shuffled[:, :genes], permutation = permute_local(values[:, :genes], source, seed=args.seed, neighbors=10)
    shuffled = torch.as_tensor(shuffled, dtype=torch.float32, device=device)
    native = GFGDynamics(FullConfig(dynamics_backend='gfg', gfg_genes=genes)).to(device).eval()
    native.core.load_pretrained(native_path)
    native.prepare(us, torch.as_tensor(transform.components, dtype=torch.float32, device=device),
        cell_ids=source['cell_ids'].tolist(), scope=FitScope(frozenset(source['cell_ids'])))
    model, joint_meta = load_full_checkpoint(joint_path, map_location=device)
    assert joint_meta['transform_hash'] == sm['transform_hash']
    assert model.dynamics.fit_ids_hash == object_hash(sorted(source['cell_ids'].tolist()))
    models = {'frozen_native': native, 'joint_corrected_v2': model.dynamics}
    velocity = {}
    inference_rows = []
    for name, dynamics in models.items():
        for null, inputs in [('real_U', us), ('shuffled_U', shuffled)]:
            chunks, native_losses = [], []
            with torch.no_grad():
                for start in range(0, len(us), args.batch_size):
                    ix = slice(start, start+args.batch_size)
                    _, v, _ = dynamics(z[ix], inputs[ix])
                    chunks.append(v.cpu().numpy())
                    native_losses.append(float(dynamics.last_native_loss))
            key = f'{name}__{null}'
            velocity[key] = np.concatenate(chunks).astype('float64')
            inference_rows.append({'model': name, 'input': null, 'saved_velocity_scale': float(dynamics.velocity_scale),
                'mean_velocity_norm': float(np.linalg.norm(velocity[key], axis=1).mean()),
                'median_velocity_norm': float(np.median(np.linalg.norm(velocity[key], axis=1))),
                'native_loss_mean': float(np.mean(native_losses))})
            save_json(output/'status.json', {'status': 'running', 'stage': key, 'completed': len(velocity)})
    labels = sorted(set(source['conditions']))
    delta = np.stack([target['z'][target['conditions']==c].mean(0)-source['z'][source['conditions']==c].mean(0) for c in labels]).astype('float64')
    dn = np.linalg.norm(delta, axis=1)
    fit_rows, rows, arrays = [], [], {'source_ids': source['cell_ids'], 'conditions': source['conditions'], 'null_permutation': permutation}
    for name in models:
        real = velocity[f'{name}__real_U']
        real_mean = np.stack([real[source['conditions']==c].mean(0) for c in labels])
        ratios = dn/np.maximum(np.linalg.norm(real_mean, axis=1),1e-12)
        alpha, lo, hi = bootstrap_median(ratios, args.seed)
        speed, speed_lo, speed_hi = bootstrap_median(dn, args.seed)
        fit_rows.append({'model': name, 'alpha': alpha, 'ci_low': lo, 'ci_high': hi,
            'direction_only_common_speed': speed, 'direction_only_speed_ci_low': speed_lo, 'direction_only_speed_ci_high': speed_hi,
            'n_training_conditions': len(labels), 'resampling_unit': 'condition', 'fit_scope': 'training_day4_to_day5_only'})
        for null in ('real_U','shuffled_U'):
            v = velocity[f'{name}__{null}']
            arrays[f'{name}__{null}__velocity'] = v.astype('float32')
            vm = np.stack([v[source['conditions']==c].mean(0) for c in labels])
            for idx,c in enumerate(labels):
                ci=source['conditions']==c
                loco_alpha=float(np.median(np.delete(ratios,idx)))
                loco_speed=float(np.median(np.delete(dn,idx)))
                for mode,pred in [('native_scale1',vm[idx]),('robust_calibrated',alpha*vm[idx]),
                                  ('direction_only',speed*unit(vm[idx])),('loco_robust_calibrated',loco_alpha*vm[idx]),
                                  ('loco_direction_only',loco_speed*unit(vm[idx]))]:
                    cosine=float(unit(pred)@unit(delta[idx]))
                    rows.append({'model':name,'input':null,'condition':c,'scale_mode':mode,
                        'source_cells':int(ci.sum()),'target_cells':int((target['conditions']==c).sum()),
                        'raw_cell_norm_mean':float(np.linalg.norm(v[ci],axis=1).mean()),
                        'raw_centroid_velocity_norm':float(np.linalg.norm(vm[idx])),
                        'predicted_displacement_norm':float(np.linalg.norm(pred)),
                        'true_centroid_displacement_norm':float(dn[idx]),
                        'centroid_cosine':cosine,'centroid_displacement_rmse':float(np.sqrt(np.mean((pred-delta[idx])**2))),
                        'cell_direction_vs_centroid_cosine':float(np.mean(unit(v[ci])@unit(delta[idx]))),
                        'positive_scale':alpha if mode=='robust_calibrated' else loco_alpha if mode=='loco_robust_calibrated' else None})
    save_csv(output/'metrics.csv',rows)
    save_csv(output/'scale_fits.csv',fit_rows)
    save_csv(output/'inference_metrics.csv',inference_rows)
    np.savez_compressed(output/'training_velocities.npz',**arrays)
    aggregate=[]
    for name in models:
        for null in ('real_U','shuffled_U'):
            for mode in ('native_scale1','robust_calibrated','direction_only','loco_robust_calibrated','loco_direction_only'):
                rr=[r for r in rows if r['model']==name and r['input']==null and r['scale_mode']==mode]
                aggregate.append({'model':name,'input':null,'scale_mode':mode,**{k:float(np.mean([r[k] for r in rr])) for k in
                    ('predicted_displacement_norm','true_centroid_displacement_norm','centroid_cosine','centroid_displacement_rmse','cell_direction_vs_centroid_cosine')}})
    result={'status':'TRAINING_ONLY_OFFLINE_SCALE_AUDIT_COMPLETE','scope':'exploratory_training_adjacent_time_centroid_audit_not_cell_pairing',
        'source_day':4,'target_day':5,'source_cells':len(source['z']),'target_cells':len(target['z']),'conditions':labels,
        'heldout_target_read':False,'formal_generalization_claim':False,'anchored_joint_training':'NOT_RUN_offline_magnitude_anchor_only',
        'scale_fits':fit_rows,'aggregate_metrics':aggregate,'native_inference_metrics':inference_rows,
        'direction_invariance_expected':True,'null_permutation_moved_fraction':float(np.mean(permutation!=np.arange(len(permutation)))),
        'limits':['No true cell pairing: delta is condition mean(day5)-mean(day4).','Positive scalar calibration cannot change direction cosine.',
            'Joint model has already trained on these targets: joint direction/fit scores are in-sample, not independent validation.',
            'LOCO excludes a condition only from scale fitting, not from GFG joint pretraining.',
            'No day2/day3 quantified source folds found; audit uses already processed adjacent day4/day5.',
            'U-null uses same real-U-fitted alpha; it is not separately recalibrated for an advantage.',
            'This is dimension matching, not proof of absolute physical RNA speed.'],
        'elapsed_seconds':time.time()-started}
    save_json(output/'summary.json',result)
    text='# GFG velocity scale calibration audit\n\n'+json.dumps(result,indent=2)+'\n'
    (output/'RESULTS.md').write_text(text)
    outputs={x.name:sha256(x) for x in output.iterdir() if x.is_file() and x.name not in {'provenance.json','status.json'}}
    save_json(output/'provenance.json',{'status':'complete','config':config,'inputs':{str(p):sha256(p) for p in paths},'outputs':outputs,
        'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform()},'heldout_target_read':False})
    save_json(output/'status.json',{'status':'complete'})
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
