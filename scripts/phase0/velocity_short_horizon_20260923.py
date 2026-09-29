"""Exploratory train-calibrated, prediction-frozen RENGE day4->day5 geometry audit."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

from veloroute.artifacts import Run, load_config, save_csv, save_json, sha256, object_hash
from veloroute.contracts import FitScope
from veloroute.full_model import FullConfig, FullVeloRoute, load_full_checkpoint
from veloroute.gfg_experiments import read_gene_input
from veloroute.gpu_policy import enforce_gpu_policy
from veloroute.latent import FrozenSplicingTransform, load_pack
from veloroute.probes import permute_local


def nn_displacement(source, target, neighbors=10):
    result = np.zeros_like(source['z'])
    for condition in sorted(set(source['conditions'])):
        a = source['conditions'] == condition
        b = target['conditions'] == condition
        if not b.any():
            raise ValueError('Training condition missing target')
        nn = NearestNeighbors(n_neighbors=min(neighbors, int(b.sum()))).fit(target['z'][b])
        idx = nn.kneighbors(source['z'][a], return_distance=False)
        result[a] = target['z'][b][idx].mean(1) - source['z'][a]
    return result


def cosine(a, b):
    return np.sum(a*b, axis=1)/np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-12)


def robust_scale(v, displacement, conditions):
    # Calibrate only the positive magnitude against training condition
    # centroids. NN barycenters are too noisy and can select alpha=0.
    ratios=[]
    for c in sorted(set(conditions)):
        m=conditions==c
        vm=np.asarray(v)[m].mean(0)
        dm=np.asarray(displacement)[m].mean(0)
        if np.linalg.norm(vm)>1e-10: ratios.append(np.linalg.norm(dm)/np.linalg.norm(vm))
    return float(np.median(ratios)) if ratios else 1.0


def decoded_velocity(model, values, states, device):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(values), 16):
            result = model.dynamics(
                torch.as_tensor(states[start:start+16], dtype=torch.float32, device=device),
                torch.as_tensor(values[start:start+16], dtype=torch.float32, device=device))
            chunks.append(result[1].detach().cpu().numpy())
    return np.concatenate(chunks)


def energy(x, y):
    """Unpaired energy V-statistic; lower is better, no NN pseudo-pairing."""
    x, y = np.asarray(x, dtype='float64'), np.asarray(y, dtype='float64')
    return float(max(0., 2*cdist(x,y).mean()-cdist(x,x).mean()-cdist(y,y).mean()))


def comparison_rows(rows):
    comparisons = []
    rng = np.random.default_rng(20260923)
    for real in ('GFG_native', 'GFG_joint'):
        for control in ('identity', 'train_mean_shift', 'static_z_ridge', 'raw_SU_ridge',
                        real+'_preUshuffle', real+'_postVshuffle'):
            for key in ('energy_distance', 'pseudobulk_mse', 'scaled_nn_barycenter_mse'):
                gains = []
                conditions = sorted({r['condition'] for r in rows if r['role']=='validation'})
                for c in conditions:
                    group = [r for r in rows if r['role']=='validation' and r['condition']==c]
                    rv = [r[key] for r in group if r['arm']==real]
                    cv = [r[key] for r in group if r['arm']==control or r['arm'].startswith(control+'_s')]
                    if len(rv)!=1 or not cv:
                        raise ValueError('Incomplete paired comparison')
                    gains.append(float(np.mean(cv)-rv[0]))
                gains = np.asarray(gains)
                boot = gains[rng.integers(len(gains),size=(10000,len(gains)))].mean(1)
                lo, hi = np.quantile(boot,[.025,.975])
                comparisons.append(dict(real=real,control=control,metric=key,gain=float(gains.mean()),
                    ci_low=float(lo),ci_high=float(hi),n_conditions=len(gains),
                    inference='exploratory_condition_bootstrap_after_null_seed_averaging_not_confirmatory'))
    return comparisons


def metrics(role, arm, source, target, vectors, scale, seed):
    rows = []
    for condition in sorted(set(source['conditions'])):
        a, b = source['conditions'] == condition, target['conditions'] == condition
        z, y, v = source['z'][a], target['z'][b], vectors[a]
        if min(len(z), len(y)) < 20:
            continue
        nn = NearestNeighbors(n_neighbors=min(10, len(y))).fit(y)
        indices = nn.kneighbors(z, return_distance=False)
        displacement = y[indices].mean(1)-z
        endpoint = z+scale*v
        mean_delta = y.mean(0)-z.mean(0)
        # Geometry-only ranking: among observed targets, how highly does the
        # extrapolated point rank the original source's nearest target?
        dist = cdist(endpoint, y)
        anchor = indices[:, 0]
        anchor_distance = dist[np.arange(len(z)), anchor]
        rank_fraction = (dist < anchor_distance[:, None]).mean(1)
        rows.append(dict(role=role, arm=arm, condition=str(condition), seed=seed,
            source_n=len(z), target_n=len(y), scale=float(scale),
            nn_barycenter_direction_cosine=float(cosine(v, displacement).mean()),
            condition_mean_direction_cosine=float(cosine(v.mean(0)[None], mean_delta[None])[0]),
            scaled_nn_barycenter_mse=float(np.square(scale*v-displacement).mean()),
            endpoint_target_nn_distance=float(np.min(dist, axis=1).mean()),
            original_nn_rank_fraction=float(rank_fraction.mean()),
            original_nn_retention_top10=float((rank_fraction <= min(10/len(y), 1)).mean()),
            pseudobulk_mse=float(np.square(endpoint.mean(0)-y.mean(0)).mean()),
            energy_distance=energy(endpoint,y),
            velocity_norm_mean=float(np.linalg.norm(v, axis=1).mean())))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--fold', default='outputs/veloroute_real_pipeline_20260912_v2/fold')
    p.add_argument('--gene-dir', default='outputs/veloroute_gfg_inputs_20260914')
    p.add_argument('--joint-checkpoint', default='/data/yuchang/veloroute_gainprobe_20260922/corrected_renge_v2/gfg_joint/train/model.pt')
    args = p.parse_args()
    enforce_gpu_policy(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    fold, gene_dir = Path(args.fold), Path(args.gene_dir)
    transform = FrozenSplicingTransform.load(fold/'transform.npz')
    cfg = load_config('configs/veloroute_gfg_joint_20260914.yaml')
    data = {}
    for role in ('train', 'validation'):
        data[role] = read_gene_input(gene_dir/f'{role}_gene_source.npz', fold/f'{role}_source.npz')
    train_target, train_meta = load_pack(fold/'train_target.npz', expected_side='target')
    if data['train'][2]['day'] != 4 or train_meta['day'] != 5:
        raise ValueError('Only registered day4->day5 is allowed')
    model_cfg = dict(cfg['model'])
    model_cfg.update(gfg_genes=len(transform.selected), state_dim=transform.components.shape[0])
    native = FullVeloRoute(FullConfig(**model_cfg)).to(args.device)
    native.dynamics.core.load_pretrained(cfg['gfg_checkpoint'])
    native.dynamics.prepare(torch.as_tensor(data['train'][0], dtype=torch.float32, device=args.device),
        torch.as_tensor(transform.components, dtype=torch.float32, device=args.device),
        cell_ids=list(data['train'][1]['cell_ids']), scope=FitScope(frozenset(data['train'][1]['cell_ids'])))
    native.eval()
    joint, joint_meta = load_full_checkpoint(args.joint_checkpoint, map_location=args.device)
    if joint_meta['transform_hash'] != data['train'][2]['transform_hash']:
        raise ValueError('Joint checkpoint transform mismatch')
    if joint.dynamics.fit_ids_hash != object_hash(sorted(data['train'][1]['cell_ids'].tolist())):
        raise ValueError('Joint dynamics normalization fit scope mismatch')
    joint.eval()
    sources = {role: triple[1] for role, triple in data.items()}
    displacements = nn_displacement(sources['train'], train_target)
    per_condition = []
    for c in sorted(set(sources['train']['conditions'])):
        per_condition.append(train_target['z'][train_target['conditions'] == c].mean(0)-sources['train']['z'][sources['train']['conditions'] == c].mean(0))
    mean_shift = np.mean(per_condition, axis=0)
    predictors = {'static_z_ridge': Ridge(alpha=1.).fit(sources['train']['z'], displacements),
        'raw_SU_ridge': Ridge(alpha=1.).fit(np.concatenate((sources['train']['z'], sources['train']['u_features']), axis=1), displacements)}
    predictions, scales = {}, {}
    # All source-only vectors and training-only scales are built before reading
    # validation_target. Shuffle is pre-GFG, within the existing local strata.
    for role, (values, source, _) in data.items():
        vectors = {'identity': np.zeros_like(source['z']),
            'train_mean_shift': np.broadcast_to(mean_shift, source['z'].shape).copy(),
            'raw_steady_state_velocity': source['velocity'],
            'raw_U_residual': source['u_features']-source['u_predicted'],
            'static_z_ridge': predictors['static_z_ridge'].predict(source['z']),
            'raw_SU_ridge': predictors['raw_SU_ridge'].predict(np.concatenate((source['z'], source['u_features']), axis=1))}
        for label, model in [('GFG_native', native), ('GFG_joint', joint)]:
            vectors[label] = decoded_velocity(model, values, source['z'], args.device)
            for seed in range(3):
                shuffled = values.copy()
                n_genes = values.shape[1]//2
                shuffled[:, :n_genes] = permute_local(values[:, :n_genes], source, seed=seed, neighbors=10)[0]
                vectors[f'{label}_preUshuffle_s{seed}'] = decoded_velocity(model, shuffled, source['z'], args.device)
                vectors[f'{label}_postVshuffle_s{seed}'] = permute_local(vectors[label], source, seed=seed, neighbors=10)[0]
        predictions[role] = vectors
        if role == 'train':
            for arm, v in vectors.items():
                # Negative controls use the real arm's fixed scale, not a separately
                # optimized scale. All fitting happens on training conditions only.
                base = arm.split('_preUshuffle')[0].split('_postVshuffle')[0]
                if base != arm:
                    scales[arm] = scales[base]
                else:
                    scales[arm] = 1. if arm in {'identity','train_mean_shift','static_z_ridge','raw_SU_ridge'} else robust_scale(v, displacements, source['conditions'])
    config = dict(fold=str(fold), gene_dir=str(gene_dir), joint_checkpoint=args.joint_checkpoint,
        task='ER-short_day4_to_day5', seed=0, shuffle_seeds=[0,1,2], robust_scale='nonnegative_condition_balanced_Huber_NN10_barycenter_train_only',
        rank_semantics='geometry_only_original_nearest_target_retention_not_actual_pairing_accuracy',
        null_scale='same_as_real_arm', distribution_metric='energy_V_statistic')
    inputs = [fold/'transform.npz', fold/'train_source.npz', fold/'train_target.npz', fold/'validation_source.npz',
        gene_dir/'train_gene_source.npz', gene_dir/'validation_gene_source.npz', cfg['gfg_checkpoint'], args.joint_checkpoint, Path(__file__)]
    with Run(args.output, stage='velocity_short_horizon_exploratory', kind='development', config=config, inputs=inputs, seed=0) as run:
        save_json(run.directory/'scales.json', scales)
        payload = {f'{role}__{arm}': vector for role, vectors in predictions.items() for arm, vector in vectors.items()}
        for role, source in sources.items():
            payload[role+'__cell_ids'] = source['cell_ids']
            payload[role+'__conditions'] = source['conditions']
        with (run.directory/'frozen_source_vectors.npz').open('xb') as stream:
            np.savez_compressed(stream, **payload)
        save_json(run.directory/'prediction_freeze.json', dict(source_vectors_sha256=sha256(run.directory/'frozen_source_vectors.npz'),
            scales_sha256=sha256(run.directory/'scales.json'), validation_target_opened=False, confirmation_opened=False))
        # The evaluation boundary: predictions and calibration were frozen above.
        validation_target, vm = load_pack(fold/'validation_target.npz', expected_side='target')
        if vm['transform_hash'] != data['validation'][2]['transform_hash'] or vm['day'] != 5:
            raise ValueError('Evaluation target contract mismatch')
        run.provenance['evaluation_target'] = {'path':str(fold/'validation_target.npz'),
            'sha256':sha256(fold/'validation_target.npz'),'read_after_prediction_freeze':True}
        rows = []
        for role, target in [('train', train_target), ('validation', validation_target)]:
            for arm, v in predictions[role].items():
                suffix = arm.rsplit('_s', 1)[1] if '_s' in arm else ''
                seed = int(suffix) if suffix.isdigit() else 0
                rows += metrics(role, arm, sources[role], target, v, scales[arm], seed)
        save_csv(run.directory/'metrics.csv', rows)
        fields = ['nn_barycenter_direction_cosine','condition_mean_direction_cosine','scaled_nn_barycenter_mse',
            'endpoint_target_nn_distance','original_nn_rank_fraction','original_nn_retention_top10','pseudobulk_mse','velocity_norm_mean','energy_distance']
        aggregate = []
        for role in ('train','validation'):
            for arm in predictions[role]:
                sub = [r for r in rows if r['role']==role and r['arm']==arm]
                aggregate.append(dict(role=role,arm=arm,n_conditions=len(sub),scale=scales[arm],
                    **{key:float(np.mean([r[key] for r in sub])) for key in fields}))
        save_csv(run.directory/'aggregate.csv', aggregate)
        comparisons = comparison_rows(rows)
        save_csv(run.directory/'comparisons.csv', comparisons)
        summary = dict(status='EXPLORATORY_ER_SHORT_COMPLETE', task='RENGE_day4_to_day5',
            adjacent_day2_day3_available=False, confirmation='SEALED', validation_target_used_for_fitting=False,
            source_vectors_frozen_before_validation=True, scale_fitted_on='training_conditions_NN10_barycentric_displacements',
            transform_scope='existing_train_conditions_day4_and_day5_not_independent_time_anchor',
            gfg_native_scope='MouseBrain_pretrained_source_only_normalization',
            gfg_joint_scope='existing_task_adapted_train_conditions_checkpoint',
            paired_ground_truth=False, biological_fate_accuracy=False,
            limitations=['No true individual-cell pairing; NN/rank scores are geometric diagnostics.',
                'Existing PCA and raw residual estimator fitted on training day4+day5; held-out conditions safe but not independent temporal validation.',
                'Only 4 held-out validation TFs; one GFG model seed. Three shuffle seeds are null repeats, not biological replicates.',
                'Day2/day3 S/U missing; no de novo 2->5 result.',
                'Joint checkpoint was selected/trained in prior exploratory run; this is not fresh preregistered confirmation.'],
            comparisons=comparisons, aggregate=aggregate)
        save_json(run.directory/'summary.json', summary)
        (run.directory/'RESULTS.md').write_text('# RENGE short-horizon velocity geometry audit\n\n'+json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2), flush=True)


if __name__ == '__main__':
    main()
