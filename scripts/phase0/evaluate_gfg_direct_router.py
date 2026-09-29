"""Evaluate the complete fixed direct-router follow-up, including exact mixtures."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from veloroute.artifacts import Run, load_config, save_csv, save_json, sha256
from veloroute.experiments import evaluate_prediction
from veloroute.latent import FrozenSplicingTransform, load_pack
from veloroute.metrics import paired_condition_bootstrap
from veloroute.mixture_energy import mixture_energy

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
config = load_config('configs/veloroute_gfg_direct_router_20260914.yaml')
base = load_config('configs/veloroute_gfg_joint_20260914.yaml')
root, fold = Path('outputs/veloroute_gfg_direct_router_20260914'), Path(base['fold'])
frozen = []
for seed in config['seeds']:
    for arm in config['arms']:
        selection = json.loads((root/f'seed{seed}'/arm/'selection.json').read_text())
        if selection['arm'] != arm or selection['seed'] != seed:
            raise ValueError('Direct-router result grid identity mismatch')
        training, prediction = Path(selection['training']), Path(selection['prediction'])
        for directory in (training, prediction):
            if json.loads((directory/'provenance.json').read_text())['status'] != 'complete':
                raise ValueError('Complete all nine fixed-budget models before any target evaluation')
        if sha256(training/'model.pt') != selection['model_sha256'] or sha256(prediction/'predictions.npz') != selection['prediction_sha256']:
            raise ValueError('Direct-router artifacts modified after freezing')
        frozen.append(dict(seed=seed, arm=arm, prediction=str(prediction), sha256=selection['prediction_sha256']))
torch.set_num_threads(2)
with Run(args.output, stage='GFG_direct_router_fixed_grid_evaluation', kind='development', config=config,
         inputs=[Path(r['prediction'])/'predictions.npz' for r in frozen], seed=config['seeds']) as run:
    save_json(run.directory/'frozen_grid.json', frozen)
    target, tm = load_pack(fold/'validation_target.npz', expected_side='target')
    if tm['role'] != 'validation':
        raise ValueError('Final confirmation remains sealed')
    with np.load(fold/'validation_target_genes.npz', allow_pickle=False) as file:
        gene_target = file['gene_logspliced']
    transform = FrozenSplicingTransform.load(fold/'transform.npz')
    rows, sample_rows, geometry_by_seed = [], [], {}
    for record in frozen:
        prediction = Path(record['prediction'])
        # Shared evaluator audits target/gene/ID/hash contracts and reports sampled metrics too.
        sample_rows.extend(evaluate_prediction(prediction, fold/'validation_target.npz',
            run.directory/f"sampled_{record['arm']}_seed{record['seed']}",
            target_genes_path=fold/'validation_target_genes.npz'))
        with np.load(prediction/'predictions.npz', allow_pickle=False) as file:
            pred = {key: file[key] for key in file.files}
        key = record['seed']
        if key not in geometry_by_seed:
            geometry_by_seed[key] = pred['candidates']
        np.testing.assert_array_equal(geometry_by_seed[key], pred['candidates'])
        gene_means, local = [], []
        for condition in sorted(set(target['conditions'])):
            pi, ti = pred['conditions'] == condition, target['conditions'] == condition
            x, q, y = pred['candidates'][pi].astype('float64'), pred['q'][pi].astype('float64'), target['z'][ti].astype('float64')
            q = q/q.sum(-1, keepdims=True)  # Remove only float32 softmax roundoff.
            exact = float(mixture_energy(torch.from_numpy(x), torch.from_numpy(q), torch.from_numpy(y)))
            mean = np.einsum('nkd,nk->d', x, q)/len(x)
            second = np.einsum('nkd,nk->d', x*x, q)/len(x)
            variance = float((second-mean*mean).sum())
            ratio = variance/max(float(y.var(0).sum()), 1e-12)
            local.append(dict(condition=str(condition), seed=record['seed'], arm=record['arm'],
                exact_mixture_energy=max(0., exact), log_variance_error=abs(float(np.log(max(ratio, 1e-12)))),
                variance_ratio=ratio, mode_entropy=float((-(q*np.log(np.maximum(q, 1e-12))).sum(-1)).mean()),
                router_probability_std=float(q.std(0).mean()), predicted_mode_1_fraction=float(q[:, 1].mean())))
            gene_means.append((transform.decode(mean[None])[0], gene_target[ti].mean(0)))
        pm, ym = np.stack([v[0] for v in gene_means]), np.stack([v[1] for v in gene_means])
        for row, error in zip(local, (((pm-pm.mean(0))-(ym-ym.mean(0)))**2).mean(1)):
            row['condition_centered_gene_mean_mse'] = float(error)
        rows.extend(local)
    conditions = sorted(set(target['conditions']))
    if conditions != ['LIN28A', 'NANOG', 'POU5F1', 'ZIC3']:
        raise ValueError('Cannot silently change the four-condition development panel')
    comparisons = []
    for metric in config['evaluation']['metrics']:
        for control in ('static', 'gfg_joint_shuffled'):
            result = paired_condition_bootstrap(rows, real_arm='gfg_joint', control_arm=control,
                seeds=config['seeds'], conditions=conditions, n_bootstrap=config['evaluation']['bootstrap'],
                seed=config['evaluation']['bootstrap_seed'], confidence=config['evaluation']['confidence'], metric=metric)
            comparisons.append(dict(metric=metric, **result))
    means = {arm: {metric: float(np.mean([row[metric] for row in rows if row['arm'] == arm]))
        for metric in config['evaluation']['metrics']} for arm in config['arms']}
    summary = dict(status='GFG_DIRECT_ROUTER_EVALUATED', means=means, comparisons=comparisons,
        all_primary_intervals_positive=all(r['ci_low'] > 0 for r in comparisons),
        same_expert_candidates_verified=True, confirmation='SEALED', formal_G1='NOT_RUN',
        scientific_scope='single_exploratory_followup_not_confirmatory')
    save_csv(run.directory/'metrics.csv', rows)
    save_csv(run.directory/'sampled_metrics.csv', sample_rows)
    save_csv(run.directory/'comparisons.csv', comparisons)
    save_json(run.directory/'summary.json', summary)
    (run.directory/'RESULTS.md').write_text('# GFG direct-distribution router follow-up\n\n'+json.dumps(summary, indent=2)
        +'\n\nAll three arms share identical frozen candidate endpoints. Exact-mixture energy removes sampling noise, '
         'but is a new prespecified follow-up metric, not a redefinition of the original pilot result. '
         'Sampled energy is also reported. No individual fate or real multimodal claim.\n')
print(json.dumps(summary))
