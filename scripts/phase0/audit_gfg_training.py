"""Training-source-only velocity collapse, U sensitivity, and expert diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from veloroute.artifacts import Run, load_config, save_csv, save_json
from veloroute.full_experiments import condition_token_sets
from veloroute.full_model import load_full_checkpoint
from veloroute.gfg_experiments import read_gene_input, training_gene_us

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
config = load_config('configs/veloroute_gfg_joint_20260914.yaml')
fold = Path(config['fold'])
gene_path = Path('outputs/veloroute_gfg_inputs_20260914/train_gene_source.npz')
values, source, _ = read_gene_input(gene_path, fold/'train_source.npz')
model, metadata = load_full_checkpoint(args.checkpoint, map_location='cuda')
if model.config.dynamics_backend != 'gfg':
    raise ValueError('Expected actual GFG checkpoint')
model.eval()
torch.set_num_threads(4)
indices = np.concatenate([np.flatnonzero(source['conditions'] == c)[:8] for c in sorted(set(source['conditions']))])
tokens, mask, _ = condition_token_sets(config['conditions'], source['conditions'][indices])
shuffled = training_gene_us(values, source, 'gfg_joint_shuffled', 20260914)
z, us, bad_us, protein = [torch.tensor(v, dtype=torch.float32, device='cuda')
    for v in (source['z'][indices], values[indices], shuffled[indices], tokens)]
padding = torch.tensor(mask, dtype=torch.bool, device='cuda')
with Run(args.output, stage='GFG_training_source_quality', kind='engineering',
    config=dict(seed=20260914, cells_per_TF=8, selection='first_source_barcodes',
        no_future_targets=True, reference='initial_checkpoint_not_velocity_truth'),
    inputs=[args.checkpoint, gene_path, config['gfg_checkpoint']], seed=20260914) as run:
    velocity, bad_velocity, probabilities, bad_probabilities, spread, errors = [], [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(indices), 8):
            ix = slice(start, start+8)
            context = model.make_context(z[ix], us[ix], protein[ix], padding[ix])
            velocity.append(context.velocity.cpu())
            probabilities.append(context.probabilities.cpu())
            errors.append(float(model.dynamics.last_native_loss))
            field = model.field(context.z0, 4., context.condition)[:, model.active_modes]
            spread.append(float((field-field.mean(1, keepdim=True)).square().mean().sqrt()/field.square().mean().sqrt().clamp_min(1e-8)))
            bad = model.make_context(z[ix], bad_us[ix], protein[ix], padding[ix])
            bad_velocity.append(bad.velocity.cpu())
            bad_probabilities.append(bad.probabilities.cpu())
        trained = torch.cat(velocity)
        bad = torch.cat(bad_velocity)
        q, qb = torch.cat(probabilities), torch.cat(bad_probabilities)
        model.dynamics.core.load_pretrained(config['gfg_checkpoint'])
        initial = torch.cat([model.dynamics(z[i:i+8], us[i:i+8])[1].cpu() for i in range(0, len(z), 8)])
    eigenvalues = torch.linalg.eigvalsh(torch.cov(trained.T)).clamp_min(0)
    spectral_probability = eigenvalues/eigenvalues.sum().clamp_min(1e-12)
    rank = (-(spectral_probability*spectral_probability.clamp_min(1e-12).log()).sum()).exp()
    metrics = dict(cells=len(indices), trained_velocity_norm_mean=float(trained.norm(dim=-1).mean()),
        initial_velocity_norm_mean=float(initial.norm(dim=-1).mean()),
        trained_velocity_across_cell_std=float(trained.std(0).square().mean().sqrt()),
        initial_trained_cosine_median=float(torch.nn.functional.cosine_similarity(initial, trained).median()),
        velocity_effective_covariance_rank=float(rank),
        conditional_U_shuffle_velocity_relative_RMS=float((trained-bad).square().mean().sqrt()/trained.square().mean().sqrt().clamp_min(1e-8)),
        conditional_U_shuffle_probability_MAE=float((q-qb).abs().mean()),
        mode_entropy_mean=float((-(q*q.clamp_min(1e-12).log()).sum(-1)).mean()),
        probability_across_cell_std=float(q.std(0).mean()),
        expert_relative_field_spread_mean=float(np.mean(spread)),
        native_loss_mean=float(np.mean(errors)))
    result = dict(status='GFG_TRAINING_SOURCE_AUDIT_COMPLETE', metrics=metrics,
        true_velocity_accuracy='UNKNOWN', final_confirmation_read=False,
        limits=['Nonzero velocity or sensitivity is not useful task information.',
                'Original pretraining direction is not independent truth.',
                'Condition-balanced 112-cell engineering sample, not all-cell inference.'])
    save_csv(run.directory/'metrics.csv', [metrics])
    save_json(run.directory/'summary.json', result)
    (run.directory/'RESULTS.md').write_text('# GFG training-source diagnostics\n\n'+json.dumps(result, indent=2))
print(json.dumps(result))
