"""Exploratory gain mainline: full GFG/router instrument on packaged-S/U folds.

Per fold: train static / gfg_joint / gfg_joint_shuffled arms (full budget, 8 experts,
gate+EM), predict held-out conditions, evaluate energy distance per condition.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from veloroute.artifacts import load_config, save_json, sha256, object_hash
from veloroute.full_experiments import condition_token_sets
from veloroute.full_model import load_full_checkpoint
from veloroute.gfg_experiments import read_gene_input, train_gfg, training_gene_us
from veloroute.gpu_policy import enforce_gpu_policy
from veloroute.latent import load_pack
from veloroute.metrics import energy_distance, paired_condition_bootstrap
from veloroute.velocity_background import load_background

AUDIT_REVISION = 'distribution_and_static_audit_20260921_v2'


def build_config(fold, seed, device, coupling='ot', conditions_path=None, background=None):
    base = load_config('configs/veloroute_gfg_joint_20260914.yaml')
    kang = load_config('configs/veloroute_kang_20260914.yaml')
    t = kang['training']
    if sha256(t['gfg_checkpoint']) != t['gfg_checkpoint_sha256']:
        raise ValueError('Registered GFG checkpoint changed')
    conditions_path = conditions_path or str(Path(fold)/'conditions.npz')
    with np.load(conditions_path, allow_pickle=False) as data:
        esm_dim = int(data['embeddings'].shape[1])
    base.update(fold=str(fold), source_counts=None,
        implementation_revision=AUDIT_REVISION,
        conditions=conditions_path,
        gfg_checkpoint=t['gfg_checkpoint'],
        research_status='exploratory_gain_mainline_not_formal_G1',
        cpu_threads=4, device=device)
    base['model'].update(esm_dim=esm_dim, hidden_dim=768, residual_blocks=6, max_experts=8,
        top_k=2, initial_active=1, use_adaptive_modes=True, use_intrinsic=True, use_gate=True,
        use_noise=True, support_start=0., support_end=1., gfg_genes=2000, state_dim=50)
    base['training'].update(stage_a_steps=300, stage_b_steps=2000, stage_c_steps=12000,
        batch_size=8, time_start=0., time_end=1., seed=seed, lambda_dynamic=.1, lambda_field=.01,
        checkpoint_interval=2000, reference_interval=2000, e_interval=300, usage_interval=2000)
    if coupling != 'ot':
        base['training'].update(coupling_mode=coupling)
    base['training'].update(velocity_background=background or '')
    return base


def predict_arm(config, fold, checkpoint, arm, seed, output, gene_dir=None, background_path=None):
    device = config['device']
    enforce_gpu_policy(device)
    torch.set_num_threads(4)
    gene_root = Path(gene_dir) if gene_dir else Path(fold)
    values, source, sm = read_gene_input(gene_root/'validation_gene_source.npz',
                                         Path(fold)/'validation_source.npz')
    model, metadata = load_full_checkpoint(checkpoint, map_location=device)
    if metadata['transform_hash'] != sm['transform_hash'] or metadata['seed'] != seed:
        raise ValueError('checkpoint/fold mismatch')
    values = training_gene_us(values, source, arm, seed)
    background = None
    if background_path:
        data = load_background(background_path, 'validation', cell_ids=source['cell_ids'])
        background = tuple(torch.as_tensor(data[key], dtype=torch.float32, device=device) for key in ('r', 'v'))
    tokens, mask, _ = condition_token_sets(config['conditions'], source['conditions'])
    generator = torch.Generator(device=device).manual_seed(seed)
    zs, qs = [], []
    with torch.no_grad():
        for start in range(0, len(values), 8):
            ix = slice(start, start+8)
            z, us, token = [torch.tensor(x[ix], dtype=torch.float32, device=device)
                            for x in (source['z'], values, tokens)]
            padding = torch.tensor(mask[ix], dtype=torch.bool, device=device)
            bg = None
            if background is not None:
                bg = (background[0][ix], background[1][ix])
            prediction = model.predict(z, us, token, padding, t0=0., t1=1., max_step=.25,
                                       generator=generator, stochastic=True, background=bg)
            zs.append(prediction.endpoint.cpu().numpy())
            qs.append(prediction.probabilities.cpu().numpy())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'predictions.npz', z=np.concatenate(zs), q=np.concatenate(qs),
                        conditions=source['conditions'], source_ids=source['cell_ids'], seed=np.asarray(seed))
    return out/'predictions.npz'


def evaluate(fold, predictions, tag, *, seed):
    target, tm = load_pack(Path(fold)/'validation_target.npz', expected_side='target')
    with np.load(predictions, allow_pickle=False) as data:
        z, conditions = data['z'], data['conditions']
        if 'seed' not in data or int(data['seed']) != seed:
            raise ValueError('Prediction/evaluation seed mismatch')
    rows = []
    for condition in sorted(set(conditions)):
        p = z[conditions == condition]
        y = target['z'][target['conditions'] == condition]
        rows.append({'condition': condition, 'seed': seed, 'arm': tag,
                     'energy_distance': energy_distance(p, y),
                     'variance_ratio': float(p.var(0).sum()/max(y.var(0).sum(), 1e-12))})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--arms', default='static,gfg_joint,gfg_joint_shuffled')
    parser.add_argument('--coupling', default='ot', choices=['ot', 'distribution_matching'])
    parser.add_argument('--gene-dir', default=None)
    parser.add_argument('--conditions', default=None)
    parser.add_argument('--background', default=None)
    args = parser.parse_args()
    fold = Path(args.fold)
    out_root = Path(args.output)
    # Revised semantics must never silently reuse a legacy checkpoint or overwrite
    # predictions produced under a different gradient/baseline protocol.
    out_root.mkdir(parents=True, exist_ok=False)
    config = build_config(fold, args.seed, 'cuda', coupling=args.coupling, conditions_path=args.conditions, background=args.background)
    enforce_gpu_policy('cuda')
    arms = args.arms.split(',')
    if len(set(arms)) != len(arms) or not set(arms) <= {'static', 'gfg_joint', 'gfg_joint_shuffled', 'gfg_frozen'}:
        raise ValueError('Unknown or duplicate experiment arms')
    rows = []
    frozen = []
    for arm in arms:
        training = out_root/arm/'train'
        log_path = out_root/arm/'train.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('x') as stream:
            import contextlib
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                train_gfg(config, args.gene_dir or fold, training, arm=arm, seed=args.seed, pilot=False)
        checkpoint = training/'model.pt'
        predictions = predict_arm(config, fold, checkpoint, arm, args.seed, out_root/arm/'predict', gene_dir=args.gene_dir, background_path=args.background)
        frozen.append(dict(arm=arm, path=str(predictions), sha256=sha256(predictions)))
        print(json.dumps(dict(arm=arm, done=True)), flush=True)
    save_json(out_root/'prediction_freeze.json', dict(implementation_revision=AUDIT_REVISION,
        config_hash=object_hash(config), seed=args.seed, future_target_read=False, predictions=frozen))
    for record in frozen:
        if sha256(record['path']) != record['sha256']:
            raise ValueError('Frozen prediction changed before evaluation')
        rows.extend(evaluate(fold, record['path'], record['arm'], seed=args.seed))
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.to_csv(out_root/'metrics.csv', index=False)
    summary_rows = frame.groupby('arm')[['energy_distance', 'variance_ratio']].mean()
    comparisons = []
    conditions = sorted(set(frame['condition']))
    if len(conditions) >= 2:
        for real, control in [('gfg_joint', 'static'), ('gfg_joint', 'gfg_joint_shuffled'),
                              ('static', 'gfg_joint_shuffled')]:
            if not {real, control} <= set(arms):
                continue
            comparisons.append(paired_condition_bootstrap(
                rows, real_arm=real, control_arm=control, seeds=[args.seed],
                conditions=conditions, n_bootstrap=5000, seed=20260918))
    result = dict(status='EXPLORATORY_GAIN_MAINLINE_COMPLETE', fold=str(fold), tag=args.tag,
        seed=args.seed, arms=arms, per_arm=summary_rows.round(4).to_dict('index'),
        comparisons=comparisons, formal_G1='NOT_RUN', confirmation='SEALED',
        individual_fate_claim=False, implementation_revision=AUDIT_REVISION)
    (out_root/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
