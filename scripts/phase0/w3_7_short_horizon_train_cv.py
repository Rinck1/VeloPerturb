"""Train-fold leave-one-TF-out validation for frozen no-JVP velocity heads."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from veloroute.artifacts import Run, load_config, save_csv, save_json
from veloroute.gfg_experiments import read_gene_input
from veloroute.latent import FrozenSplicingTransform, load_pack
from w3_7_short_horizon_no_jvp import infer, ridge
from w3_gfg_v2_control import seeded_control
from w3_validation import local_u_control


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/w3_7_short_horizon_no_jvp_20261015.json')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    c = load_config(args.config)
    fold, gene_root = Path(c['fold']), Path(c['gene_input_root'])
    source, sm = load_pack(fold/'train_source.npz', expected_side='source')
    target, tm = load_pack(fold/'train_target.npz', expected_side='target')
    gene_us, _, gm = read_gene_input(gene_root/'train_gene_source.npz', fold/'train_source.npz')
    transform = FrozenSplicingTransform.load(fold/'transform.npz')
    conditions = sorted(set(source['conditions']))
    if len(conditions) != 14 or set(conditions) != set(target['conditions']):
        raise ValueError('Unexpected training TF cohort')

    rows = []
    for seed in c['seeds']:
        model_path = Path(c['artifact_root'])/f'seed{seed}_v1'/'no_jvp_model.pt'
        model = seeded_control(gene_us.shape[1]//2, state_dim=transform.components.shape[0], seed=seed)
        state = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        model.eval()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        v_real = infer(model, gene_us.astype(np.float32), device)
        us_shuffled, _, _ = local_u_control(gene_us.astype(np.float64), source['z'], source['conditions'],
                                             seed=seed+20261015, neighbors=10)
        v_shuffle = infer(model, us_shuffled.astype(np.float32), device)

        n_genes = gene_us.shape[1]//2
        u, s = gene_us[:, :n_genes], gene_us[:, n_genes:]
        gamma = transform.gamma
        vk = np.where(transform.supported, (u-s*gamma)/(1+s), 0.) @ transform.components.T
        vk /= max(float(np.sqrt(np.mean(vk**2))), 1e-8)
        state_mean = {ct: source['z'][source['conditions']==ct].mean(0) for ct in conditions}
        target_mean = {ct: target['z'][target['conditions']==ct].mean(0) for ct in conditions}
        velocity_mean = {ct: v_real[source['conditions']==ct].mean(0) for ct in conditions}
        shuffled_mean = {ct: v_shuffle[source['conditions']==ct].mean(0) for ct in conditions}
        kinetic_mean = {ct: vk[source['conditions']==ct].mean(0) for ct in conditions}
        x_state = np.stack([state_mean[ct] for ct in conditions])
        x_real = np.c_[x_state, np.stack([velocity_mean[ct] for ct in conditions])]
        x_shuf = np.c_[x_state, np.stack([shuffled_mean[ct] for ct in conditions])]
        x_kin = np.c_[x_state, np.stack([kinetic_mean[ct] for ct in conditions])]
        y = np.stack([target_mean[ct]-state_mean[ct] for ct in conditions])
        features = {'S_only': x_state, 'target_retrained_no_JVP_velocity': x_real,
                    'same_model_U_shuffled': x_shuf, 'target_retrained_kinetic_reference': x_kin}
        for held, condition in enumerate(conditions):
            keep = np.arange(len(conditions)) != held
            for arm, x in features.items():
                prediction = ridge(seed).fit(x[keep], y[keep]).predict(x[~keep])[0]
                rows.append(dict(seed=seed, heldout_training_TF=condition, arm=arm,
                                 group_mean_mse=float(np.mean((prediction-y[held])**2)),
                                 heldout_training_cells=int(np.sum(target['conditions']==condition))))

    arms = sorted(set(row['arm'] for row in rows))
    summaries = []
    keyed = {(row['seed'], row['heldout_training_TF'], row['arm']): row['group_mean_mse'] for row in rows}
    for arm in arms:
        summaries.append(dict(arm=arm, mean_LOTO_mse=float(np.mean([r['group_mean_mse'] for r in rows if r['arm']==arm])),
                              n_TFs=len(conditions), n_seeds=len(c['seeds'])))
    contrasts = []
    for arm in arms:
        if arm == 'S_only':
            continue
        diffs = [keyed[(seed, condition, 'S_only')]-keyed[(seed, condition, arm)]
                 for seed in c['seeds'] for condition in conditions]
        contrasts.append(dict(arm=arm, improvement_vs_S_only=float(np.mean(diffs)),
                              positive_TF_fraction=float(np.mean(np.asarray(diffs)>0)),
                              n_TF_seed_pairs=len(diffs)))

    with Run(args.output, stage='W3_7_no_JVP_trainfold_LOTO', kind='development', config=c,
             inputs=[Path(args.config), fold/'train_source.npz', fold/'train_target.npz',
                     gene_root/'train_gene_source.npz', fold/'transform.npz',
                     *[Path(c['artifact_root'])/f'seed{s}_v1'/'no_jvp_model.pt' for s in c['seeds']]]) as run:
        save_csv(run.directory/'condition_seed_metrics.csv', rows)
        save_csv(run.directory/'arm_summary.csv', summaries)
        result = dict(status='TRAIN_FOLD_LOTO_COMPLETE', dataset='RENGE_day4_to_day5_trainfold',
                      arms=summaries, contrasts=contrasts, heldout_target_opened=False,
                      independent_unit='TF condition, not cells', shuffle_control='local U permutation within condition and S-depth strata')
        save_json(run.directory/'summary.json', result)
        (run.directory/'RESULTS.md').write_text('# RENGE training-fold LOTO\n\n'+str(result)+'\n')
        print(result, flush=True)


if __name__ == '__main__':
    main()
