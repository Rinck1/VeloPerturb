"""Target-retrained, no-JVP RNA velocity for held-out RENGE day4 -> day5 groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from veloroute.artifacts import Run, load_config, save_csv, save_json, sha256
from veloroute.gpu_policy import enforce_gpu_policy
from veloroute.gfg_experiments import read_gene_input
from veloroute.latent import FrozenSplicingTransform, load_pack
from w3_gfg_v2_control import seeded_control
from w3_validation import local_u_control


def grouped_mean(values, conditions, wanted):
    return np.stack([np.asarray(values)[np.asarray(conditions) == c].mean(0) for c in wanted])


def ridge(seed):
    # Fixed alpha; no hyperparameter selection on held-out conditions.
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def infer(model, values, device, chunk=64):
    model.eval(); out = []
    with torch.no_grad():
        for start in range(0, len(values), chunk):
            gene_us = torch.as_tensor(values[start:start+chunk], dtype=torch.float32, device=device)
            u, s = gene_us.chunk(2, -1)
            un, sn = (u-model.u_mean)/model.u_std, (s-model.s_mean)/model.s_std
            innov = model.innovation(u, s)/model.u_std
            raw_ds = model.core.vel_head(torch.stack((un, sn, innov), -1)).squeeze(-1)
            v = (raw_ds/(1+s)) @ model.components.T / model.velocity_scale.clamp_min(1e-8)
            out.append(v.detach().cpu().numpy())
    result = np.concatenate(out)
    if not np.isfinite(result).all():
        raise ArithmeticError('Nonfinite no-JVP velocity estimate')
    return result


def fit_one(c, seed, output, device='cuda:0'):
    start = time.monotonic()
    fold = Path(c['fold']); gd = Path(c['gene_input_root'])
    train, tm = load_pack(fold / 'train_source.npz', expected_side='source')
    target, ttm = load_pack(fold / 'train_target.npz', expected_side='target')
    source_gene, _, gm = read_gene_input(gd / 'train_gene_source.npz', fold / 'train_source.npz')
    if len(train['cell_ids']) != len(source_gene) or len(train['cell_ids']) != len(train['conditions']):
        raise ValueError('Train source and gene-input rows differ')
    if gm['transform_hash'] != tm['transform_hash'] or ttm['transform_hash'] != tm['transform_hash']:
        raise ValueError('Train fold transforms differ')
    conditions = sorted(set(train['conditions']))
    target_conditions = set(target['conditions'])
    train_tf = conditions
    if len(train_tf) != 14 or set(train_tf) != target_conditions:
        raise ValueError('Unexpected registered RENGE training-condition cohort')
    if set(train_tf) != target_conditions:
        raise ValueError('Training target/source condition mismatch')

    vtr = source_gene.astype(np.float64)
    vtrsh, _, shuffle_audit = local_u_control(vtr, train['z'], train['conditions'],
        seed=seed + 20261015, neighbors=10)
    transform = FrozenSplicingTransform.load(fold / 'transform.npz')
    comps = transform.components
    model = seeded_control(vtr.shape[1] // 2, state_dim=comps.shape[0], seed=seed)
    device = torch.device(device); model.to(device)
    ids = train['cell_ids'].astype(str).tolist()
    model.fit_source(torch.as_tensor(vtr, dtype=torch.float32, device=device), cell_ids=ids, allowed_ids=set(ids))
    rng = np.random.default_rng(seed)
    tx = torch.as_tensor(vtr, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=c['learning_rate'])
    loss_trace = []
    for step in range(c['training_steps']):
        ix = torch.as_tensor(rng.integers(len(tx), size=c['batch_size']), dtype=torch.long, device=device)
        u, s = tx[ix].chunk(2, -1)
        un, sn = (u-model.u_mean)/model.u_std, (s-model.s_mean)/model.s_std
        innov = model.innovation(u, s)/model.u_std
        raw_ds = model.core.vel_head(torch.stack((un, sn, innov), -1)).squeeze(-1)
        beta, gamma = model.core.log_beta.exp(), model.core.log_gamma.exp()
        dyn = ((raw_ds-(beta*u-gamma*s))/(1+s)).square().mean()
        anchor = ((model.core.log_gamma-model.core.log_beta)-model.gamma_ss.log()).square().mean()
        loss = 20*dyn + 50*anchor
        if not torch.isfinite(loss):
            raise ArithmeticError(f'Nonfinite native no-JVP training loss at step {step}')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        loss_trace.append(float(loss.detach()))
    vtrain = infer(model, vtr, device)
    vtrain_shuf = infer(model, vtrsh, device)

    # A simple frozen kinetic-rate reference, estimated on the training source only.
    ng = vtr.shape[1] // 2
    sn, un = vtr[:, ng:], vtr[:, :ng]
    gamma = transform.gamma
    raw_kin_train = np.where(transform.supported, (un - sn * gamma) / (1 + sn), 0.) @ comps.T
    scale = max(float(np.sqrt(np.mean(raw_kin_train**2))), 1e-8)
    vk_train = raw_kin_train / scale

    z_src = {condition: train['z'][train['conditions'] == condition].mean(0) for condition in train_tf}
    z_tgt = {condition: target['z'][target['conditions'] == condition].mean(0) for condition in train_tf}
    v_real = {condition: vtrain[train['conditions'] == condition].mean(0) for condition in train_tf}
    v_shuf = {condition: vtrain_shuf[train['conditions'] == condition].mean(0) for condition in train_tf}
    vk_group = {condition: vk_train[train['conditions'] == condition].mean(0) for condition in train_tf}
    # Predict the complete adjacent-time group displacement. The ridge intercept
    # learns shared temporal drift from training TFs; no held-out future data or
    # unavailable control condition is needed.
    y_train = np.stack([z_tgt[c] - z_src[c] for c in train_tf])
    xz = np.stack([z_src[c] for c in train_tf])
    xv = np.stack([v_real[c] for c in train_tf])
    xvs = np.stack([v_shuf[c] for c in train_tf])
    xvk = np.stack([vk_group[c] for c in train_tf])
    models = {
        'S_only': ridge(seed).fit(xz, y_train),
        'target_retrained_no_JVP_velocity': ridge(seed).fit(np.c_[xz, xv], y_train),
        'same_model_U_shuffled': ridge(seed).fit(np.c_[xz, xvs], y_train),
        'target_retrained_kinetic_reference': ridge(seed).fit(np.c_[xz, xvk], y_train),
    }
    # Leave-one-training-TF-out estimates are reported as internal diagnostics.
    cv_rows = []
    for held in range(len(train_tf)):
        take = np.arange(len(train_tf)) != held
        for arm, features in [('S_only', xz), ('target_retrained_no_JVP_velocity', np.c_[xz, xv]),
                              ('same_model_U_shuffled', np.c_[xz, xvs]),
                              ('target_retrained_kinetic_reference', np.c_[xz, xvk])]:
            pred = ridge(seed).fit(features[take], y_train[take]).predict(features[~take])[0]
            cv_rows.append(dict(heldout_training_TF=train_tf[held], arm=arm,
                                mean_squared_error=float(np.mean((pred-y_train[held])**2))))

    # Validation inputs are source-only. Do not read validation_target in this stage.
    val, vm = load_pack(fold / 'validation_source.npz', expected_side='source')
    vva, _, vgm = read_gene_input(gd / 'validation_gene_source.npz', fold / 'validation_source.npz')
    if vgm['transform_hash'] != vm['transform_hash'] or vm['transform_hash'] != tm['transform_hash']:
        raise ValueError('Validation and training transforms are not identical')
    if set(val['conditions']) != {'LIN28A', 'NANOG', 'POU5F1', 'ZIC3'}:
        raise ValueError('Unexpected validation conditions')
    vva_shuf, _, val_shuffle_audit = local_u_control(vva.astype(np.float64), val['z'], val['conditions'],
        seed=seed + 20261015, neighbors=10)
    vval = infer(model, vva, device); vval_shuf = infer(model, vva_shuf, device)
    sv = {condition: val['z'][val['conditions'] == condition].mean(0)
          for condition in sorted(set(val['conditions']))}
    vv = {condition: vval[val['conditions'] == condition].mean(0) for condition in sv}
    vvs = {condition: vval_shuf[val['conditions'] == condition].mean(0) for condition in sv}
    snv, unv = vva[:, ng:], vva[:, :ng]
    vkin_val_gene = np.where(transform.supported, (unv - snv * gamma) / (1 + snv), 0.)
    # Project each gene velocity with the frozen source PCA and velocity scale.
    vk_val = vkin_val_gene @ comps.T / scale
    vkv = {condition: vk_val[val['conditions'] == condition].mean(0) for condition in sv}
    pred = {}
    for arm, est in models.items():
        entries = []
        for condition in sorted(sv):
            base = sv[condition]
            if arm == 'S_only': features = base[None]
            elif arm == 'target_retrained_no_JVP_velocity': features = np.r_[base, vv[condition]][None]
            elif arm == 'same_model_U_shuffled': features = np.r_[base, vvs[condition]][None]
            else: features = np.r_[base, vkv[condition]][None]
            delta = est.predict(features)[0]
            entries.append(dict(condition=condition, source_mean=base.tolist(), predicted_delta=delta.tolist(),
                predicted_future_mean=(base + delta).tolist()))
        pred[arm] = entries
    destination = Path(c['artifact_root']) / f'seed{seed}_v1'
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / 'frozen_source_predictions.npz',
        **{arm+'_predicted_future_mean': np.asarray([r['predicted_future_mean'] for r in pred[arm]]) for arm in pred},
        conditions=np.asarray(sorted(sv), dtype=str), arms=np.asarray(list(pred), dtype=str))
    torch.save(model.state_dict(), destination / 'no_jvp_model.pt')
    save_json(destination / 'frozen_prediction_manifest.json', dict(status='frozen_source_only', seed=seed,
        validation_target_read=False, future_target_read=False, validation_conditions=sorted(sv), arms=list(pred),
        source_file='frozen_source_predictions.npz', source_sha256=sha256(destination / 'frozen_source_predictions.npz'),
        training_source_sha256=sha256(gd / 'train_gene_source.npz'),
        training_target_sha256=sha256(fold / 'train_target.npz'),
        validation_source_sha256=sha256(gd / 'validation_gene_source.npz'),
        model_sha256=sha256(destination / 'no_jvp_model.pt'),
        protocol_hash=sha256(Path('configs/w3_7_short_horizon_no_jvp_20261015.json'))))
    run_output = Path(output).with_name(Path(output).name + '_training_run')
    with Run(run_output, stage='W3_7_no_JVP_target_retrain_source_predictions', kind='development',
             config=dict(c, seed=seed, source_only_prediction=True), seed=seed,
             inputs=[Path('configs/w3_7_short_horizon_no_jvp_20261015.json'), Path(__file__),
                     Path('scripts/phase0/w3_gfg_v2_control.py'), Path('scripts/phase0/w3_gfg_v2.py'),
                     Path('scripts/phase0/w3_validation.py'), fold/'train_source.npz', fold/'train_target.npz',
                     gd/'train_gene_source.npz', fold/'validation_source.npz', gd/'validation_gene_source.npz',
                     fold/'transform.npz']) as run:
        result = dict(status='FROZEN_SOURCE_ONLY_PREDICTIONS', dataset='RENGE_day4_to_day5', seed=seed,
            training_conditions=train_tf, training_cells=len(train['cell_ids']), validation_cells=len(val['cell_ids']),
            steps=c['training_steps'], batch_size=c['batch_size'], final_loss=float(np.mean(loss_trace[-100:])),
            arms=list(pred), condition_level_LOTO= cv_rows,
            mean_LOTO_mse={arm: float(np.mean([r['mean_squared_error'] for r in cv_rows if r['arm']==arm])) for arm in pred},
            source_only_prediction=True, validation_target_read=False,
            U_shuffle_train=shuffle_audit, U_shuffle_validation=val_shuffle_audit,
            velocity_sensitivity_to_U={arm: None for arm in pred},
            model='GFGv2 no-JVP shallow velocity head only; target-domain kinetic-loss fit; beta=1 and training-fold gamma/innovation frozen; encoder/attention/decoder bypassed',
            model_parameter_count=sum(p.numel() for p in model.parameters()),
            runtime_seconds=time.monotonic()-start)
        # Report seed-conditional whole-query direction sensitivity as a mechanism audit.
        result['velocity_sensitivity_to_U'] = {arm: float(np.median(np.sum(vval * vval_shuf, axis=1) /
            np.maximum(np.linalg.norm(vval,axis=1)*np.linalg.norm(vval_shuf,axis=1),1e-12)))
            for arm, vval, vval_shuf in [('validation', vval, vval_shuf)]}
        save_csv(run.directory/'training_LOTO.csv', cv_rows)
        save_json(run.directory/'summary.json', result)
        save_json(run.directory/'status.json', dict(status='complete'))
        (run.directory/'RESULTS.md').write_text('# RENGE no-JVP short-horizon source predictions\n\n'+json.dumps(result,indent=2)+'\n')
        print(json.dumps(result), flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/w3_7_short_horizon_no_jvp_20261015.json')
    ap.add_argument('--seed',type=int,required=True); ap.add_argument('--output',required=True)
    args=ap.parse_args(); c=load_config(args.config); enforce_gpu_policy('cuda:0')
    if args.seed not in c['seeds']: raise ValueError('Unregistered seed')
    start=time.monotonic(); fit_one(c,args.seed,args.output)


if __name__=='__main__': main()
