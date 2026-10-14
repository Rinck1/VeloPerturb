"""Retrain no-JVP kinetic GFGv2 and freeze short-time population predictions.

All states, velocity derivatives and endpoints use the SAME existing training
PCA. No validation target is read by this prediction script. Population U
controls exchange populations, not cells within a population.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from veloroute.artifacts import Run, load_config, save_csv, save_json, sha256
from veloroute.experiments import condition_vectors, validate_pair
from veloroute.gfg_experiments import read_gene_input
from veloroute.gpu_policy import enforce_gpu_policy
from veloroute.latent import FrozenSplicingTransform, load_pack
from w3_gfg_v2_control import seeded_control
from w3_validation import LogStateProjection, local_u_control


def population_means(x, conditions, labels):
    result = []
    for label in labels:
        mask = np.asarray(conditions) == label
        if not mask.any():
            raise ValueError('Missing registered source population')
        result.append(np.asarray(x)[mask].mean(0))
    return np.stack(result)


def fit_ridge_training_cv(x, y, grid):
    """One row per training TF. Scaling fitted anew in every CV fold."""
    if len(x) < 5 or len(x) != len(y):
        raise ValueError('Insufficient independent training TFs')
    scores = []
    for alpha in grid:
        errors = []
        for i in range(len(x)):
            keep = np.arange(len(x)) != i
            scaler = StandardScaler().fit(x[keep])
            ridge = Ridge(alpha=alpha).fit(scaler.transform(x[keep]), y[keep])
            errors.append(float(np.square(ridge.predict(scaler.transform(x[i:i+1]))[0] - y[i]).mean()))
        scores.append(float(np.mean(errors)))
    best = grid[int(np.argmin(scores))]
    scaler = StandardScaler().fit(x)
    return scaler, Ridge(alpha=best).fit(scaler.transform(x), y), dict(alpha=best, training_TF_cv_mse=scores)


def assert_shared_geometry(values, source, projection):
    genes = values.shape[1] // 2
    np.testing.assert_allclose(projection.state(values[:, genes:]), source['z'], rtol=2e-4, atol=2e-4,
                               err_msg='Source and target must share the frozen original training PCA')


def fit_predictions(c, seed, output):
    enforce_gpu_policy('cuda:0'); start = time.monotonic()
    fold, inputs = Path(c['fold']), Path(c['gene_inputs'])
    us_tr, train, sm = read_gene_input(inputs / 'train_gene_source.npz', fold / 'train_source.npz')
    us_va, val, vm = read_gene_input(inputs / 'validation_gene_source.npz', fold / 'validation_source.npz')
    target, tm = load_pack(fold / 'train_target.npz', expected_side='target')
    validate_pair(train, sm, target, tm, 'train')
    if sm['transform_hash'] != vm['transform_hash'] or set(train['conditions']) & set(val['conditions']):
        raise ValueError('Outer heldout TF / geometry mismatch')
    frozen = FrozenSplicingTransform.load(fold / 'transform.npz')
    if frozen.fit_ids_hash != sm['fit_ids_hash']:
        raise ValueError('PCA fitted outside the registered source scope')
    projection = LogStateProjection(frozen.mean.copy(), frozen.components.copy(), 1.)
    assert_shared_geometry(us_tr, train, projection); assert_shared_geometry(us_va, val, projection)
    genes = us_tr.shape[1] // 2
    inputs_list = [Path(__file__), Path(__file__).with_name('w3_gfg_v2_control.py'),
        Path(__file__).with_name('w3_validation.py'), fold / 'transform.npz', fold / 'train_source.npz',
        fold / 'train_target.npz', fold / 'validation_source.npz', inputs / 'train_gene_source.npz',
        inputs / 'validation_gene_source.npz', Path(c['condition_embeddings'])]
    with Run(output, stage='shorttime_no_JVP_train_and_freeze', kind='development', config=c,
             inputs=inputs_list, seed=seed) as run:
        # Seed before construction, train-only rate/innovation fit, fixed PCA.
        model = seeded_control(genes, seed=seed, state_dim=50, dim=c['dim'], codes=c['codes'], hidden=tuple(c['hidden'])).to('cuda:0')
        model.fit_source(torch.tensor(us_tr), cell_ids=list(train['cell_ids']), allowed_ids=set(train['cell_ids']))
        with torch.no_grad():
            model.components.copy_(torch.tensor(frozen.components, dtype=torch.float32, device='cuda:0'))
            model.state_log_mean.copy_(torch.tensor(frozen.mean, dtype=torch.float32, device='cuda:0'))
            model.velocity_scale.fill_(1.)
        data = torch.tensor(us_tr, device='cuda:0')
        opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=c['lr'])
        rows = []
        for step in range(c['steps']):
            batch = data[torch.randint(0, len(data), (c['batch'],), device='cuda:0')]
            u, s = batch.chunk(2, -1)
            losses = model.core(u, s, model.u_mean, model.u_std, model.s_mean, model.s_std,
                model.innovation(u, s), model.gamma_ss)
            opt.zero_grad(); losses['native'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            if not torch.isfinite(losses['native']):
                raise ArithmeticError('Nonfinite no-JVP source loss')
            if (step + 1) % 100 == 0:
                row = dict(step=step+1, source_loss=float(losses['native'].detach()),
                           dynamics_loss=float(losses['dyn_loss']))
                rows.append(row); save_json(run.directory / 'status.json', dict(status='training', **row))
                print(json.dumps(row), flush=True)
        model.eval()
        torch.save(dict(state_dict=model.state_dict(), config=c, seed=seed, no_JVP=True,
                        condition_input=False, source_only=True), run.directory / 'estimator.pt')

        def velocity(values):
            values = np.asarray(values, dtype=np.float32)
            result = []
            with torch.no_grad():
                for i in range(0, len(values), 128):
                    u, s = torch.tensor(values[i:i+128], device='cuda:0').chunk(2, -1)
                    features = torch.stack(((u-model.u_mean)/model.u_std, (s-model.s_mean)/model.s_std,
                                            model.innovation(u, s)/model.u_std), -1)
                    rates = model.core.vel_head(features).squeeze(-1)
                    result.append(((rates/(1+s)) @ model.components.T).cpu().numpy())
            return np.concatenate(result)

        train_labels = np.array(sorted(set(train['conditions'])))
        val_labels = np.array(sorted(set(val['conditions'])))
        if (len(train_labels), len(val_labels)) != (14, 4):
            raise ValueError('Unexpected registered TF coverage')
        mean_tr = population_means(us_tr, train['conditions'], train_labels)
        mean_va = population_means(us_va, val['conditions'], val_labels)
        ztr = population_means(train['z'], train['conditions'], train_labels)
        zva = population_means(val['z'], val['conditions'], val_labels)
        # kNN control geometry is source group mean expression; estimator is c-blind.
        shuffled_tr, _, atr = local_u_control(mean_tr, ztr, np.repeat('source_day4', len(ztr)),
            seed=seed, neighbors=c['group_shuffle_neighbors'], match_depth=True)
        shuffled_va, _, ava = local_u_control(mean_va, zva, np.repeat('source_day4', len(zva)),
            seed=seed, neighbors=c['group_shuffle_neighbors'], match_depth=True)
        gamma = model.gamma_ss.cpu().numpy()
        vp_tr, vp_va = velocity(mean_tr), velocity(mean_va)
        vt, ve = velocity(us_tr), velocity(us_va)
        analytic_tr = projection.velocity(mean_tr[:, genes:], mean_tr[:, :genes]-gamma*mean_tr[:, genes:])
        analytic_va = projection.velocity(mean_va[:, genes:], mean_va[:, :genes]-gamma*mean_va[:, genes:])
        arms = dict(static=(None, None), gfg_population=(vp_tr, vp_va),
            gfg_mean_cell_velocity=(population_means(vt, train['conditions'], train_labels),
                                    population_means(ve, val['conditions'], val_labels)),
            gfg_population_shuffled_U=(velocity(shuffled_tr), velocity(shuffled_va)),
            analytic_population=(analytic_tr, analytic_va))
        et, _ = condition_vectors(c['condition_embeddings'], train_labels)
        ev, _ = condition_vectors(c['condition_embeddings'], val_labels)
        epca = PCA(c['embedding_dimensions'], svd_solver='full').fit(et)
        et, ev = epca.transform(et), epca.transform(ev)
        target_means = population_means(target['z'], target['conditions'], train_labels)
        displacement = target_means - ztr
        predictions = {}; selections = {}
        for arm in c['arms']:
            a, b = arms[arm]
            # All arms see the same source state and ESM condition information.
            x, xe = np.concatenate((ztr, et), 1), np.concatenate((zva, ev), 1)
            if a is not None:
                x, xe = np.concatenate((x, a), 1), np.concatenate((xe, b), 1)
            scaler, ridge, selection = fit_ridge_training_cv(x, displacement, c['ridge_grid'])
            predictions[arm] = zva + ridge.predict(scaler.transform(xe))
            selections[arm] = selection
        path = run.directory / 'validation_predictions.npz'
        np.savez_compressed(path, conditions=val_labels, **predictions)
        save_json(run.directory / 'selection.json', selections)
        save_json(run.directory / 'population_control.json', dict(train=atr, validation_source=ava))
        save_csv(run.directory / 'source_losses.csv', rows)
        summary = dict(status='SHORTTIME_PREDICTIONS_FROZEN', seed=seed, no_JVP=True,
            train_cells=len(us_tr), validation_source_cells=len(us_va), train_TFs=14, validation_TFs=4,
            parameter_count=sum(p.numel() for p in model.parameters()), genes=genes, estimator_retrained=True,
            model_sha256=sha256(run.directory / 'estimator.pt'), prediction_sha256=sha256(path),
            future_validation_target_read=False, forecast_unit='population',
            cell_comparison='mean of cell velocities, not observed cell-to-cell fate prediction',
            geometry='one frozen training log-PCA for sources, velocities and targets',
            runtime_seconds=time.monotonic()-start)
        save_json(run.directory / 'summary.json', summary)
        save_json(run.directory / 'status.json', dict(status='complete'))
        (run.directory / 'RESULTS.md').write_text('# Short-time source estimator retraining\n\n' + json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default='configs/w3_shorttime_20261014.json')
    ap.add_argument('--seed', type=int, required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args(); c = load_config(args.config)
    if args.seed not in c['seeds']:
        raise ValueError('Unregistered estimator seed')
    if args.smoke:
        c = dict(c, steps=5, smoke=True)
    fit_predictions(c, args.seed, args.output)


if __name__ == '__main__':
    main()
