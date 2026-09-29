"""Nonlinear field test: does the trainfield residual signal of Finding 1 survive
a flexible E[v|z] model (MLP), or was it linear-model misspecification?"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
sys.path.insert(0, '/home/yuchang/wangjiaxuan/scripts/phase0')
from veloroute.latent import load_pack
from gfg_innovation_diagnostic import gfg_velocity
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

N_PERM = 2000


def future_directions(z_src, z_tgt, k=10):
    nn = NearestNeighbors(n_neighbors=min(k, len(z_tgt))).fit(z_tgt)
    idx = nn.kneighbors(z_src, return_distance=False)
    d = z_tgt[idx] - z_src[:, None, :]
    d = d/np.maximum(np.linalg.norm(d, axis=2, keepdims=True), 1e-9)
    d = d.mean(1)
    return d/np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)


def cos_with(v, d):
    vn = v/np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    return (vn*d).sum(1)


def mlp():
    return MLPRegressor(hidden_layer_sizes=(128, 128), early_stopping=True,
                        max_iter=800, random_state=0, learning_rate_init=1e-3)


def crossfit(fit_model, z, v):
    pred = np.zeros_like(v)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in kf.split(z):
        scaler = StandardScaler().fit(z[tr])
        model = fit_model()
        model.fit(scaler.transform(z[tr]), v[tr])
        pred[te] = model.predict(scaler.transform(z[te]))
    return pred


def perm_p(resid, d, rng, n_perm=N_PERM):
    n = len(resid)
    nulls = []
    for _ in range(n_perm):
        nulls.append(float(cos_with(resid[rng.permutation(n)], d).mean()))
    return float((np.asarray(nulls) >= cos_with(resid, d).mean()).mean())


def run(tag, fold, gene_dir, conditions):
    data = gfg_velocity(fold, gene_dir, conditions)
    train_source, train_v = data['train']
    print(f'=== {tag}')
    rng = np.random.default_rng(7)
    for role in ('train', 'validation'):
        source, v = data[role]
        if role == 'train':
            pred_lin = crossfit(Ridge, source['z'], v)
            pred_mlp = crossfit(mlp, source['z'], v)
            pred_mlp_within = pred_mlp
        else:
            pred_mlp_within = crossfit(mlp, source['z'], v)
            scaler = StandardScaler().fit(train_source['z'])
            lin = Ridge(alpha=1.0).fit(train_source['z'], train_v)
            pred_lin = lin.predict(source['z'])
            nn_model = mlp()
            nn_model.fit(scaler.transform(train_source['z']), train_v)
            pred_mlp = nn_model.predict(scaler.transform(source['z']))
        target, tm = load_pack(Path(fold)/f'{role}_target.npz', expected_side='target')
        for cond in sorted(set(source['conditions'])):
            ms = np.asarray(source['conditions'] == cond)
            mt = np.asarray(target['conditions'] == cond)
            if ms.sum() < 40 or mt.sum() < 40:
                continue
            d = future_directions(source['z'][ms], target['z'][mt])
            v_ms = v[ms]
            cos_v = float(cos_with(v_ms, d).mean())
            for label, pred in (('linear', pred_lin), ('mlp', pred_mlp), ('mlp-within', pred_mlp_within)):
                resid = v_ms-pred[ms]
                cos_r = cos_with(resid, d)
                p = perm_p(resid, d, rng)
                print(f"  {role:10s} {str(cond):12s} {label:6s} cos(v)={cos_v:+.3f} "
                      f"resid_mean={cos_r.mean():+.3f} (p={p:.4f}) "
                      f"frac>0.3={(cos_r > 0.3).mean():.3f} q90={np.quantile(cos_r, .9):+.3f}", flush=True)


if __name__ == '__main__':
    run('RENGE', '/home/yuchang/wangjiaxuan/outputs/veloroute_real_pipeline_20260912_v2/fold',
        '/home/yuchang/wangjiaxuan/outputs/veloroute_gfg_inputs_20260914',
        'data/renge/conditions/esm2_3b_v1/conditions.npz')
    run('pancreas', '/data/yuchang/veloroute_gainprobe_20260915/prepared_pancreas_pathways',
        '/data/yuchang/veloroute_gainprobe_20260915/prepared_pancreas_pathways',
        '/data/yuchang/veloroute_gainprobe_20260915/prepared_pancreas_pathways/conditions.npz')
