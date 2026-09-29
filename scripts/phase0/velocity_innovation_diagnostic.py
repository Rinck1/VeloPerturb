"""Decompose velocity into z-predictable + innovation; test direction of the innovation.

Alignment to kNN future neighbours is an exploratory proxy. A null result does
not establish zero conditional information or rule out lineage prediction.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
from veloroute.latent import load_pack
from veloroute.velocity_diagnostics import centered_velocity_r2
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors


def direction_cos(z_src, v_src, z_tgt, k=10):
    nn = NearestNeighbors(n_neighbors=min(k, len(z_tgt))).fit(z_tgt)
    idx = nn.kneighbors(z_src, return_distance=False)
    d = z_tgt[idx] - z_src[:, None, :]
    d = d/np.maximum(np.linalg.norm(d, axis=2, keepdims=True), 1e-9)
    d = d.mean(1)
    d = d/np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    v = v_src/np.maximum(np.linalg.norm(v_src, axis=1, keepdims=True), 1e-9)
    return (v*d).sum(1)


def crossfit_innovation(z, v, seed=0):
    pred = np.zeros_like(v)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in kf.split(z):
        model = Ridge(alpha=1.0).fit(z[tr], v[tr])
        pred[te] = model.predict(z[te])
    innovation = v-pred
    r2 = centered_velocity_r2(v, pred)
    return pred, innovation, float(r2)


def analyze(fold, tag, device_note=''):
    print(f'=== {tag} ({fold})')
    train, _ = load_pack(Path(fold)/'train_source.npz', expected_side='source')
    background = Ridge(alpha=1.).fit(train['z'], train['velocity'])
    for role in ('train', 'validation'):
        try:
            source, sm = load_pack(Path(fold)/f'{role}_source.npz', expected_side='source')
        except Exception:
            continue
        v = source['velocity']
        if float(np.abs(v).max()) == 0.:
            print(f'  {role}: zero velocity; skip')
            continue
        target, tm = load_pack(Path(fold)/f'{role}_target.npz', expected_side='target')
        if role == 'train':
            pred, innov, r2 = crossfit_innovation(source['z'], v)
        else:
            pred = background.predict(source['z'])
            innov, r2 = v-pred, centered_velocity_r2(v, pred)
        for cond in sorted(set(source['conditions']))[:6]:
            ms = np.asarray(source['conditions'] == cond)
            mt = np.asarray(target['conditions'] == cond)
            if ms.sum() < 20 or mt.sum() < 20:
                continue
            cos_v = direction_cos(source['z'][ms], v[ms], target['z'][mt]).mean()
            cos_pred = direction_cos(source['z'][ms], pred[ms], target['z'][mt]).mean()
            cos_innov = direction_cos(source['z'][ms], innov[ms], target['z'][mt]).mean()
            print(f'  {role:10s} {str(cond):12s} n={int(ms.sum()):4d} '
                  f'z-redundant R2={r2:.3f} cos(v)={cos_v:+.3f} cos(E[v|z])={cos_pred:+.3f} cos(innov)={cos_innov:+.3f}')
        print(f'  {role}: velocity z-redundancy R2={r2:.3f}  (|v| rms={float(np.sqrt(np.square(v).mean())):.3f})')


if __name__ == '__main__':
    analyze('/home/yuchang/wangjiaxuan/outputs/veloroute_real_pipeline_20260912_v2/fold', 'RENGE residual')
    analyze('/data/yuchang/veloroute_gainprobe_20260915/prepared_pancreas_pathways', 'pancreas residual')
