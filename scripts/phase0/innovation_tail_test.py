"""Per-cell innovation tail test (fixed): permute innovation VECTORS, not cos values.

Two reference fields:
  field_within  : cross-fit E[v|z] inside the same population (idealized field)
  field_train   : field fitted on training source, applied to validation (what the
                  model effectively has for a held-out condition)
"""
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
from sklearn.neighbors import NearestNeighbors

N_PERM = 2000
TAU = 0.3


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


def within_field(source, v):
    pred = np.zeros_like(v)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in kf.split(source['z']):
        pred[te] = Ridge(alpha=1.0).fit(source['z'][tr], v[tr]).predict(source['z'][te])
    return v-pred


def run(tag, fold, gene_dir, conditions, output):
    data = gfg_velocity(fold, gene_dir, conditions)
    train_source, train_v = data['train']
    report = {'tag': tag, 'rows': []}
    for role in ('train', 'validation'):
        source, v = data[role]
        innov_within = within_field(source, v)
        if role == 'train':
            innov_train = innov_within
        else:
            pred = Ridge(alpha=1.0).fit(train_source['z'], train_v).predict(source['z'])
            innov_train = v-pred
        target, tm = load_pack(Path(fold)/f'{role}_target.npz', expected_side='target')
        rng = np.random.default_rng(20260919)
        for cond in sorted(set(source['conditions'])):
            ms = np.asarray(source['conditions'] == cond)
            mt = np.asarray(target['conditions'] == cond)
            if ms.sum() < 20 or mt.sum() < 20:
                continue
            d = future_directions(source['z'][ms], target['z'][mt])
            n = int(ms.sum())
            for label, innov in (('within', innov_within[ms]), ('trainfield', innov_train[ms])):
                cos_i = cos_with(innov, d)
                null_means, null_tails = [], []
                for _ in range(N_PERM):
                    nc = cos_with(innov[rng.permutation(n)], d)
                    null_means.append(float(nc.mean()))
                    null_tails.append(float((nc > TAU).mean()))
                p_mean = float((np.asarray(null_means) >= cos_i.mean()).mean())
                p_tail = float((np.asarray(null_tails) >= (cos_i > TAU).mean()).mean())
                row = dict(role=role, condition=str(cond), variant=label, n=n,
                    cos_v_mean=float(cos_with(v[ms], d).mean()),
                    innov_mean=float(cos_i.mean()),
                    innov_q90=float(np.quantile(cos_i, .9)),
                    frac_above_tau=float((cos_i > TAU).mean()),
                    null_frac_mean=float(np.mean(null_tails)),
                    null_frac_q99=float(np.quantile(null_tails, .99)),
                    p_mean=p_mean, p_tail=p_tail)
                report['rows'].append(row)
                print(f"{role:10s} {str(cond):12s} {label:10s} n={n:4d} cos(v)={row['cos_v_mean']:+.3f} "
                      f"innov_mean={row['innov_mean']:+.3f} (p={p_mean:.4f}) q90={row['innov_q90']:+.3f} "
                      f"frac>0.3={row['frac_above_tau']:.3f} vs null {row['null_frac_mean']:.3f} "
                      f"(q99={row['null_frac_q99']:.3f}, p={p_tail:.4f})", flush=True)
    Path(output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    run('RENGE', '/home/yuchang/wangjiaxuan/outputs/veloroute_real_pipeline_20260912_v2/fold',
        '/home/yuchang/wangjiaxuan/outputs/veloroute_gfg_inputs_20260914',
        'data/renge/conditions/esm2_3b_v1/conditions.npz',
        '/home/yuchang/wangjiaxuan/outputs/renge_innovation_tail.json')
    run('pancreas', '/data/yuchang/veloroute_gainprobe_20260915/prepared_pancreas_pathways',
        '/data/yuchang/veloroute_gainprobe_20260915/prepared_pancreas_pathways',
        '/data/yuchang/veloroute_gainprobe_20260915/prepared_pancreas_pathways/conditions.npz',
        '/data/yuchang/veloroute_gainprobe_20260915/pancreas_innovation_tail.json')
