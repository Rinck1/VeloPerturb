"""Single-sample clone x program x state analysis for MeRLin SRR33960308 (Day21).

Tests whether clone identity explains DTP program scores beyond cell state:
- state-predicted score via kNN regression in z
- clone-mean residual variance vs permuted-label null
"""
import json
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
from veloroute.preprocess import load_usa
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ROOT = Path('/data/yuchang/veloroute_ucheck_20260915')
QUANT = ROOT/'quant_merlin308'/'af_quant'
CLONES = ROOT/'clones'/'SRR33960308'/'cell_clone_assignments.csv'
PROGRAMS = ROOT/'merlin_programs.json'
NAME_MAP = Path('/data/yuchang/veloroute_kang_20260914/reference/splici_r98_v2/index/gene_id_to_name.tsv')
OUT = ROOT/'merlin308_clone_program.json'
MIN_TOTAL = 200
N_PERM = 2000


def load_name_map():
    mapping = {}
    with NAME_MAP.open() as stream:
        for line in stream:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh --output; preserve historical clone analysis')
    barcodes, genes, matrices, meta = load_usa(QUANT)
    s, u, a = [m.tocsr() for m in matrices]
    total = np.asarray((s+u+a).sum(1)).ravel()
    keep = total >= MIN_TOTAL
    barcodes = np.asarray(barcodes)[keep]
    s, u, a = s[keep], u[keep], a[keep]
    print(f'cells(total>={MIN_TOTAL}): {len(barcodes)}', flush=True)

    assignments = {}
    with CLONES.open() as stream:
        header = stream.readline()
        for line in stream:
            cb, clone = line.split(',')[:2]
            assignments[cb] = clone
    has_clone = np.array([b in assignments for b in barcodes])
    print(f'cells with clone: {int(has_clone.sum())}', flush=True)

    name_map = load_name_map()
    symbols = [name_map.get(g, g.split('.')[0]) for g in genes]
    symbol_index = {}
    for i, sym in enumerate(symbols):
        symbol_index.setdefault(sym, []).append(i)
    programs = json.loads(PROGRAMS.read_text())['signatures']
    expression = (s+a).astype('float64')
    depth = np.asarray(expression.sum(1)).ravel()
    scaled = expression.multiply(1e4/np.maximum(depth, 1)[:, None]).tocsr()
    scaled.data = np.log1p(scaled.data)
    scores = {}
    for name, gene_list in programs.items():
        idx = [i for g in gene_list for i in symbol_index.get(g, [])]
        if len(idx) < 3:
            continue
        scores[name] = np.asarray(scaled[:, idx].mean(1)).ravel()
    print('programs:', {k: len(v) for k, v in scores.items()}, flush=True)

    logs = s.astype('float64').multiply(1e4/np.maximum(np.asarray(s.sum(1)).ravel(), 1)[:, None]).tocsr()
    logs.data = np.log1p(logs.data)
    variance = np.asarray(logs.power(2).mean(0)-np.square(logs.mean(0))).ravel()
    candidates = np.flatnonzero(np.asarray((s > 0).sum(0)).ravel() >= 20)
    selected = candidates[np.argsort(-variance[candidates], kind='stable')[:2000]]
    z = PCA(50, svd_solver='randomized', random_state=20260920).fit_transform(np.asarray(logs[:, selected].todense()))

    clone_ids = np.array([assignments.get(b, '') for b in barcodes])
    clone_counts = Counter(clone_ids[has_clone])
    multi = {c for c, n in clone_counts.items() if n >= 3}
    print(f'clones with >=3 cells: {len(multi)} covering {sum(clone_counts[c] for c in multi)} cells', flush=True)

    rng = np.random.default_rng(20260920)
    report = dict(cells=int(len(barcodes)), cells_with_clone=int(has_clone.sum()),
                  multi_cell_clones=len(multi), programs=dict(), top_clones=dict(),
                  analysis_revision='same_time_association_audit_20260921_v2',
                  cross_time_fate_test=False, conditional_independence_established=False,
                  limitations=['Global residual permutation assumes exchangeability; state/depth confounding not excluded.',
                               'Clone assignments need concordance audit against published annotations.'])
    for name, score in scores.items():
        idx = np.flatnonzero(has_clone)
        zz, sc = z[idx], score[idx]
        # With X=None sklearn already excludes the query itself. Do not discard
        # the closest nonself neighbour a second time.
        nn = NearestNeighbors(n_neighbors=min(20, len(zz)-1)).fit(zz)
        neigh = nn.kneighbors(return_distance=False)
        state_pred = sc[neigh].mean(1)
        resid = sc-state_pred
        r2_state = float(1.-resid.var()/max(sc.var(), 1e-12))
        labels = clone_ids[idx]
        stats = {}
        for group_name, threshold in (('multi3', 3),):
            members = np.array([c in multi for c in labels])
            lab, res = labels[members], resid[members]
            unique = np.unique(lab)
            means = np.array([res[lab == c].mean() for c in unique])
            sizes = np.array([(lab == c).sum() for c in unique])
            observed = float(np.average(np.square(means), weights=sizes))
            null = []
            for _ in range(N_PERM):
                perm = rng.permutation(res)
                m = np.array([perm[lab == c].mean() for c in unique])
                null.append(float(np.average(np.square(m), weights=sizes)))
            p = float((1+np.count_nonzero(np.asarray(null) >= observed))/(N_PERM+1))
            stats[group_name] = dict(observed=observed, null_mean=float(np.mean(null)),
                                     null_q99=float(np.quantile(null, .99)), p_value=p,
                                     clones=int(len(unique)), cells=int(members.sum()))
        report['programs'][name] = dict(state_r2=r2_state, clone_test=stats)
        print(f"{name}: state R2={r2_state:.3f} | clone residual var={stats['multi3']['observed']:.5f} "
              f"null={stats['multi3']['null_mean']:.5f} (q99={stats['multi3']['null_q99']:.5f}) "
              f"p={stats['multi3']['p_value']:.4f}", flush=True)
        # top enriched clones
        per_clone = {}
        for c in unique:
            m = lab == c
            if m.sum() >= 5:
                per_clone[c] = float(res[m].mean())
        top = sorted(per_clone.items(), key=lambda kv: -kv[1])[:5]
        report['top_clones'][name] = [dict(clone=c, mean_residual=v, cells=int((lab == c).sum())) for c, v in top]
    names = list(report['programs'])
    pvalues = np.array([report['programs'][name]['clone_test']['multi3']['p_value'] for name in names])
    order = np.argsort(pvalues)
    adjusted = np.minimum.accumulate((pvalues[order]*len(names)/np.arange(1, len(names)+1))[::-1])[::-1].clip(0, 1)
    for index, value in zip(order, adjusted):
        report['programs'][names[index]]['clone_test']['multi3']['q_value_BH'] = float(value)
    args.output.write_text(json.dumps(report, indent=2))
    print('saved', args.output)


if __name__ == '__main__':
    main()
