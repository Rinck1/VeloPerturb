"""Train-defined, held-clone source reliability screen. Never reads future counts.

The future sample contributes barcode membership/count metadata only. This is
the first part of W3-6a, not a future-fate test. Repeated splits are dependent;
only prespecified repeat0/component0 is primary. Everything else is descriptive.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

from veloroute.artifacts import Run, load_config, save_csv, save_json, sha256
from veloroute.gpu_policy import enforce_gpu_policy
from veloroute.preprocess import lines, load_usa
from w3_validation import conditional_score_test, split_us


def clone_map(path, valid_barcodes, single=True):
    result = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            if row['cell_barcode'] not in valid_barcodes or (single and int(row['n_barcodes']) != 1):
                continue
            if row['cell_barcode'] in result:
                raise ValueError('Duplicate clone assignment')
            result[row['cell_barcode']] = row['clone_barcode']
    return result


def normalize(m):
    total = np.asarray(m.sum(1)).ravel()
    out = sparse.diags(1e4 / np.maximum(total, 1)) @ m
    out = out.tocsr().astype(np.float64)
    out.data = np.log1p(out.data)
    return out


def clone_mean(values, labels, clone_ids):
    lookup = {k: i for i, k in enumerate(clone_ids)}
    indices = np.array([lookup[k] for k in labels])
    counts = np.bincount(indices, minlength=len(clone_ids))
    if (counts == 0).any():
        raise ValueError('Empty source clone')
    result = np.zeros((len(clone_ids), values.shape[1]))
    np.add.at(result, indices, values)
    return result / counts[:, None]


def source_residuals(s, u, train_mask, dimensions, neighbors):
    """Train-only PCA and kNN; exclude a training cell from its own reference."""
    train_idx = np.flatnonzero(train_mask)
    d = min(dimensions, len(train_idx) - 1, s.shape[1])
    pca = PCA(d, svd_solver='randomized', random_state=0).fit(s[train_mask])
    z = pca.transform(s)
    k = min(neighbors, len(train_idx) - 1)
    if k < 2:
        raise ValueError('Insufficient training-cell support')
    nn = NearestNeighbors(n_neighbors=k + 1).fit(z[train_mask])
    nearest = train_idx[nn.kneighbors(z, return_distance=False)]
    idx = np.stack([row[row != i][:k] for i, row in enumerate(nearest)])
    # Block summation avoids n_cells x neighbors x genes dense temporaries.
    rs, ru = s.copy(), u.copy()
    for start in range(0, len(s), 128):
        sl = slice(start, start + 128)
        rs[sl] -= s[idx[sl]].mean(1)
        ru[sl] -= u[idx[sl]].mean(1)
    return rs, ru


def nuisance_slopes(s1, s2, u1, u2, train, *, min_correlation, max_abs):
    a, b, c, d = [v[train] - v[train].mean(0) for v in (s1, s2, u1, u2)]
    cov = (a * b).mean(0)
    correlation = cov / np.maximum(np.sqrt(np.square(a).mean(0) * np.square(b).mean(0)), 1e-12)
    good = (cov > 1e-10) & (correlation >= min_correlation)
    b1 = np.divide((d * a).mean(0), cov, out=np.zeros_like(cov), where=good)
    b2 = np.divide((c * b).mean(0), cov, out=np.zeros_like(cov), where=good)
    good &= (np.abs(b1) <= max_abs) & (np.abs(b2) <= max_abs)
    return b1, b2, good


def fit_cca(a, b, ridge, dimensions):
    """All scale/PCA/CCA parameters fitted on the provided training clones."""
    def encode(x):
        mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
        p = PCA(min(dimensions, len(x) - 2, x.shape[1]), svd_solver='full').fit((x - mean) / scale)
        score = p.transform((x - mean) / scale)
        score_scale = np.maximum(score.std(0), 1e-6)
        return dict(mean=mean, scale=scale, pca=p, score_scale=score_scale), score / score_scale
    ea, x = encode(a); eb, y = encode(b)
    n = max(len(a) - 1, 1)
    cxx = x.T @ x / n + ridge * np.eye(x.shape[1])
    cyy = y.T @ y / n + ridge * np.eye(y.shape[1])
    def inverse_sqrt(m):
        value, vector = np.linalg.eigh(m)
        return (vector / np.sqrt(np.maximum(value, 1e-10))) @ vector.T
    wa, wb = inverse_sqrt(cxx), inverse_sqrt(cyy)
    left, _, right = np.linalg.svd(wa @ (x.T @ y / n) @ wb, full_matrices=False)
    ea['weights'], eb['weights'] = wa @ left, wb @ right.T
    return ea, eb


def project_cca(x, encoder):
    e = encoder
    return (e['pca'].transform((x - e['mean']) / e['scale']) / e['score_scale']) @ e['weights']


def affine_cca(encoder, components):
    e = encoder
    w = (e['pca'].components_.T / e['score_scale']) @ e['weights'][:, :components]
    return w / e['scale'][:, None], -e['mean'] / e['scale'] @ w - e['pca'].mean_ @ w


def corr(x, y):
    x, y = x - np.mean(x), y - np.mean(y)
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / den) if den > 1e-12 else 0.


def select_ridge(a, b, config, seed):
    splits = list(KFold(config['inner_folds'], shuffle=True, random_state=seed).split(a))
    scores = []
    for ridge in config['cca_ridge_grid']:
        folds = []
        for train, val in splits:
            ea, eb = fit_cca(a[train], b[train], ridge, config['cca_dimensions'])
            folds.append(corr(project_cca(a[val], ea)[:, 0], project_cca(b[val], eb)[:, 0]))
        scores.append(float(np.mean(folds)))
    return config['cca_ridge_grid'][int(np.argmax(scores))], scores


def run(config, output):
    root = Path(config['root'])
    qs = root / f'quant_merlin{config["source_run"]}' / 'af_quant'
    qt = root / f'quant_merlin{config["future_barcode_metadata_run"]}' / 'af_quant'
    cs = root / f'clones/SRR33960{config["source_run"]}/cell_clone_assignments.csv'
    ct = root / f'clones/SRR33960{config["future_barcode_metadata_run"]}/cell_clone_assignments.csv'
    sr, tr = qs / 'alevin/quants_mat_rows.txt', qt / 'alevin/quants_mat_rows.txt'
    calibration = Path(config['calibration_directory'])
    cal = json.loads((calibration / 'summary.json').read_text())
    prov = json.loads((calibration / 'provenance.json').read_text())
    if cal['status'] != 'CALIBRATION_PASS' or prov['status'] != 'complete' or sha256(calibration / 'summary.json') != prov['outputs']['summary.json']:
        raise ValueError('Validated synthetic calibration is required')
    inputs = [cs, ct, sr, tr, qs / 'quant.json', qs / 'alevin/quants_mat.mtx', qs / 'alevin/quants_mat_cols.txt',
              Path(__file__), Path(__file__).with_name('w3_validation.py'), calibration / 'summary.json']
    start = time.monotonic()
    with Run(output, stage='W3_6a_source_reliability_only', kind='development', config=config,
             inputs=inputs, seed=config['seed']) as record:
        source_barcodes, future_barcodes = set(lines(sr)), set(lines(tr))
        sm = clone_map(cs, source_barcodes, config['single_barcode_only'])
        tm = clone_map(ct, future_barcodes, config['single_barcode_only'])
        scount, tcount = Counter(sm.values()), Counter(tm.values())
        clones = sorted(k for k in scount if scount[k] >= config['minimum_source_cells']
                        and tcount[k] >= config['minimum_future_metadata_cells'])
        save_json(record.directory / 'cohort_metadata.json', dict(before_expression_qc=len(clones),
            source_mapped_cells=len(sm), future_mapped_cells=len(tm),
            both_sides_at_least_three=sum(scount[k] >= 3 and tcount[k] >= 3 for k in scount),
            future_expression_read=False))
        rows, genes, layers, _ = load_usa(qs)
        allowed = set(clones)
        selected = np.array([i for i, row in enumerate(rows) if sm.get(row) in allowed])
        labels = np.array([sm[rows[i]] for i in selected])
        s, u = layers[0][selected].tocsr(), layers[1][selected].tocsr()
        # QC on source full-count library only; no Day21 expression/axes.
        keep = np.asarray(s.sum(1)).ravel() >= config['minimum_source_selected_gene_counts']
        s, u, labels = s[keep], u[keep], labels[keep]
        counts = Counter(labels)
        clones = sorted(k for k in clones if counts[k] >= config['minimum_source_cells'])
        keep = np.isin(labels, clones)
        s, u, labels = s[keep], u[keep], labels[keep]
        del layers
        if len(clones) < 120:
            raise ValueError('Fewer than 120 QC clones: protocol requires re-review, not silent rank changes')
        uparts, sparts = split_us(u, s, parts=3, seed=config['count_split_seed'])
        # Each split is normalized from its own full-gene count depth.
        sn = [normalize(m) for m in sparts]
        un = [normalize(m) for m in uparts[:2]]
        clones = np.array(clones)
        save_csv(record.directory / 'cohort.csv', [dict(clone_id=k, source_cells=counts[k], future_metadata_cells=tcount[k]) for k in clones])
        all_rows = []
        for repeat in range(config['repeats']):
            order = np.random.default_rng(config['seed'] + repeat).permutation(len(clones))
            ntrain = int(len(clones) * config['train_fraction'])
            train, test = order[:ntrain], order[ntrain:]
            train_cells = np.isin(labels, clones[train])
            # HVG comes from independent B counts of TRAINING clones only.
            hb = sn[2][train_cells]
            var = np.asarray(hb.power(2).mean(0)).ravel() - np.asarray(hb.mean(0)).ravel() ** 2
            detection = np.asarray((sparts[2][train_cells] > 0).sum(0)).ravel()
            eligible = np.flatnonzero(detection >= config['minimum_training_gene_detection'])
            panel = eligible[np.argsort(-var[eligible], kind='stable')[:config['genes']]]
            panel.sort()
            ss, uu = [v[:, panel].toarray() for v in sn], [v[:, panel].toarray() for v in un]
            rs1, ru1 = source_residuals(ss[0], uu[0], train_cells, config['neighbor_state_dimensions'], config['neighbors'])
            rs2, ru2 = source_residuals(ss[1], uu[1], train_cells, config['neighbor_state_dimensions'], config['neighbors'])
            b1, b2, reliable = nuisance_slopes(rs1, rs2, ru1, ru2, train_cells,
                min_correlation=config['minimum_nuisance_split_correlation'], max_abs=config['maximum_abs_nuisance_slope'])
            if reliable.sum() < config['minimum_retained_genes']:
                raise ValueError(f'Repeat {repeat}: too few stable nuisance genes ({reliable.sum()}); do not force a CCA result')
            k1, k2 = (ru1 - rs1 * b1)[:, reliable], (ru2 - rs2 * b2)[:, reliable]
            sb = clone_mean(ss[2], labels, clones)
            state_pca = PCA(config['state_dimensions'], svd_solver='full').fit(sb[train])
            control = state_pca.transform(sb)
            artifact = dict(clone_ids=clones, train_indices=train, test_indices=test,
                gene_ids=np.array(genes)[panel][reliable], state_control=control,
                source_pca_genes=np.array(genes)[panel], state_pca_components=state_pca.components_, state_pca_mean=state_pca.mean_,
                nuisance_slope_a=b1[reliable], nuisance_slope_b=b2[reliable])
            for name, a, b in [('kappa', k1, k2), ('eps_s', rs1[:, reliable], rs2[:, reliable])]:
                a, b = clone_mean(a, labels, clones), clone_mean(b, labels, clones)
                ridge, cv_scores = select_ridge(a[train], b[train], config, config['seed'] + repeat)
                ea, eb = fit_cca(a[train], b[train], ridge, config['cca_dimensions'])
                xa, xb = project_cca(a, ea), project_cca(b, eb)
                k = min(config['reported_components'], xa.shape[1])
                artifact[name + '_scores_a'] = xa[:, :k]
                artifact[name + '_scores_b'] = xb[:, :k]
                for side, encoder in [('a', ea), ('b', eb)]:
                    # Frozen affine gene-level CCA map for later preregistered use.
                    w, offset = affine_cca(encoder, k)
                    artifact[f'{name}_gene_weights_{side}'] = w
                    artifact[f'{name}_offset_{side}'] = offset
                for component in range(k):
                    stat = conditional_score_test(xb[test, component], xa[test, component], control[test],
                        simulations=config['conditional_bootstrap_simulations'], seed=config['seed'] + repeat)
                    nuisance = np.column_stack((np.ones(len(test)), control[test]))
                    pair = np.column_stack((xa[test, component], xb[test, component]))
                    residual = pair - nuisance @ np.linalg.lstsq(nuisance, pair, rcond=None)[0]
                    row = dict(repeat=repeat, arm=name, component=component,
                        train_clones=len(train), test_clones=len(test), retained_genes=int(reliable.sum()),
                        ridge=ridge, heldout_correlation=corr(xa[test, component], xb[test, component]),
                        conditional_correlation=corr(residual[:, 0], residual[:, 1]),
                        conditional_score_p=stat['p_value'], primary=(repeat == config['primary_repeat']
                            and component == config['primary_component'] and name == 'kappa'))
                    all_rows.append(row)
            np.savez_compressed(record.directory / f'frozen_source_directions_{repeat:02d}.npz', **artifact)
            save_json(record.directory / 'status.json', dict(status='running', completed_repeats=repeat + 1,
                total_repeats=config['repeats'], future_expression_read=False))
            print(json.dumps(dict(repeat=repeat, retained_genes=int(reliable.sum()),
                first_components=[r for r in all_rows if r['repeat'] == repeat and r['component'] == 0])), flush=True)
        primary = next(r for r in all_rows if r['primary'])
        supported = (primary['heldout_correlation'] > 0 and primary['conditional_correlation'] > 0
                     and primary['conditional_score_p'] < config['alpha'])
        result = dict(status='SOURCE_RELIABILITY_SCREEN_COMPLETE', primary=primary,
            reliable_kappa_direction_supported=supported, n_clones=len(clones), n_source_cells=len(labels),
            future_expression_read=False, fate_prediction_evaluated=False,
            gap_6a_closed=False, repeated_splits_independent=False,
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            runtime_seconds=time.monotonic() - start,
            interpretation='Repeat0/component0 is primary; all other splits/components are descriptive. '
                'No future expression read: neither velocity usefulness nor absence of fate information is established.')
        save_csv(record.directory / 'source_reliability.csv', all_rows)
        save_json(record.directory / 'summary.json', result)
        save_json(record.directory / 'status.json', dict(status='complete'))
        (record.directory / 'RESULTS.md').write_text('# MeRLin source-only multigene reliability\n\n' + json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/w3_6a_source_screen_20261014.json')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    enforce_gpu_policy('cpu')
    run(load_config(args.config), args.output)


if __name__ == '__main__':
    main()
