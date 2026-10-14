"""Audit W3-6c measurement correction before interpreting biological gain.

Only declared train/control CSR rows are read. Four source count parts reserve
D for gene selection; future training-condition S is the outcome. Shared control
uncertainty and instrument exclusion are NOT certified by condition bootstrap.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
from scipy import sparse
from anndata.experimental import read_elem

from veloroute.artifacts import Run, load_config, save_csv, save_json, sha256
from veloroute.gpu_policy import enforce_gpu_policy
from w3_validation import split_us, two_stage_iv


def condition_roles(path):
    result = {}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            name, role = row['condition'], row['role']
            if name in result and result[name] != role:
                raise ValueError('Condition has conflicting reservation roles')
            result[name] = role
    return result


def csr_selected_rows(group, rows):
    """Do not materialize full data/indices arrays containing excluded cells."""
    if group.attrs.get('encoding-type') != 'csr_matrix':
        raise ValueError('Only audited CSR storage supported')
    shape = tuple(group.attrs['shape'])
    rows = np.asarray(rows, dtype=int)
    if not len(rows) or (np.diff(rows) <= 0).any() or rows.min() < 0 or rows.max() >= shape[0]:
        raise ValueError('Rows must be nonempty sorted unique allowed indices')
    pointers = group['indptr'][:]  # offsets only, not expression values
    breaks = np.r_[0, np.flatnonzero(np.diff(rows) > 1) + 1, len(rows)]
    values, indices, out_ptr = [], [], [0]
    total = 0
    for a, b in zip(breaks[:-1], breaks[1:]):
        first, last = rows[a], rows[b - 1]
        low, high = int(pointers[first]), int(pointers[last + 1])
        values.append(group['data'][low:high])
        indices.append(group['indices'][low:high])
        out_ptr.extend((pointers[first + 1:last + 2] - low + total).tolist())
        total += high - low
    return sparse.csr_matrix((np.concatenate(values), np.concatenate(indices), np.array(out_ptr)), shape=(len(rows), shape[1]))


def load_allowed(path, roles, wanted_layers):
    with h5py.File(path, 'r') as handle:
        cond = np.asarray(read_elem(handle['obs']['condition'])).astype(str)
        allowed = np.array([roles.get(c) in ('train', 'control') for c in cond])
        rows = np.flatnonzero(allowed)
        index_key = handle['obs'].attrs['_index']
        ids = np.asarray(read_elem(handle['obs'][index_key])).astype(str)[rows]
        var_index = handle['var'].attrs['_index']
        genes = np.asarray(read_elem(handle['var'][var_index])).astype(str)
        layers = {name: csr_selected_rows(handle['layers'][name], rows) for name in wanted_layers}
    digest = hashlib.sha256()
    digest.update(json.dumps([ids.tolist(), genes.tolist(), cond[rows].tolist()]).encode())
    for name, m in sorted(layers.items()):
        digest.update(name.encode())
        for data in (m.data, m.indices, m.indptr):
            digest.update(data.tobytes())
    return layers, cond[rows], genes, dict(path=str(Path(path).resolve()),
        selected_cells=len(rows), selected_data_sha256=digest.hexdigest(),
        count_access='CSR_data_slices_for_explicit_train_and_control_rows_only',
        loaded_conditions=sorted(set(cond[rows])), excluded_count_rows_read=False)


def log_bulk(counts, mask):
    total = np.asarray(counts[mask].sum(0)).ravel().astype(float)
    if total.sum() <= 0:
        raise ValueError('Empty pseudobulk counts')
    return np.log1p(total * 1e4 / total.sum())


def standardize(x):
    sd = x.std()
    if sd <= 1e-10:
        raise ValueError('Degenerate gene panel')
    return (x - x.mean()) / sd


def coefficients(y, u, sb, sc):
    y, u, sb, sc = [standardize(v) for v in (y, u, sb, sc)]
    iv = two_stage_iv(y, sb, u, sc)
    ols = np.linalg.lstsq(np.column_stack((np.ones(len(y)), u, sb)), y, rcond=None)[0]
    double = np.linalg.lstsq(np.column_stack((np.ones(len(y)), u, sb, sc)), y, rcond=None)[0]
    return iv, float(ols[1]), float(double[1])


def descriptive_interval(x, repetitions, seed):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    b = x[rng.integers(len(x), size=(repetitions, len(x)))].mean(1)
    return dict(mean=float(x.mean()), lower=float(np.quantile(b, .025)), upper=float(np.quantile(b, .975)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/w3_6c_identifiability_20261014.json')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    c = load_config(args.config)
    enforce_gpu_policy('cpu')
    calroot = Path(c['calibration_directory'])
    cp = json.loads((calroot / 'provenance.json').read_text())
    if (cp['status'] != 'complete' or sha256(calroot / 'summary.json') != cp['outputs']['summary.json']
        or json.loads((calroot / 'summary.json').read_text())['status'] != 'CALIBRATION_PASS'):
        raise ValueError('Synthetic calibration gate not met')
    if c['count_parts'] != 4 or set(c['allowed_roles']) != {'train', 'control'}:
        raise ValueError('Protocol requires A/B/C tests plus independent D selection and train/control only')
    roles = condition_roles(c['reservation'])
    conditions = sorted(k for k, role in roles.items() if role == 'train')
    controls = sorted(k for k, role in roles.items() if role == 'control')
    if len(conditions) != c['expected_train_conditions']:
        raise ValueError('Unexpected reservation cohort')
    start = time.monotonic()
    with Run(args.output, stage='W3_6c_IV_identifiability_training_conditions', kind='development', config=c,
             inputs=[Path(args.config), Path(c['reservation']), Path(__file__), Path(__file__).with_name('w3_validation.py'),
                     calroot / 'summary.json'], seed=c['seeds']) as run:
        a, ca, genes, am = load_allowed(c['day4'], roles, ['spliced', 'unspliced'])
        b, cb, genes_b, bm = load_allowed(c['day5'], roles, ['spliced'])
        if not np.array_equal(genes, genes_b):
            raise ValueError('Day4/day5 gene index mismatch')
        save_json(run.directory / 'selected_data_manifest.json', dict(day4=am, day5=bm))
        refa, refb = np.isin(ca, controls), np.isin(cb, controls)
        if min(refa.sum(), refb.sum()) < c['minimum_cells_per_condition']:
            raise ValueError('Insufficient declared control cells')
        for cond in conditions:
            if min(np.sum(ca == cond), np.sum(cb == cond)) < c['minimum_cells_per_condition']:
                raise ValueError(f'Insufficient cells for registered condition {cond}')
        y = {cond: log_bulk(b['spliced'], cb == cond) - log_bulk(b['spliced'], refb) for cond in conditions}
        rows, summaries = [], []
        for seed in c['seeds']:
            up, sp = split_us(a['unspliced'], a['spliced'], parts=4, seed=seed)
            sd = sp[3].astype(float).copy()
            sd = (sparse.diags(1e4 / np.maximum(np.asarray(sd.sum(1)).ravel(), 1)) @ sd).tocsr()
            sd.data = np.log1p(sd.data)
            var = np.asarray(sd.power(2).mean(0)).ravel() - np.asarray(sd.mean(0)).ravel() ** 2
            eligible = np.flatnonzero(np.asarray((sp[3] > 0).sum(0)).ravel() >= c['minimum_gene_detection_in_D'])
            panel = eligible[np.argsort(-var[eligible], kind='stable')[:c['genes']]]
            panel.sort()
            if len(panel) != c['genes']:
                raise ValueError('Insufficient eligible D-selected genes')
            save_json(run.directory / f'genes_seed{seed}.json', dict(gene_ids=genes[panel].tolist(), selected_on='day4_D_train_control_only'))
            for cond in conditions:
                mask = ca == cond
                ds = [log_bulk(part, mask)[panel] - log_bulk(part, refa)[panel] for part in sp]
                du = log_bulk(up[0], mask)[panel] - log_bulk(up[0], refa)[panel]
                yy = y[cond][panel]
                response = np.argsort(-np.abs(ds[3]), kind='stable')[:c['response_genes']]
                for name, ix in [('all', np.arange(len(panel))), ('response_D', response)]:
                    real, ols, double = coefficients(yy[ix], du[ix], ds[1][ix], ds[2][ix])
                    check, check_ols, _ = coefficients(yy[ix], ds[0][ix], ds[1][ix], ds[2][ix])
                    rows.append(dict(seed=seed, condition=cond, panel=name, genes=len(ix),
                        day4_cells=int(mask.sum()), day5_cells=int((cb == cond).sum()),
                        delta_u_iv=real['extra_coefficient'], delta_u_ols=ols, delta_u_double_control=double,
                        self_s_iv=check['extra_coefficient'], self_s_ols=check_ols,
                        nominal_first_stage_f=real['first_stage_f'], self_first_stage_f=check['first_stage_f']))
            for panel_name in ('all', 'response_D'):
                rr = [r for r in rows if r['seed'] == seed and r['panel'] == panel_name]
                ui = descriptive_interval([r['delta_u_iv'] for r in rr], c['condition_bootstrap_repetitions'], seed)
                si = descriptive_interval([r['self_s_iv'] for r in rr], c['condition_bootstrap_repetitions'], seed)
                strong = all(min(r['nominal_first_stage_f'], r['self_first_stage_f']) >= c['nominal_first_stage_f_min'] for r in rr)
                margin = c['self_control_equivalence_margin']
                screen = strong and si['lower'] >= -margin and si['upper'] <= margin
                summary = dict(seed=seed, panel=panel_name, n_conditions=len(rr), delta_u=ui, self_s=si,
                    all_nominal_instruments_above_threshold=strong, self_control_screen_passed=screen,
                    primary=seed == c['primary_seed'] and panel_name == 'all')
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
            save_json(run.directory / 'status.json', dict(status='running', completed_seed=seed))
        primary = next(r for r in summaries if r['primary'])
        result = dict(status='IV_IDENTIFIABILITY_AUDIT_COMPLETE', primary=primary, summaries=summaries,
            decision='SELF_CONTROL_SCREEN_PASSED_REQUIRES_ROBUST_VALIDATION' if primary['self_control_screen_passed'] else 'DO_NOT_INTERPRET_U_COEFFICIENT_AS_INDEPENDENT_GAIN',
            biological_increment_established=False, confirmation_counts_read=False,
            bootstrap_scope='descriptive_across_training_conditions_given_observed_shared_controls; genes not iid inference units',
            git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            runtime_seconds=time.monotonic() - start)
        save_csv(run.directory / 'condition_coefficients.csv', rows)
        save_json(run.directory / 'summary.json', result)
        save_json(run.directory / 'status.json', dict(status='complete'))
        (run.directory / 'RESULTS.md').write_text('# RENGE measurement-error identifiability audit\n\n' + json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
