"""W4 day4->5 development: raw counts -> source fit -> frozen predictions.

Predict and evaluate are separate invocations. HDF5 sparse row loading means
the prediction stage reads future counts ONLY for the registered training TFs
and controls. Held-out future expression is first read by evaluate.
Replication execution is deferred until the preregistration is frozen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse
from anndata._io.specs import read_elem

from w4_artifacts import Artifacts, sha256
from w4_arms import CORE_ARMS, average_predictions_then_errors, predict_arms
from w4_streams import SourceTransform, add_swap_feature, group_mean, population_features


DEVELOPMENT_TFS = ('LIN28A', 'NANOG', 'POU5F1', 'ZIC3')
CONTROL_LABELS = ('AAVS1', 'CTRL')


def read_metadata(path):
    with h5py.File(path, 'r') as f:
        return read_elem(f['obs']), read_elem(f['var']).index.to_numpy(dtype=str)


def read_count_rows(path, layer, rows):
    """Read only requested CSR rows; no future held-out count array is loaded."""
    rows = np.asarray(rows, dtype=int)
    if rows.ndim != 1 or not len(rows) or (np.diff(rows) <= 0).any():
        raise ValueError('Sorted unique nonempty cell rows required')
    with h5py.File(path, 'r') as f:
        node = f['layers'][layer]
        if node.attrs.get('encoding-type') != 'csr_matrix':
            raise ValueError('Row-restricted loading requires a CSR-encoded layer')
        shape = tuple(node.attrs['shape'])
        if rows[0] < 0 or rows[-1] >= shape[0]:
            raise ValueError('Count row index outside matrix')
        pointers = node['indptr'][:]
        lengths = pointers[rows+1]-pointers[rows]
        data, indices = [], []
        # Merge only adjacent requested rows; never bridge an excluded cell.
        splits = np.flatnonzero(np.diff(rows) != 1)+1
        for segment in np.split(rows, splits):
            start, stop = int(pointers[segment[0]]), int(pointers[segment[-1]+1])
            data.append(node['data'][start:stop]); indices.append(node['indices'][start:stop])
    return sparse.csr_matrix((np.concatenate(data), np.concatenate(indices),
        np.r_[0, np.cumsum(lengths)]), shape=(len(rows), shape[1]))


def qc_mask(obs):
    for key in ('technical_qc_pass', 'guide_call_pass', 'role', 'condition', 'day'):
        if key not in obs:
            raise ValueError(f'Missing required technical metadata: {key}')
    return obs['technical_qc_pass'].astype(bool).to_numpy() & obs['guide_call_pass'].astype(bool).to_numpy()


def condition_embeddings(path, labels):
    with np.load(path, allow_pickle=False) as f:
        names, matrix = f['conditions'].astype(str), f['embeddings']
        metadata = json.loads(str(f['metadata_json']))
    if len(set(names)) != len(names) or not metadata.get('frozen') or not np.isfinite(matrix).all():
        raise ValueError('Invalid frozen ESM embeddings')
    lookup = dict(zip(names, matrix))
    missing = set(labels)-set(lookup)-set(CONTROL_LABELS)
    if missing:
        raise ValueError(f'Missing TF ESM embedding: {sorted(missing)}')
    return np.stack([np.zeros(matrix.shape[1]) if label in CONTROL_LABELS else lookup[label] for label in labels])


def validate_development_times(source_obs, target_obs):
    if set(source_obs['day'].astype(int)) != {4} or set(target_obs['day'].astype(int)) != {5}:
        raise ValueError('This runner permits day4->5 development only; replication requires frozen protocol')


def predict_development(source_path, target_path, embeddings_path, output, *, seeds=(0,1,2,3),
                        n_hvg=2000, dimensions=50):
    source_obs, genes = read_metadata(source_path)
    target_obs, target_genes = read_metadata(target_path)
    validate_development_times(source_obs, target_obs)
    if not np.array_equal(genes, target_genes):
        raise ValueError('Source/target gene identifiers or order differ')
    sqc, tqc = qc_mask(source_obs), qc_mask(target_obs)
    slabel, tlabel = source_obs['condition'].astype(str).to_numpy(), target_obs['condition'].astype(str).to_numpy()
    train_labels = sorted(set(slabel[sqc & source_obs['role'].isin(['train','control']).to_numpy()]))
    # Labels/cell counts are technical metadata. Training targets must exist;
    # never silently drop a TF after looking at expression.
    if not set(train_labels) <= set(tlabel[tqc]):
        raise ValueError('Registered training TF/control lacks a target cohort')
    for label in DEVELOPMENT_TFS:
        if min(np.sum(sqc & (slabel == label)), np.sum(tqc & (tlabel == label))) < 50:
            raise ValueError(f'Insufficient cells for registered development TF {label}')
    si = np.flatnonzero(sqc & np.isin(slabel, train_labels))
    vi = np.flatnonzero(sqc & np.isin(slabel, DEVELOPMENT_TFS))
    ti = np.flatnonzero(tqc & np.isin(tlabel, train_labels))
    if set(tlabel[ti]) & set(DEVELOPMENT_TFS):
        raise AssertionError('Held-out future TF in supervised training rows')
    inputs = [Path(__file__), Path(__file__).with_name('w4_streams.py'),
        Path(__file__).with_name('w4_arms.py'), Path(__file__).with_name('w4_artifacts.py'),
        Path(source_path), Path(target_path), Path(embeddings_path)]
    run = Artifacts(output, seed=list(seeds), inputs=inputs)
    s, u = read_count_rows(source_path, 'spliced', si), read_count_rows(source_path, 'unspliced', si)
    sv, uv = read_count_rows(source_path, 'spliced', vi), read_count_rows(source_path, 'unspliced', vi)
    # No held-out target expression read above or below this boundary.
    target_s = read_count_rows(target_path, 'spliced', ti)
    ids = np.array([f'day4:{i}' for i in source_obs.index[si]])
    vids = np.array([f'day4:{i}' for i in source_obs.index[vi]])
    prediction_store = {a: [] for a in CORE_ARMS}; geometry_hash = None
    controls = np.isin(train_labels, CONTROL_LABELS)
    et, ev = condition_embeddings(embeddings_path, train_labels), condition_embeddings(embeddings_path, DEVELOPMENT_TFS)
    selections, swaps = {}, {}
    for seed in seeds:
        transform = SourceTransform.fit(s, u, cell_ids=ids, gene_names=genes, allowed_ids=set(ids),
            n_hvg=n_hvg, dimensions=dimensions, clusters=8, seed=seed)
        # PCA is deterministic and shared across seeds; only cluster centers may differ.
        if geometry_hash is not None:
            np.testing.assert_array_equal(previous_components, transform.components)
        previous_components = transform.components.copy(); geometry_hash = transform.fingerprint()
        transform.save(run.directory/f'transform_seed{seed}.npz')
        tr = population_features(transform, s, u, slabel[si], train_labels, ids)
        va = population_features(transform, sv, uv, slabel[vi], DEVELOPMENT_TFS, vids)
        tr, atr = add_swap_feature(transform, tr, tr, seed)
        va, ava = add_swap_feature(transform, va, tr, seed)
        target_means = group_mean(transform.state(target_s), tlabel[ti], train_labels)
        preds, selection = predict_arms(tr, va, target_means, et, ev, controls=controls)
        for arm in CORE_ARMS:
            prediction_store[arm].append(preds[arm])
        selections[str(seed)], swaps[str(seed)] = selection, dict(train=atr, development_source=ava)
        run.json('status.json', dict(status='predicting', completed_seed=seed))
        print(json.dumps(dict(status='development_seed_frozen', seed=seed)), flush=True)
    np.savez_compressed(run.directory/'predictions.npz', labels=np.array(DEVELOPMENT_TFS),
        seeds=np.array(seeds), **{a: np.stack(v) for a,v in prediction_store.items()})
    run.json('selection.json', dict(selections=selections))
    run.json('U_swap_audit.json', dict(swaps=swaps))
    run.json('prediction_manifest.json', dict(status='PREDICTIONS_FROZEN', kind='development',
        source_day=4, target_day=5, target_path=str(Path(target_path).resolve()),
        source_path=str(Path(source_path).resolve()), training_TFs=[c for c in train_labels if c not in CONTROL_LABELS],
        controls=[c for c in train_labels if c in CONTROL_LABELS], heldout_TFs=list(DEVELOPMENT_TFS),
        training_source_cells=len(si), development_source_cells=len(vi), training_target_cells=len(ti),
        heldout_future_expression_read=False, source_transform_fit_uses_target=False,
        training_targets_used_for_response_ridge=True, prediction_sha256=sha256(run.directory/'predictions.npz'),
        transform_sha256={str(seed): sha256(run.directory/f'transform_seed{seed}.npz') for seed in seeds},
        arm_status={**{a:'implemented' for a in CORE_ARMS}, 'learned_head':'pending',
                    'empirical_disp':'pending_previous_timepoint', 'cluster_velocity':'pending_secondary'},
        control_embedding='zero_empty_perturbation', kinetic_estimator='analytic_two_tail_OLS_no_JVP'))
    run.json('status.json', dict(status='complete'))


def evaluate_development(prediction_directory, output):
    directory = Path(prediction_directory)
    manifest = json.loads((directory/'prediction_manifest.json').read_text())
    if manifest['kind'] != 'development' or manifest['heldout_future_expression_read']:
        raise ValueError('Development prediction contract violated')
    if sha256(directory/'predictions.npz') != manifest['prediction_sha256']:
        raise ValueError('Predictions changed after freezing')
    target_path = Path(manifest['target_path'])
    # Input hashes can have relative keys; match by resolved identity.
    expected = [digest for path,digest in manifest['data_hashes'].items() if Path(path).resolve() == target_path]
    if len(expected) != 1 or sha256(target_path) != expected[0]:
        raise ValueError('Target identity/hash changed')
    seeds = manifest['seed']
    for seed in seeds:
        if sha256(directory/f'transform_seed{seed}.npz') != manifest['transform_sha256'][str(seed)]:
            raise ValueError('Frozen source transform changed')
    run = Artifacts(output, seed=seeds, inputs=[Path(__file__), directory/'prediction_manifest.json',
        directory/'predictions.npz', target_path]+[directory/f'transform_seed{s}.npz' for s in seeds])
    obs, genes = read_metadata(target_path)
    labels = np.array(manifest['heldout_TFs'])
    rows = np.flatnonzero(qc_mask(obs) & obs['condition'].isin(labels).to_numpy())
    target_s = read_count_rows(target_path, 'spliced', rows)
    transform = SourceTransform.load(directory/f'transform_seed{seeds[0]}.npz')
    np.testing.assert_array_equal(genes, transform.gene_names)
    truth = group_mean(transform.state(target_s), obs['condition'].astype(str).to_numpy()[rows], labels)
    with np.load(directory/'predictions.npz', allow_pickle=False) as p:
        np.testing.assert_array_equal(p['labels'], labels)
        errors = {arm: average_predictions_then_errors(p[arm], truth) for arm in CORE_ARMS}
    summary = dict(status='DEVELOPMENT_EVALUATED', kind='development', inference='descriptive_only_four_preused_TFs',
        mse={arm: float(error.mean()/truth.shape[1]) for arm,error in errors.items()},
        relative_gain_over_static={arm: float(1-error.mean()/errors['static'].mean()) for arm,error in errors.items()},
        per_TF_sse={arm: dict(zip(labels, error.tolist())) for arm,error in errors.items()},
        seeds_averaged_before_scoring=True, no_formal_replication_claim=True)
    run.json('summary.json', summary); run.json('status.json', dict(status='complete'))
    np.savez_compressed(run.directory/'targets_and_errors.npz', labels=labels, truth=truth, **errors)
    print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='stage', required=True)
    pred = sub.add_parser('predict-development')
    pred.add_argument('--source', default='data/renge/processed_release_v1/day4/day4.h5ad')
    pred.add_argument('--target', default='data/renge/processed_release_v1/day5/day5.h5ad')
    pred.add_argument('--embeddings', default='data/renge/conditions/esm2_3b_v1/conditions.npz')
    pred.add_argument('--output', required=True)
    evaluate = sub.add_parser('evaluate-development')
    evaluate.add_argument('--predictions', required=True); evaluate.add_argument('--output', required=True)
    args = p.parse_args()
    if args.stage == 'predict-development':
        predict_development(args.source, args.target, args.embeddings, args.output)
    else:
        evaluate_development(args.predictions, args.output)


if __name__ == '__main__':
    main()
