"""Prepare train-only target data using the external GFG preprocessing function."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse

from veloroute.artifacts import Run, load_config, save_json, sha256
from veloroute.preprocess import load_usa
from w3_6a_source_screen import clone_map
from w3_6c_deltaU_identifiability import condition_roles, load_allowed
from w3_validation import split_us


def native_imports(c):
    root = Path(c['native_root']).resolve()
    if not (root / 'model/model.py').is_file():
        raise ValueError('Registered original GFG repository unavailable')
    sys.path.insert(0, str(Path(c['isolated_scvelo']).resolve()))
    sys.path.insert(0, str(root))
    import scvelo as scv
    if scv.__version__ != c['scvelo_version']:
        raise ValueError('Unexpected scVelo version')
    return root


def scoped_artifact_root(c):
    path = Path(c['artifact_root']).resolve()
    if not path.is_relative_to(Path('/data/yuchang/veloroute_w3_20261014')):
        raise ValueError('Artifacts must use the registered task data directory')
    return path


def merlin_training_counts(c):
    """Use fixed source-only clone split; no future counts or group labels."""
    root = Path(c['merlin_root'])
    quant = root / 'quant_merlin310/af_quant'
    ids, genes, layers, _ = load_usa(quant)
    s, u = layers[:2]
    assignments = root / 'clones/SRR33960310/cell_clone_assignments.csv'
    mapping = clone_map(assignments, set(ids), single=True)
    with np.load(c['merlin_source_split'], allow_pickle=False) as f:
        clones = f['clone_ids']; train = set(clones[f['train_indices']]); test = set(clones[f['test_indices']])
    if train & test or not train or not test:
        raise ValueError('Invalid frozen clone split')
    depth = np.asarray(s.sum(1)).ravel()
    mask = np.array([mapping.get(cell) in set(clones) for cell in ids]) & (depth >= c['minimum_source_s_counts'])
    selected_ids = np.asarray(ids)[mask]
    labels = np.array([mapping[cell] for cell in selected_ids])
    # Split both modalities on the fixed complete source cohort before selecting
    # training clones, so a later source query can reproduce the identical A half.
    up, sp = split_us(u[mask], s[mask], parts=2, seed=c['count_split_seed'])
    training = np.isin(labels, sorted(train))
    if set(labels[training]) != train:
        raise ValueError('A registered training clone lost all source cells')
    return sp[0][training], up[0][training], np.asarray(genes), selected_ids[training], labels[training], dict(
        training_clones=len(train), heldout_clones=len(test), split='A of two independent U/S count parts',
        split_seed=c['count_split_seed'], source_cells_in_split=len(selected_ids),
        future_counts_read=False, heldout_clones_used_in_fit=False), [
        quant / 'alevin/quants_mat.mtx', quant / 'alevin/quants_mat_rows.txt',
        quant / 'alevin/quants_mat_cols.txt', quant / 'quant.json', assignments, Path(c['merlin_source_split'])]


def validate_prepared(adata, operator):
    for layer, moment in [('spliced', 'Ms'), ('unspliced', 'Mu')]:
        normalized = adata.layers[layer]
        actual = adata.layers[moment]
        expected = (operator @ sparse.csr_matrix(normalized)).toarray().astype(np.float32)
        if not np.isfinite(actual).all() or not np.allclose(expected, actual, rtol=1e-5, atol=1e-6):
            raise ValueError('Native moments do not match the recorded frozen operator')
    if adata.n_vars < 32 or adata.n_obs < 32:
        raise ValueError('Insufficient source training data')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/w3_7_native_gfg_20261014.json')
    ap.add_argument('--dataset', required=True, choices=['renge', 'merlin'])
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    c = load_config(args.config); native = native_imports(c)
    import anndata as ad
    from preprocessing import preprocess_data_scv
    from scvelo.preprocessing.neighbors import get_connectivities
    start = time.monotonic()
    if args.dataset == 'renge':
        roles = condition_roles(c['reservation'])
        layers, labels, genes, manifest = load_allowed(c['renge_source'], roles, ['spliced', 'unspliced'])
        # load_allowed preserves the source row IDs in its digest; read only IDs
        # and reservation metadata to recover the same selected row order.
        import h5py
        from anndata.experimental import read_elem
        with h5py.File(c['renge_source'], 'r') as f:
            conditions = np.asarray(read_elem(f['obs']['condition'])).astype(str)
            ids = np.asarray(read_elem(f['obs'][f['obs'].attrs['_index']])).astype(str)
            ids = ids[np.array([roles.get(x) in ('train', 'control') for x in conditions])]
        s, u = layers['spliced'], layers['unspliced']
        inputs = [Path(c['reservation'])]  # full h5 counts are not hashed/read
    else:
        s, u, genes, ids, labels, manifest, inputs = merlin_training_counts(c)
    artifact = scoped_artifact_root(c) / args.dataset / 'input_v1'
    if artifact.exists():
        raise ValueError('Prepared artifact already exists; use a new protocol/version')
    with Run(args.output, stage='W3_7_native_train_source_preparation', kind='engineering', config=c,
             inputs=[Path(args.config), Path(__file__), Path(__file__).with_name('w3_6c_deltaU_identifiability.py'),
                     Path(__file__).with_name('w3_validation.py'), native / 'preprocessing.py', *inputs], seed=c['preprocess_seed']) as run:
        artifact.mkdir(parents=True, exist_ok=False)
        raw = ad.AnnData(X=s.astype(np.float32).copy(), obs=pd.DataFrame({'condition': labels}, index=ids),
                         var=pd.DataFrame({'original_column': np.arange(len(genes))}, index=genes))
        raw.layers['spliced'] = s.astype(np.float32).copy()
        raw.layers['unspliced'] = u.astype(np.float32).copy()
        processed = preprocess_data_scv(raw, n_top_genes=c['top_genes'], n_neighbors=20, seed=c['preprocess_seed'])
        operator = get_connectivities(processed, mode='connectivities', n_neighbors=30, recurse_neighbors=False).tocsr()
        validate_prepared(processed, operator)
        original = processed.var['original_column'].to_numpy(dtype=int)
        row_lookup = {cell: i for i, cell in enumerate(ids)}
        rows = np.array([row_lookup[cell] for cell in processed.obs_names])
        processed.layers['raw_S'] = s[rows][:, original].astype(np.float32)
        processed.layers['raw_U'] = u[rows][:, original].astype(np.float32)
        processed.obsp['w3_moment_operator'] = operator
        path = artifact / 'train.h5ad'
        processed.write_h5ad(path, compression='gzip')
        result = dict(status='NATIVE_SOURCE_INPUT_READY', dataset=args.dataset, path=str(path), sha256=sha256(path),
            cells=processed.n_obs, genes=processed.n_vars, loaded_conditions=sorted(set(labels)),
            native_preprocessing_sha256=sha256(native / 'preprocessing.py'),
            graph_parameters=processed.uns['neighbors']['params'], source_manifest=manifest,
            original_hvg_parameter_adapted_to=c['top_genes'],
            future_counts_read=False, excluded_count_rows_read=False, runtime_seconds=time.monotonic() - start)
        save_json(run.directory / 'summary.json', result)
        save_json(artifact / 'ready.json', result)
        save_json(run.directory / 'status.json', dict(status='complete'))
        (run.directory / 'RESULTS.md').write_text('# Native GFG source input\n\n' + json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
