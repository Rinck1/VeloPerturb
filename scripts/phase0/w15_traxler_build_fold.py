"""Build a model-ready Traxler fold (0h->6h or 6h->24h) from the trainable manifest.

Follows the prepared-fold contract: source packs (z, velocity, u_features,
u_predicted, depth, cell_ids, conditions), target packs (z, cell_ids,
conditions), transform.npz, conditions.npz.

Conditions = ONE-HOT over KO (placeholder; ESM embeddings pending). No KO effect
or differential expression is read.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import anndata as ad
from scipy import sparse

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
from veloroute.contracts import FitScope
from veloroute.latent import FrozenSplicingTransform, save_pack, normalized_counts, dense_counts


def log(m):
    print(f'[FOLD] {m}', flush=True)


def load_cells(cell_ids, h5dir, cache):
    """Load S/U for cell_ids (grouped by library) with an h5ad cache."""
    by_lib = {}
    for cid in cell_ids:
        lib = cid.split(':', 1)[0]
        by_lib.setdefault(lib, set()).add(cid)
    ids_out, ss, us, genes = [], [], [], None
    for lib in sorted(by_lib):
        if lib not in cache:
            cache[lib] = ad.read_h5ad(Path(h5dir) / f'{lib}.h5ad')
        a = cache[lib]
        names = np.array([str(x) for x in a.obs_names])
        keep = np.array([n in by_lib[lib] for n in names])
        if keep.sum() == 0:
            continue
        if genes is None:
            genes = np.array([str(g) for g in a.var_names])
        else:
            assert list(genes) == [str(g) for g in a.var_names]
        ids_out.append(names[keep])
        ss.append(sparse.csr_matrix(a.layers['spliced'])[keep])
        us.append(sparse.csr_matrix(a.layers['unspliced'])[keep])
    if not ids_out:
        raise ValueError('No cells loaded')
    return np.concatenate(ids_out), sparse.vstack(ss).tocsr(), sparse.vstack(us).tocsr(), genes


def target_z(s, selected, mean, components, target_sum):
    tot = np.asarray(s.sum(1)).ravel().astype(np.float64)
    scale = np.where(tot > 0, target_sum / np.maximum(tot, 1e-9), 0.0)
    logs = s.multiply(scale[:, None]).tocsr()
    logs.data = np.log1p(logs.data)
    dense = np.asarray(logs[:, selected].todense(), dtype=np.float32)
    return ((dense - mean) @ components.T).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default='outputs/w15_traxler_trainable_v1/cells.csv.gz')
    ap.add_argument('--split', required=True)
    ap.add_argument('--h5dir', default='/data/yuchang/veloroute_w4/traxler2025/processed_usa_v1')
    ap.add_argument('--t0', type=int, required=True)
    ap.add_argument('--t1', type=int, required=True)
    ap.add_argument('--n-genes', type=int, default=2000)
    ap.add_argument('--n-components', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(args.split).read_text())
    import pandas as pd
    df = pd.read_csv(args.manifest)
    df['cell_id'] = df['library'].astype(str) + ':' + df['barcode'].astype(str)
    meta = {r.cell_id: (int(r.time), str(r.KO)) for r in df.itertuples()}
    cache = {}
    def cells_for(kos, t):
        ks = set(kos)
        return [cid for cid, (tt, ko) in meta.items() if ko in ks and tt == t]
    roles = {'train': spec['train'], 'validation': spec['val'], 'test': spec['test']}
    # fit transform on train source only
    ids_tr, S_tr, U_tr, genes = load_cells(cells_for(roles['train'], args.t0), args.h5dir, cache)
    log(f'train source cells={len(ids_tr)} genes={len(genes)}')
    scope = FitScope(frozenset(ids_tr))
    transform = FrozenSplicingTransform.fit(S_tr, U_tr, gene_ids=genes, cell_ids=ids_tr, scope=scope,
                                            n_genes=args.n_genes, n_components=args.n_components, seed=args.seed)
    transform.save(out / 'transform.npz')
    selected = transform.selected
    gene_ids_sel = np.array(transform.gene_ids)[selected]
    log(f'selected={len(selected)} supported={int(transform.supported.sum())}')

    # one-hot conditions over all KOs in the task
    kos = sorted(set(spec['train'] + spec['val'] + spec['test']))
    ko_index = {k: i for i, k in enumerate(kos)}
    emb = np.eye(len(kos), dtype=np.float32)
    np.savez_compressed(out / 'conditions.npz', conditions=np.array(kos), embeddings=emb,
                        metadata_json=np.array(json.dumps({'kind': 'one_hot_over_KO',
                                                           'note': 'ESM embeddings pending', 'n': len(kos)})))

    def onehot(cell_ids):
        return np.array([kos[ko_index[meta[c][1]]] for c in cell_ids])

    for role, kos_role in roles.items():
        # source
        ids, S, U, _ = load_cells(cells_for(kos_role, args.t0), args.h5dir, cache)
        arr = transform.transform(S, U, gene_ids=genes)
        arr['depth'] = np.asarray(S.sum(1)).ravel()
        arr['cell_ids'] = ids
        arr['conditions'] = onehot(ids)
        save_pack(out / f'{role}_source.npz', arr, {'side': 'source', 'role': role, 'day': args.t0,
                                                    'task': f'traxler_{args.t0}_{args.t1}', 'kind': 'development',
                                                    'transform_hash': 'local', 'estimator': transform.estimator,
                                                    'fit_ids_hash': transform.fit_ids_hash, 'formal_ready': False})
        # target
        ids_t, S_t, _, _ = load_cells(cells_for(kos_role, args.t1), args.h5dir, cache)
        zt = target_z(S_t, selected, transform.mean, transform.components, transform.target_sum)
        save_pack(out / f'{role}_target.npz', {'z': zt, 'cell_ids': ids_t, 'conditions': onehot(ids_t)},
                  {'side': 'target', 'role': role, 'day': args.t1, 'task': f'traxler_{args.t0}_{args.t1}',
                   'kind': 'development', 'transform_hash': 'local', 'formal_ready': False})
        # gene packs (selected genes; U first then S)
        Sd = normalize_sparse(S, transform.target_sum)[:, selected]
        Ud = normalize_sparse(U, transform.target_sum)[:, selected]
        gene_us = np.concatenate([np.asarray(Ud.todense(), dtype=np.float32),
                                  np.asarray(Sd.todense(), dtype=np.float32)], 1)
        save_pack(out / f'{role}_gene_source.npz', {'gene_us': gene_us, 'cell_ids': ids, 'gene_ids': gene_ids_sel},
                  {'side': 'gene_source', 'role': role, 'task': f'traxler_{args.t0}_{args.t1}',
                   'source_hash': 'local', 'transform_hash': 'local'})
        log(f'{role}: source={len(ids)} target={len(ids_t)}')

    summary = dict(status='TRAXLER_FOLD_READY', task=f'{args.t0}->{args.t1}', n_kos=len(kos),
                   n_train_kos=len(spec['train']), n_val=len(spec['val']), n_test=len(spec['test']),
                   n_genes=len(selected), conditions='one_hot_over_KO (ESM pending)',
                   source_semantics='dominant_HTO_90pct_not_author_GMM')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary))


def normalize_sparse(mat, target_sum):
    tot = np.asarray(mat.sum(1)).ravel().astype(np.float64)
    scale = np.where(tot > 0, target_sum / np.maximum(tot, 1e-9), 0.0)
    m = mat.astype(np.float32).multiply(scale[:, None]).tocsr()
    m.data = np.log1p(m.data)
    return m


if __name__ == '__main__':
    main()
