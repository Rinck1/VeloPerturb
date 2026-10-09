"""A2 + kappa on LARRY day2 sister cells (shared clone == sisters).

Loads the LARRY in-vitro h5ad, filters day2 cells, labels clones from X_clone,
and runs the A2 reliability + kappa(IV) + clone-ICC pipeline.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import anndata as ad
from scipy import sparse

sys.path.insert(0, '/home/yuchang/wangjiaxuan/src')
spec = importlib.util.spec_from_file_location('a2mod', '/home/yuchang/wangjiaxuan/scripts/phase0/a2_u_innovation_reliability.py')
a2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a2)


def main():
    h5 = '/data/yuchang/veloroute_larry_20261011/larry_invitro_adata_sub_raw.h5ad'
    out = 'outputs/w1_a2_larry_day2.json'
    a = ad.read_h5ad(h5)
    day2 = (a.obs['time_info'] == 2.0).to_numpy()
    print(f'[LARRY] day2 cells={int(day2.sum())}', flush=True)
    s = sparse.csr_matrix(a.layers['spliced'][day2])
    u = sparse.csr_matrix(a.layers['unspliced'][day2])
    xc = a.obsm['X_clone'][day2]
    xc = xc if sparse.issparse(xc) else sparse.csr_matrix(xc)
    labels = np.full(xc.shape[0], -1, dtype=np.int64)
    if xc.nnz:
        am = np.asarray(xc.argmax(1)).ravel()
        has = np.asarray(xc.sum(1)).ravel() > 0
        labels[has] = am[has]
    print(f'[LARRY] clone-labeled cells={int((labels>=0).sum())} clones={len(np.unique(labels[labels>=0]))}', flush=True)
    rng = np.random.default_rng(0)
    result = dict(tag='A2_LARRY_day2', h5=h5)
    a2.run_dataset('larry_day2', s, u, rng, 2000, result, clone_labels=labels)
    Path(out).write_text(__import__('json').dumps(result, indent=2))
    print(__import__('json').dumps({k: v for k, v in result.items() if isinstance(v, dict)}, indent=2))


if __name__ == '__main__':
    main()
