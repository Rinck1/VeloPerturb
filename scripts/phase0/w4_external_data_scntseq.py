"""GEO scNT-seq neuronal labeling release: technical QC only, no S/U inference.

C = new and T = OLD according to GSM4512669 metadata. Total = C + T.
No published response-selected cell subset, embedding, or gene list is used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from w4_external_data_scifate import digest, fetch, write_json


def read_gene_table(path):
    """Avoid dense whole-table loading; retain original gene/barcode strings."""
    with gzip.open(path, 'rt') as f:
        barcodes = f.readline().rstrip('\r\n').split('\t')
    if len(set(barcodes)) != len(barcodes):
        raise ValueError(f'Duplicate barcode in {path}')
    blocks, genes = [], []
    names = ['gene_identifier', *barcodes]
    dtypes = dict.fromkeys(barcodes, np.int32)
    dtypes['gene_identifier'] = str
    for frame in pd.read_csv(path, sep='\t', header=None, skiprows=1, names=names,
                             dtype=dtypes, keep_default_na=False, chunksize=512):
        genes.extend(frame.pop('gene_identifier').tolist())
        values = frame.to_numpy()
        if (values < 0).any():
            raise ValueError('Negative raw counts')
        blocks.append(sparse.csr_matrix(values))
    if len(set(genes)) != len(genes):
        raise ValueError(f'Duplicate gene ID in {path}; do not silently merge')
    matrix = sparse.vstack(blocks, format='csr').T.tocsr()
    return matrix, pd.Index(barcodes), pd.Index(genes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='configs/w4_external_data_scntseq_files.csv')
    p.add_argument('--data-root', default='/data/yuchang/veloroute_w4/scntseq')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    started = time.monotonic()
    output, root = Path(args.output), Path(args.data_root)
    output.mkdir(parents=True, exist_ok=False)
    raw = root / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest).open() as f:
        rows = list(csv.DictReader(f))
    total_bytes = sum(int(r['expected_bytes']) for r in rows)
    if total_bytes > 50_000_000_000:
        raise ValueError('Dataset exceeds approved download budget')
    context = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   seed=None, command=[sys.executable, *sys.argv], parameters=vars(args),
                   biological_analysis_performed=False, preregistration_status='DRAFT_SEALED',
                   data_hashes={str(p): digest(p) for p in [Path(__file__),
                       Path(__file__).with_name('w4_external_data_scifate.py'), Path(args.manifest)]})
    with ThreadPoolExecutor(max_workers=2) as pool:
        downloads = list(pool.map(lambda row: fetch(row, raw), rows))
    context['data_hashes'].update({d['path']: d['sha256'] for d in downloads})
    write_json(output / 'downloads.json', dict(**context, files=downloads,
               runtime_seconds=time.monotonic()-started))
    samples, qc = [], []
    for sample in sorted({r['sample'] for r in rows}):
        selected = {r['role']: r for r in downloads if r['sample'] == sample}
        new, nb, ng = read_gene_table(selected['new']['path'])
        old, ob, og = read_gene_table(selected['old']['path'])
        if set(nb) != set(ob) or set(ng) != set(og):
            raise ValueError(f'New/old gene or barcode set mismatch: {sample}')
        old = old[ob.get_indexer(nb)][:, og.get_indexer(ng)]
        total = new + old
        minute = int(selected['new']['time_minutes'])
        obs = pd.DataFrame({'original_barcode': nb, 'sample': sample, 'time_minutes': minute},
                           index=[f'{sample}:{b}' for b in nb])
        obs['total_counts'] = np.asarray(total.sum(1)).ravel()
        obs['new_counts'] = np.asarray(new.sum(1)).ravel()
        obs['old_counts'] = np.asarray(old.sum(1)).ravel()
        data = ad.AnnData(X=total, obs=obs, var=pd.DataFrame(index=ng))
        data.layers['new'], data.layers['old'], data.layers['total'] = new, old, total
        samples.append(data)
        qc.append(dict(sample=sample, time_minutes=minute, cells=len(nb), genes=len(ng),
                       new_old_barcode_match_fraction=1.0, new_old_gene_match_fraction=1.0,
                       new_old_same_cell_order=bool(nb.equals(ob)), new_old_same_gene_order=bool(ng.equals(og)),
                       total_count_median=float(obs['total_counts'].median()),
                       new_count_median=float(obs['new_counts'].median()),
                       old_count_median=float(obs['old_counts'].median()),
                       total_zero_cells=int((obs['total_counts']==0).sum())))
        print(json.dumps({'status':'sample_technical_qc_complete','sample':sample,'cells':len(nb)}), flush=True)
    # Outer union preserves any genes present in only one published timepoint.
    joined = ad.concat(samples, join='outer', merge='same', fill_value=0)
    if not joined.obs_names.is_unique or not joined.var_names.is_unique:
        raise ValueError('Concatenated cell/gene identities are not unique')
    joined.uns['w4_modalities'] = dict(status='new_old_total_only_no_velocity_estimated',
        spliced_available=False, unspliced_available=False,
        new_semantics='C: detected labeled RNA', old_semantics='T: unlabeled old RNA, NOT total',
        total_semantics='C + T', original_accession='GSE141851')
    processed = root / 'processed'; processed.mkdir(parents=True, exist_ok=True)
    target = processed / 'scntseq_neuron_new_old_total_only.h5ad'
    if target.exists():
        raise FileExistsError(f'Will not replace previous processed release: {target}')
    joined.write_h5ad(target, compression='gzip')
    pd.DataFrame(qc).to_csv(output / 'technical_qc_by_time.csv', index=False)
    summary = dict(**context, status='LABELING_READY_GENE_SU_MISSING', cells=joined.n_obs,
        genes=joined.n_vars, downloaded_bytes=total_bytes, times_minutes=[0,15,30,60,120],
        gene_layers=['new','old','total'], missing_gene_layers=['spliced','unspliced'],
        population_filter='None; all GEO published barcodes retained, no response-based selection.',
        raw_reads_downloaded=False, processed_path=str(target), processed_sha256=digest(target),
        runtime_seconds=time.monotonic()-started)
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
