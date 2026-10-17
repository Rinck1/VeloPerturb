"""Exact technical-only alignment of GEO labeling counts and official Dynamo S/U.

Keep all available splicing cells without expression/response/cell-type selection.
Use exact full barcode and exact unique gene names. Do not infer missing genes.
Historical author-selected/normalized labeling is audited, never used as counts.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from w4_external_data_scifate import digest, write_json


def strings(group, key):
    return pd.Index(group[key].asstr()[:])


def frame_index(group):
    return strings(group, group.attrs['_index'])


def read_csr(group):
    encoding = group.attrs['encoding-type']
    if encoding != 'csr_matrix':
        raise ValueError(f'Expected explicit CSR matrix, got {encoding}')
    result = sparse.csr_matrix((group['data'][:], group['indices'][:], group['indptr'][:]),
                               shape=tuple(group.attrs['shape']))
    if not np.isfinite(result.data).all() or (result.data < 0).any():
        raise ValueError('Nonfinite or negative matrix entry')
    return result


def count_qc(matrix):
    return dict(shape=list(matrix.shape), dtype=str(matrix.dtype), nnz=int(matrix.nnz),
                noninteger_nonzero_entries=int(np.count_nonzero(matrix.data != np.rint(matrix.data))),
                negative_entries=int(np.count_nonzero(matrix.data < 0)),
                nonfinite_entries=int(np.count_nonzero(~np.isfinite(matrix.data))),
                max_entry=float(matrix.data.max()) if matrix.nnz else 0.)


def barcode_time(barcode):
    if re.fullmatch(r'Neu-4sU-only-run1n2_[ACGTN]+', barcode):
        return 0
    match = re.fullmatch(r'Neu-4sU-Kcl-(15|30|60|120)min-run1n2_[ACGTN]+', barcode)
    if not match:
        raise ValueError(f'Unrecognized complete sample/barcode encoding: {barcode}')
    return int(match.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geo', default='/data/yuchang/veloroute_w4/scntseq/processed/scntseq_neuron_new_old_total_only.h5ad')
    parser.add_argument('--splicing', default='/data/yuchang/veloroute_w4/scntseq/dynamo_official/neuron_splicing_4_11.h5ad')
    parser.add_argument('--historical-labeling', default='/data/yuchang/veloroute_w4/scntseq/dynamo_official/neuron_labeling.h5ad')
    parser.add_argument('--processed', default='/data/yuchang/veloroute_w4/scntseq/processed/scntseq_geo_labeling_dynamo_su_exact.h5ad')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    target = Path(args.processed)
    if target.exists():
        raise FileExistsError('Never replace an existing technical release')
    common = dict(parameters=vars(args), command=[sys.executable, *sys.argv], seed=None,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  biological_analysis_performed=False, prediction_or_velocity_fit_performed=False,
                  data_hashes={str(p): digest(p) for p in [Path(__file__),
                      Path(__file__).with_name('w4_external_data_scifate.py'),
                      Path(args.geo), Path(args.splicing), Path(args.historical_labeling)]})
    geo = ad.read_h5ad(args.geo)
    geo_barcodes = pd.Index(geo.obs['original_barcode'].astype(str))
    if not geo_barcodes.is_unique or not geo.var_names.is_unique:
        raise ValueError('GEO cell/gene keys must be unique')
    with h5py.File(args.splicing, 'r') as handle:
        su_barcodes, su_genes = frame_index(handle['obs']), frame_index(handle['var'])
        if not su_barcodes.is_unique or not su_genes.is_unique:
            raise ValueError('S/U cell/gene keys must be unique')
        if not strings(handle['obs'], 'cellname').equals(su_barcodes):
            raise ValueError('Splicing cellname disagrees with index')
        su = {name: read_csr(handle['layers'][name]) for name in ('spliced', 'unspliced')}
    layer_qc = {name: count_qc(matrix) for name, matrix in su.items()}
    layer_qc.update({f'geo_{name}': count_qc(geo.layers[name]) for name in ('new', 'old', 'total')})
    if any(q['noninteger_nonzero_entries'] for q in layer_qc.values()):
        raise ValueError('Input advertised as raw counts contains fractional entries')
    cells = geo_barcodes.get_indexer(su_barcodes)
    missing_cells = su_barcodes[cells < 0]
    pd.DataFrame({'missing_splicing_barcode': missing_cells}).to_csv(output/'missing_barcodes.csv', index=False)
    if len(missing_cells):
        raise ValueError('Splicing cells missing from raw GEO; do not force partial cohort alignment')
    expected_time = np.array([barcode_time(str(b)) for b in su_barcodes])
    observed_time = geo.obs.iloc[cells]['time_minutes'].to_numpy(dtype=int)
    if not np.array_equal(expected_time, observed_time):
        raise ValueError('Exact barcode matches disagree on sampling time')
    genes = geo.var_names[geo.var_names.isin(su_genes)]
    gi, si = geo.var_names.get_indexer(genes), su_genes.get_indexer(genes)
    if (gi < 0).any() or (si < 0).any():
        raise ValueError('Gene intersection failed')
    pd.DataFrame({'gene_identifier': geo.var_names[~geo.var_names.isin(su_genes)]}).to_csv(output/'geo_only_genes.csv', index=False)
    pd.DataFrame({'gene_identifier': su_genes[~su_genes.isin(geo.var_names)]}).to_csv(output/'splicing_only_genes.csv', index=False)
    pd.DataFrame({'gene_identifier': genes, 'geo_column': gi, 'splicing_column': si}).to_csv(output/'gene_alignment.csv', index=False)
    layers = {name: geo.layers[name][cells][:,gi].tocsr() for name in ('new','old','total')}
    layers.update({name: matrix[:,si].tocsr() for name,matrix in su.items()})
    if (layers['total'] - layers['new'] - layers['old']).nnz:
        raise ValueError('GEO total differs from new + old')
    obs = geo.obs.iloc[cells][['sample','time_minutes']].copy()
    obs['geo_release_cell_id'] = obs.index.astype(str)
    obs.index = su_barcodes
    obs['original_barcode'] = su_barcodes
    for name,matrix in layers.items():
        obs[f'{name}_counts_common_genes'] = np.asarray(matrix.sum(1)).ravel()
    paired = ad.AnnData(X=layers['total'], obs=obs, var=pd.DataFrame(index=genes))
    for name,matrix in layers.items():
        paired.layers[name] = matrix
    paired.uns['w4_modalities'] = dict(status='technical_exact_alignment_only_no_velocity',
        geo_total_semantics='GEO C + T; new + old; gene-wise exonic/intronic reads combined',
        splicing_semantics='Separate official Dynamo splicing quantification; not assumed equal to GEO total',
        gene_alignment='Exact unique gene-symbol intersection, no alias remapping or imputation',
        population='All cells available in official splicing release, exactly matched to GEO; upstream SU selection unknown',
        additional_response_filter=False, additional_celltype_filter=False, additional_hvg_filter=False,
        historical_3060_labeling_subset_used=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    paired.write_h5ad(target, compression='gzip')
    mapping = pd.DataFrame({'splicing_barcode': su_barcodes, 'splicing_row': np.arange(len(su_barcodes)),
                            'geo_cell_id': geo.obs_names[cells], 'geo_row': cells,
                            'time_minutes': observed_time})
    mapping.to_csv(output/'barcode_alignment.csv', index=False)
    rows=[]
    for minute in sorted(set(observed_time)):
        subset=obs[obs.time_minutes==minute]
        row=dict(time_minutes=int(minute), geo_available_cells=int((geo.obs.time_minutes==minute).sum()),
                 exact_paired_cells=len(subset))
        row['geo_cell_coverage']=row['exact_paired_cells']/row['geo_available_cells']
        for name in layers:
            row[f'{name}_library_median_common_genes']=float(subset[f'{name}_counts_common_genes'].median())
        rows.append(row)
    pd.DataFrame(rows).to_csv(output/'technical_qc_by_time.csv', index=False)
    # Audit historical scaled labeling separately; never feed it into paired counts.
    with h5py.File(args.historical_labeling, 'r') as handle:
        label_cells,label_genes=frame_index(handle['obs']),frame_index(handle['var'])
        historical=dict(cells=len(label_cells),genes=len(label_genes),
            exact_splicing_cell_overlap=len(label_cells.intersection(su_barcodes)),
            exact_splicing_gene_overlap=len(label_genes.intersection(su_genes)),
            layers={name:count_qc(read_csr(handle['layers'][name])) for name in ('new','total')},
            biological_fields_not_read=['cell_type','activity_genes'], used_in_paired_release=False)
    diff=layers['total']-layers['spliced']-layers['unspliced'];diff.eliminate_zeros()
    result=dict(**common,status='TECHNICAL_SU_LABELING_PAIRING_READY_SCIENTIFIC_PROTOCOL_SEALED',
        cells=paired.n_obs,genes=paired.n_vars,geo_cells=geo.n_obs,geo_genes=geo.n_vars,
        splicing_cells=len(su_barcodes),splicing_genes=len(su_genes),
        splicing_barcode_coverage=1.,geo_barcode_coverage=paired.n_obs/geo.n_obs,
        exact_time_conflicts=0,duplicate_cell_keys=0,duplicate_gene_keys=0,
        discarded_geo_only_genes=geo.n_vars-len(genes),discarded_splicing_only_genes=len(su_genes)-len(genes),
        original_layer_qc=layer_qc,historical_labeling_audit=historical,
        geo_total_vs_spliced_plus_unspliced_nonidentical_matrix_entries=int(diff.nnz),
        counts_definition_caution='Different quantification pipelines; total is NOT defined as S + U.',
        processed_path=str(target),processed_sha256=digest(target),
        runtime_seconds=time.monotonic()-started)
    write_json(output/'summary.json', result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
