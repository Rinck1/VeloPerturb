"""Download sci-fate A549 public matrices and perform *technical-only* QC.

No HVG/PCA, kinetic fitting, treatment effects, trajectory, or prediction runs.
GEO supplies total/new gene matrices, NOT gene-wise spliced/unspliced matrices.
Those missing modalities are never synthesized from annotation-level totals.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fetch(row, root):
    path = root / row['filename']
    expected = int(row['expected_bytes'])
    if not path.exists():
        partial = path.with_suffix(path.suffix + '.part')
        # Exclusive creation avoids clobbering a concurrently running transfer.
        with partial.open('xb') as f:
            with urllib.request.urlopen(row['url'], timeout=60) as response:
                count = 0
                while block := response.read(1 << 20):
                    count += len(block)
                    if count > expected:
                        raise ValueError('Remote file exceeds registered byte size')
                    f.write(block)
        if partial.stat().st_size != expected:
            raise ValueError(f'Truncated download retained for audit: {partial}')
        # Gzip CRC is checked to EOF before accepting the transfer.
        with gzip.open(partial, 'rb') as f:
            for _ in iter(lambda: f.read(1 << 20), b''):
                pass
        partial.rename(path)
    if path.stat().st_size != expected:
        raise ValueError(f'Existing file does not match manifest size: {path}')
    return dict(**row, path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='configs/w4_external_data_scifate_files.csv')
    p.add_argument('--data-root', default='/data/yuchang/veloroute_w4/scifate')
    p.add_argument('--output', required=True)
    p.add_argument('--skip-build', action='store_true', help='Download and metadata QC only')
    args = p.parse_args()
    start = time.monotonic()
    root, output = Path(args.data_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    raw = root / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest).open() as f:
        rows = list(csv.DictReader(f))
    if sum(int(r['expected_bytes']) for r in rows) > 50_000_000_000:
        raise ValueError('Dataset budget exceeds 50 GB; explicit approval required')
    common = dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  seed=None, command=[sys.executable, *sys.argv], parameters=vars(args),
                  biological_analysis_performed=False, preregistration_status='DRAFT_SEALED',
                  data_hashes={str(Path(args.manifest)): digest(args.manifest),
                               str(Path(__file__)): digest(__file__)})
    with ThreadPoolExecutor(max_workers=2) as executor:
        files = list(executor.map(lambda row: fetch(row, raw), rows))
    common['data_hashes'].update({f['path']: f['sha256'] for f in files})
    write_json(output / 'downloads.json', dict(**common, files=files, runtime_seconds=time.monotonic()-start))
    print(json.dumps(dict(status='PUBLIC_MATRICES_DOWNLOADED', bytes=sum(f['bytes'] for f in files))), flush=True)

    import numpy as np
    import pandas as pd
    byrole = {f['role']: Path(f['path']) for f in files}
    cells = pd.read_csv(byrole['total_cell_annotation']).set_index('sample', verify_integrity=True)
    new_cells = pd.read_csv(byrole['new_cell_annotation']).set_index('sample', verify_integrity=True)
    genes = pd.read_csv(byrole['total_gene_annotation']).set_index('gene_id', verify_integrity=True)
    new_genes = pd.read_csv(byrole['new_gene_annotation']).set_index('gene_id', verify_integrity=True)
    if set(cells.index) != set(new_cells.index) or set(genes.index) != set(new_genes.index):
        raise ValueError('Total/new barcode or gene sets differ; explicit alignment decision needed')
    if not cells['treatment_time'].equals(new_cells.loc[cells.index, 'treatment_time']):
        raise ValueError('Total/new metadata disagree on sampling time')
    qc = dict(status='TOTAL_NEW_READY_US_MISSING', cells=len(cells), genes=len(genes),
              total_new_barcode_overlap_fraction=1.0, total_new_gene_overlap_fraction=1.0,
              same_cell_order=bool(cells.index.equals(new_cells.index)),
              same_gene_order=bool(genes.index.equals(new_genes.index)),
              available_gene_layers=['total', 'new'], missing_gene_layers=['spliced', 'unspliced'],
              population_filter='None; raw GEO cells retained. Published 6680-cell subset not assumed.',
              new_semantics='Detected labeled RNA over preceding 2h; not unspliced RNA or net velocity.',
              annotation_semantics='all_exon/all_intron are cell-level UMI totals, not gene-wise S/U.',
              response_columns_or_effects_computed=False)
    # QC is restricted to cell/library sizes and barcode annotation completeness.
    summaries = []
    for label, group in cells.groupby('treatment_time', observed=True):
        ratio = group['all_intron'] / (group['all_intron'] + group['all_exon'])
        summaries.append(dict(time=str(label), cells=len(group),
                              annotation_exon_umi_median=float(group['all_exon'].median()),
                              annotation_intron_umi_median=float(group['all_intron'].median()),
                              annotation_intron_fraction_median=float(ratio.median()),
                              annotation_total_umi_median=float(group['all_reads'].median())))
    pd.DataFrame(summaries).to_csv(output / 'technical_qc_by_time.csv', index=False)

    # Archive raw-run size metadata only: no FASTQ download is performed here.
    ena_url = ('https://www.ebi.ac.uk/ena/portal/api/filereport?accession=SRX5848248&result=read_run'
               '&fields=run_accession,fastq_ftp,fastq_bytes,fastq_md5,library_name&format=tsv')
    with urllib.request.urlopen(ena_url, timeout=60) as r:
        ena = r.read()
    with (output / 'scifate_raw_runs.tsv').open('xb') as f:
        f.write(ena)
    raw_rows = list(csv.DictReader(io.StringIO(ena.decode()), delimiter='\t'))
    raw_bytes = sum(int(n) for r in raw_rows for n in r['fastq_bytes'].split(';'))
    qc.update(raw_run_count=len(raw_rows), raw_fastq_compressed_bytes=raw_bytes,
              raw_download_requires_approval=raw_bytes > 50_000_000_000,
              raw_fastq_downloaded=False, raw_manifest_url=ena_url)
    common['data_hashes'][str(output / 'scifate_raw_runs.tsv')] = digest(output / 'scifate_raw_runs.tsv')
    if not args.skip_build:
        import anndata as ad
        from scipy import io as scipy_io
        from threadpoolctl import threadpool_limits
        # scipy may otherwise spawn one parser thread per visible CPU.
        with threadpool_limits(limits=4):
            with gzip.open(byrole['total_counts'], 'rb') as f:
                total = scipy_io.mmread(f).T.tocsr()
            with gzip.open(byrole['new_counts'], 'rb') as f:
                new = scipy_io.mmread(f).T.tocsr()
        if total.shape != (len(cells), len(genes)) or new.shape != (len(new_cells), len(new_genes)):
            raise ValueError('Matrix dimensions do not match annotations')
        new = new[new_cells.index.get_indexer(cells.index)][:, new_genes.index.get_indexer(genes.index)]
        if (total.data < 0).any() or (new.data < 0).any():
            raise ValueError('Negative count in released matrix')
        difference = total - new
        qc['new_count_exceeds_total_entries'] = int(np.sum(difference.data < 0))
        qc['total_matrix_nnz'] = int(total.nnz)
        qc['new_matrix_nnz'] = int(new.nnz)
        cells = cells.copy()
        cells['time_hours'] = cells['treatment_time'].str.removesuffix('h').astype(int)
        cells['matrix_total_counts'] = np.asarray(total.sum(axis=1)).ravel()
        cells['matrix_new_counts'] = np.asarray(new.sum(axis=1)).ravel()
        # Never label total counts as spliced or new counts as unspliced.
        data = ad.AnnData(X=total, obs=cells, var=genes)
        data.layers['total'] = total
        data.layers['new'] = new
        data.uns['w4_modalities'] = {'spliced_available': False, 'unspliced_available': False,
                                     'status': 'total_new_only_no_velocity_estimated'}
        processed = root / 'processed'
        processed.mkdir(parents=True, exist_ok=True)
        target = processed / 'scifate_total_new_only.h5ad'
        if target.exists():
            raise FileExistsError(f'Refusing to replace existing processed release: {target}')
        data.write_h5ad(target, compression='gzip')
        qc['processed_path'] = str(target)
        qc['processed_sha256'] = digest(target)
    write_json(output / 'summary.json', dict(**common, **qc, runtime_seconds=time.monotonic()-start))
    print(json.dumps(qc), flush=True)


if __name__ == '__main__':
    main()
