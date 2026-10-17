"""Bounded public processed S/U acquisition; metadata/schema only, no effects."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import time
import urllib.request

from w4_artifacts import Artifacts, sha256


ROOT = Path(__file__).resolve().parents[2]


def bounded_copy(source, target, *, limit, deadline, progress=None):
    total = 0
    while True:
        if time.monotonic() > deadline:raise TimeoutError('Data acquisition deadline exceeded')
        block = source.read(4*1024*1024)
        if not block:break
        total += len(block)
        if total > limit:raise ValueError('Registered byte budget exceeded; partial retained')
        target.write(block)
        if progress:progress(total)
    return total


def inspect_schema(path):
    import h5py
    from anndata.io import read_elem
    with h5py.File(path,'r') as handle:
        obs,var = read_elem(handle['obs']),read_elem(handle['var'])
        expected = (len(obs),len(var))
        layers = {}
        for name,node in handle['layers'].items():
            encoding = node.attrs.get('encoding-type','array')
            shape = tuple(node.attrs['shape']) if isinstance(node,h5py.Group) else node.shape
            layers[name] = dict(shape=list(shape),encoding=str(encoding))
            if shape != expected:raise ValueError('Layer shape disagrees with cell/gene metadata')
        missing = {'spliced','unspliced'}-set(layers)
        if missing:raise ValueError(f'Missing published raw modalities: {sorted(missing)}')
    # Do not read expression values, inferred velocity, embeddings or cell-type programs.
    grouping = [c for c in ('rep','replicate','timepoint','time','day','condition','treatment') if c in obs]
    cohorts = (obs.groupby(grouping,observed=True,dropna=False).size().rename('cells').reset_index()
               if grouping else None)
    result = dict(cells=len(obs),genes=len(var),layers=layers,obs_columns=obs.columns.tolist(),
        var_columns=var.columns.tolist(),cell_ids_unique=bool(obs.index.is_unique),
        gene_ids_unique=bool(var.index.is_unique),grouping_columns=grouping,
        header_and_metadata_only=True,expression_count_values_checked=False,
        biological_effects_read=False,ready_for_model_training=False,
        nominal_times_require_protocol_and_author_metadata_reconciliation=True)
    return result,cohorts


def acquire(config_path,output):
    config = json.loads(Path(config_path).read_text())
    if (config['kind'] != 'scifate2_public_SU_technical_only' or config['prediction_or_response_analysis']
            or config['required_layers'] != ['spliced','unspliced']
            or not config['url'].startswith('https://ftp.ncbi.nlm.nih.gov/geo/series/')
            or config['filename'] != 'GSE236512_processed_data_splicing.h5ad.gz'
            or config['expected_bytes'] > config['maximum_download_bytes']
            or config['maximum_download_bytes'] > 1_500_000_000
            or config['maximum_expanded_bytes'] > 8_000_000_000):
        raise ValueError('Unexpected technical acquisition contract')
    run = Artifacts(output,seed=None,inputs=[Path(config_path),Path(__file__),Path(__file__).with_name('w4_artifacts.py')])
    data_root = (ROOT/config['data_root']).resolve()
    if not data_root.is_relative_to(ROOT/'data/scifate2'):
        raise ValueError('Destination outside the registered new dataset')
    data_root.mkdir(parents=True,exist_ok=True)
    compressed = data_root/config['filename']
    expanded = compressed.with_suffix('')
    deadline = time.monotonic()+config['deadline_seconds']
    try:
        if not compressed.exists():
            budget = config['expected_bytes']+config['maximum_expanded_bytes']+config['minimum_free_bytes']
            if shutil.disk_usage(data_root).free < budget:raise OSError('Insufficient reserved space')
            request = urllib.request.Request(config['url'],method='HEAD')
            with urllib.request.urlopen(request,timeout=config['request_timeout_seconds']) as response:
                if int(response.headers['Content-Length']) != config['expected_bytes']:
                    raise ValueError('Official file size changed')
                run.json('remote_metadata.json',dict(url=response.url,headers=dict(response.headers)))
            last_report = 0
            def report(size):
                nonlocal last_report
                if size-last_report >= 64*1024*1024:
                    last_report = size
                    run.json('status.json',dict(status='downloading',bytes=size,total_bytes=config['expected_bytes']))
                    print(json.dumps(dict(stage='download',bytes=size,total=config['expected_bytes'])),flush=True)
            partial = compressed.with_suffix(compressed.suffix+'.part')
            with partial.open('xb') as stream, urllib.request.urlopen(config['url'],timeout=config['request_timeout_seconds']) as response:
                size = bounded_copy(response,stream,limit=config['expected_bytes'],deadline=deadline,progress=report)
            if size != config['expected_bytes']:raise ValueError('Truncated public file; partial retained')
            partial.rename(compressed)
        if compressed.stat().st_size != config['expected_bytes']:raise ValueError('Existing file size mismatch')
        if expanded.exists():raise FileExistsError('Do not overwrite an existing expanded release')
        if shutil.disk_usage(data_root).free < config['maximum_expanded_bytes']+config['minimum_free_bytes']:
            raise OSError('Insufficient expansion reserve')
        run.json('status.json',dict(status='expanding',compressed_bytes=compressed.stat().st_size))
        partial = expanded.with_suffix(expanded.suffix+'.part')
        with gzip.open(compressed,'rb') as source,partial.open('xb') as stream:
            size = bounded_copy(source,stream,limit=config['maximum_expanded_bytes'],deadline=deadline)
        # Gzip CRC verified to EOF before accepting the file.
        partial.rename(expanded)
        schema,cohorts = inspect_schema(expanded)
        if cohorts is not None:cohorts.to_csv(run.directory/'technical_cohorts.csv',index=False)
        result = dict(status='S_U_SCHEMA_METADATA_READY_COUNTS_UNCHECKED',**schema,
            compressed_path=str(compressed),compressed_bytes=compressed.stat().st_size,
            compressed_sha256=sha256(compressed),data_path=str(expanded),data_bytes=size,data_sha256=sha256(expanded))
        run.json('summary.json',result)
        (run.directory/'RESULTS.md').write_text('# sci-FATE2 processed S/U acquisition\n\n'
            'Metadata and layer headers only. Counts, time mapping, cohort QC and forecasting protocol remain to be checked.\n\n'
            +json.dumps(result,indent=2)+'\n')
        run.json('status.json',dict(status='complete'))
        print(json.dumps(result),flush=True)
    except Exception as exc:
        run.json('status.json',dict(status='failed',error_type=type(exc).__name__,error=str(exc),partial_files_retained=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--config',default='configs/w5_scifate2_technical_v1.json')
    parser.add_argument('--output',required=True);args = parser.parse_args();acquire(args.config,args.output)
