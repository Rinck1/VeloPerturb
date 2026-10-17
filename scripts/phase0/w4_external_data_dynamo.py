"""Fetch official historical Dynamo scNT-seq releases; technical schema QC only.

No model, velocity fit, biological response, mode discovery, or target scoring.
The historical labeling file is an author-selected subset, not an unbiased cohort.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

from w4_external_data_scifate import digest, write_json


def fetch_h5ad(row, root):
    import h5py

    target = root / row['filename']
    expected = int(row['expected_bytes'])
    if not target.exists():
        partial = target.with_suffix(target.suffix + '.part')
        with partial.open('xb') as stream:
            request = urllib.request.Request(row['url'], headers={'User-Agent': 'VeloRoute-public-technical-QC/1'})
            with urllib.request.urlopen(request, timeout=120) as response:
                count = 0
                while chunk := response.read(1 << 20):
                    count += len(chunk)
                    if count > expected:
                        raise ValueError('Download exceeds registered byte budget')
                    stream.write(chunk)
        if partial.stat().st_size != expected:
            raise ValueError(f'Incomplete download retained: {partial}')
        with h5py.File(partial, 'r'):
            pass
        partial.rename(target)
    if target.stat().st_size != expected:
        raise ValueError('Existing release size disagrees with manifest')
    return dict(**row, path=str(target), bytes=target.stat().st_size, sha256=digest(target))


def schema(path):
    import h5py

    def describe(node):
        if isinstance(node, h5py.Dataset):
            return dict(kind='dataset', shape=list(node.shape), dtype=str(node.dtype),
                        structured_fields=list(node.dtype.names or ()))
        return dict(kind='group', children=list(node.keys()),
                    matrix_shape=[int(x) for x in node.attrs['shape']] if 'shape' in node.attrs else None)

    with h5py.File(path, 'r') as handle:
        result = dict(root_keys=list(handle.keys()))
        for key in ('X', 'obs', 'var'):
            if key in handle:
                result[key] = describe(handle[key])
        result['layers'] = {key: describe(node) for key, node in handle.get('layers', {}).items()}
        # Only names are listed: no embeddings or biological statistics are read.
        result['uns_keys_not_read'] = list(handle.get('uns', {}).keys())
        result['obsm_keys_not_read'] = list(handle.get('obsm', {}).keys())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='configs/w4_external_data_dynamo_files.csv')
    parser.add_argument('--data-root', default='/data/yuchang/veloroute_w4/scntseq/dynamo_official')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    start = time.monotonic()
    output, root = Path(args.output), Path(args.data_root)
    output.mkdir(parents=True, exist_ok=False)
    root.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest).open() as stream:
        rows = list(csv.DictReader(stream))
    if sum(int(row['expected_bytes']) for row in rows) > 50_000_000_000:
        raise ValueError('Dataset exceeds approved 50 GB boundary')
    common = dict(command=[sys.executable, *sys.argv], parameters=vars(args), seed=None,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  biological_analysis_performed=False, prediction_or_velocity_fit_performed=False,
                  data_hashes={args.manifest: digest(args.manifest), __file__: digest(__file__),
                               str(Path(__file__).with_name('w4_external_data_scifate.py')):
                                   digest(Path(__file__).with_name('w4_external_data_scifate.py'))})
    with ThreadPoolExecutor(max_workers=2) as executor:
        files = list(executor.map(lambda row: fetch_h5ad(row, root), rows))
    common['data_hashes'].update({file['path']: file['sha256'] for file in files})
    write_json(output / 'downloads.json', dict(**common, files=files))
    result = dict(**common, status='OFFICIAL_HISTORICAL_H5AD_DOWNLOADED_SCHEMA_ONLY',
                  files={file['role']: schema(file['path']) for file in files},
                  downloaded_bytes=sum(file['bytes'] for file in files),
                  historical_labeling_population='Author-selected excitatory-neuron subset; not prospective QC selection.',
                  runtime_seconds=time.monotonic() - start)
    write_json(output / 'schema.json', result)
    print(json.dumps(result, default=int), flush=True)


if __name__ == '__main__':
    main()
