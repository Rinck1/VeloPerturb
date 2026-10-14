"""Fetch the official Day21 object and convert only its meta.data slot.

The RDS container includes expression and is parsed as a container; expression
is neither converted to a matrix nor used for labels, axes or model selection.
Downloads go to the data volume, not the nearly full workspace filesystem.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.request import Request, urlopen

from veloroute.artifacts import Run, save_csv, save_json, sha256

URL = 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9044nnn/GSM9044665/suppl/GSM9044665_D21_Clonocluster.rds'
EXPECTED_BYTES = 500555118


def unwrap(obj):
    seen = set()
    while getattr(obj, 'referenced_object', None) is not None:
        if id(obj) in seen:
            raise ValueError('Cyclic reference while reading metadata slot')
        seen.add(id(obj)); obj = obj.referenced_object
    return obj


def metadata_slot(parsed, rdata):
    obj = unwrap(parsed.object)
    node = unwrap(obj.attributes)
    while node is not None and node.info.type == rdata.parser.RObjectType.LIST:
        name = rdata.conversion.convert(unwrap(node.tag)) if node.tag is not None else None
        if name == 'meta.data':
            return rdata.conversion.convert(node.value[0], default_encoding=parsed.extra.encoding)
        node = unwrap(node.value[1])
    raise ValueError('No top-level meta.data slot: inspect the author object before adapting')


def download(path, status):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != EXPECTED_BYTES:
            raise ValueError('Existing final artifact has unexpected size; preserving it')
        return
    part = path.with_suffix(path.suffix + '.part')
    offset = part.stat().st_size if part.exists() else 0
    if offset > EXPECTED_BYTES:
        raise ValueError('Partial artifact exceeds expected file length')
    if shutil.disk_usage(path.parent).free < EXPECTED_BYTES - offset + (1 << 30):
        raise ValueError('Need download size plus 1 GiB headroom on data volume')
    for attempt in range(3):
        offset = part.stat().st_size if part.exists() else 0
        if offset == EXPECTED_BYTES:
            part.rename(path); return
        headers = {'User-Agent': 'VeloPerturb-research-metadata/1.0'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with urlopen(Request(URL, headers=headers), timeout=45) as response:
                if offset and (response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                    raise ValueError('Server did not honor exact resume range; partial file preserved')
                if not offset and response.status != 200:
                    raise ValueError('Unexpected initial download response')
                with part.open('ab' if offset else 'xb') as stream:
                    while block := response.read(8 << 20):
                        if offset + len(block) > EXPECTED_BYTES:
                            raise ValueError('Remote file exceeds recorded GEO file size')
                        stream.write(block); offset += len(block)
                        save_json(status, dict(status='downloading', bytes=offset, expected_bytes=EXPECTED_BYTES))
            if offset != EXPECTED_BYTES:
                raise OSError('Incomplete download')
            part.rename(path); return
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
    raise RuntimeError('Download not completed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--artifact-root', default='/data/yuchang/veloroute_w3_20261014/merlin_author')
    args = ap.parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    if not artifact_root.is_relative_to(Path('/data/yuchang')):
        raise ValueError('Official RDS must be placed on the scoped data volume')
    root = Path('/data/yuchang/veloroute_ucheck_20260915')
    assignments = root / 'clones/SRR33960308/cell_clone_assignments.csv'
    quant_rows = root / 'quant_merlin308/af_quant/alevin/quants_mat_rows.txt'
    config = dict(url=URL, expected_bytes=EXPECTED_BYTES, artifact_root=str(artifact_root),
                  metadata_only=True, rdata_version='1.1.0',
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    start = time.monotonic()
    with Run(args.output, stage='W3_6b_official_metadata_acquisition', kind='engineering', config=config,
             inputs=[Path(__file__), assignments, quant_rows]) as run:
        path = artifact_root / 'GSM9044665_D21_Clonocluster.rds'
        download(path, run.directory / 'status.json')
        save_json(run.directory / 'download_manifest.json', dict(url=URL, path=str(path), bytes=path.stat().st_size, sha256=sha256(path)))
        save_json(run.directory / 'status.json', dict(status='parsing_RDS_metadata'))
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/w3-rdata'))
        import rdata
        if rdata.__version__ != '1.1.0':
            raise ValueError('Unexpected RDS parser version')
        parsed = rdata.parser.parse_file(path)
        metadata = metadata_slot(parsed, rdata)
        save_json(run.directory / 'metadata_schema.json', dict(rows=len(metadata), columns=[str(c) for c in metadata.columns],
            index_examples=[str(c) for c in metadata.index[:3]], parser_version=rdata.__version__))
        if not {'Barcode', 'bc_group'} <= set(metadata.columns):
            raise ValueError('Official metadata has no Barcode/bc_group; do not invent a label')
        quant = set(quant_rows.read_text().splitlines())
        with assignments.open() as stream:
            local = {r['cell_barcode']: r['clone_barcode'] for r in csv.DictReader(stream)
                     if int(r['n_barcodes']) == 1 and r['cell_barcode'] in quant}
        rows = []
        for cell, record in metadata.iterrows():
            match = re.search(r'([ACGT]{16})(?:-\d+)?$', str(cell))
            barcode = match.group(1) if match else ''
            group, clone = str(record['bc_group']), str(record['Barcode'])
            rows.append(dict(author_cell_id=str(cell), cell_barcode=barcode, author_clone=clone,
                bc_group=group, quant_barcode_match=barcode in quant,
                local_single_clone=local.get(barcode, ''), clone_exact_match=local.get(barcode) == clone,
                valid_author_group=group in [f'Barcode group {i}' for i in range(1, 6)]))
        save_csv(run.directory / 'author_cell_labels.csv', rows)
        matched = [r for r in rows if r['quant_barcode_match'] and r['clone_exact_match'] and r['valid_author_group']]
        groups_by_clone = {}
        for r in matched:
            groups_by_clone.setdefault(r['author_clone'], set()).add(r['bc_group'])
        conflicting = sum(len(v) > 1 for v in groups_by_clone.values())
        result = dict(status='OFFICIAL_LABEL_METADATA_EXTRACTED', cells_in_author_metadata=len(rows),
            author_group_counts=dict(Counter(r['bc_group'] for r in rows)),
            local_quant_cell_matches=sum(r['quant_barcode_match'] for r in rows),
            exact_clone_and_valid_group_matches=len(matched), matched_clones=len(groups_by_clone),
            clone_group_conflicts=conflicting, rds_sha256=sha256(path),
            expression_container_parsed=True, expression_matrix_converted_or_used=False,
            biological_increment_tested=False, runtime_seconds=time.monotonic() - start,
            interpretation='Author bc_group is clone-derived. This restores group labels, not independent within-clone fate proportions.')
        save_json(run.directory / 'summary.json', result)
        save_json(run.directory / 'status.json', dict(status='complete'))
        (run.directory / 'RESULTS.md').write_text('# Official MeRLin Day21 label metadata\n\n' + json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
