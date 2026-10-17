"""Export small immutable technical-QC evidence while retaining original hashes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from veloroute.artifacts import save_json, sha256, utc_now


def main():
    start = time.monotonic()
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source = args.source.resolve()
    summary = json.loads((source/'summary.json').read_text())
    provenance = json.loads((source/'provenance.json').read_text())
    if provenance['status'] != 'complete' or summary.get('replication_effects_read') is not False:
        raise ValueError('Only completed technical-only stages can be exported')
    names = ('summary.json', 'provenance.json', 'config.yaml', 'condition_cell_counts.csv')
    records = []
    for name in names:
        path = source/name
        if not path.is_file() or path.stat().st_size > 2_000_000:
            raise ValueError(f'Missing or unexpectedly large technical artifact: {name}')
        digest = sha256(path)
        if name != 'provenance.json' and provenance['outputs'].get(name) != digest:
            raise ValueError(f'Completed technical artifact changed: {name}')
        records.append(dict(name=name, original_path=str(path), original_sha256=digest,
                            size_bytes=path.stat().st_size))
    args.output.mkdir(parents=True, exist_ok=False)
    for row in records:
        copied = args.output/row['name']
        shutil.copyfile(row['original_path'], copied)
        if sha256(copied) != row['original_sha256']:
            raise ValueError('Copy verification failed')
    manifest = dict(status='TECHNICAL_EVIDENCE_EXPORTED', original_root=str(source),
                    created_utc=utc_now(), seed=None, argv=sys.argv,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    elapsed_seconds=time.monotonic()-start, files=records,
                    large_original_artifacts_not_copied=True,
                    original_provenance_paths_refer_to_original_root=True,
                    replication_effects_read=False)
    save_json(args.output/'export_manifest.json', manifest)
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
