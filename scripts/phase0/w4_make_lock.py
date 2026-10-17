"""Create a formal W4 lock only after a git-committed protocol freeze.

Reads protocol/QC/embeddings hashes; never fits a source transform, decodes
expression, changes the protocol, or performs biological evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from w4_artifacts import sha256
from w4_arms import CORE_ARMS
from w4_replication import INTERVALS, PREPROCESS, PROTOCOL, SCHEMA, SEEDS, resolved, verify_lock
from w4_wait_calibration import validate_qc


ROOT = Path(__file__).resolve().parents[2]


def build_lock(freeze_commit, qc_path, embeddings_path, *, repo=ROOT):
    start = time.monotonic()
    repo, qc_path = Path(repo).resolve(), Path(qc_path).resolve()
    if not re.fullmatch(r'[0-9a-f]{40}', freeze_commit):
        raise ValueError('Full 40-character freeze commit required')
    content = subprocess.check_output(['git','-C',str(repo),'show',f'{freeze_commit}:{PROTOCOL}'])
    if not re.search(r'^状态：\*\*已冻结\*\*', content.decode(), flags=re.MULTILINE):
        raise ValueError('Protocol is still DRAFT; do not create a formal replication lock')
    digest = hashlib.sha256(content).hexdigest()
    if sha256(repo/PROTOCOL) != digest:
        raise ValueError('Local protocol does not match its frozen git content')
    qc = json.loads(qc_path.read_text())
    validate_qc(qc)
    files = {}
    for day in (2,3,4):
        candidates = [(resolved(p, repo), h) for p,h in qc.get('data_hashes',{}).items()
                      if Path(p).name == f'day{day}.h5ad']
        if len(candidates) != 1:
            raise ValueError(f'QC must identify exactly one day{day} count file')
        path, expected = candidates[0]
        if sha256(path) != expected:
            raise ValueError(f'Count file changed after QC: day{day}')
        files[f'day{day}'] = dict(path=str(path), sha256=expected)
    embeddings_path = resolved(embeddings_path, repo)
    files['embeddings'] = dict(path=str(embeddings_path), sha256=sha256(embeddings_path))
    return dict(schema=SCHEMA, kind='formal_replication', freeze_commit=freeze_commit,
        protocol_path=PROTOCOL, protocol_file_sha256=digest, qc_manifest_path=str(qc_path),
        QC_manifest_sha256=sha256(qc_path), data_hashes=files, H1_TFs=qc['H1_TFs'],
        intervals=[list(x) for x in INTERVALS], seeds=list(SEEDS), arms=list(CORE_ARMS),
        preprocess=dict(PREPROCESS), statistics=dict(bootstrap_repetitions=10000,bootstrap_seed=20261017),
        training_TF_scope='H1_TFs_only',
        git_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        created_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        seed=list(SEEDS), runtime_seconds=time.monotonic()-start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-commit', required=True)
    parser.add_argument('--qc', default='outputs/w4_cohort_qc_v1/summary.json')
    parser.add_argument('--embeddings', default='data/renge/conditions/esm2_3b_v1/conditions.npz')
    parser.add_argument('--output', required=True, help='Exclusive new directory for the immutable lock')
    args = parser.parse_args()
    lock = build_lock(args.freeze_commit, args.qc, args.embeddings)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    pending = output/'protocol_lock.pending.json'
    pending.write_text(json.dumps(lock, indent=2, allow_nan=False)+'\n')
    verify_lock(pending, repo=ROOT)
    final = output/'protocol_lock.json'
    pending.rename(final)
    print(json.dumps(dict(status='FORMAL_LOCK_VERIFIED', protocol_lock=str(final.resolve()),
        freeze_commit=lock['freeze_commit'], H1_TFs=lock['H1_TFs'])), flush=True)


if __name__ == '__main__':
    main()
