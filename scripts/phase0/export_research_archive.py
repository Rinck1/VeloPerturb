"""Inventory publishable experiment records; never include cell matrices or weights."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
RECORD_SUFFIXES = {'.md','.csv','.json','.jsonl','.xml','.log','.png','.pdf'}
SNAPSHOT_SUFFIXES = {'.py','.yaml','.yml'}
SECRET_PATTERNS = {
    'private_key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'github_credential': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})'),
    'aws_access_key': re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'openai_credential': re.compile(rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{35,}'),
    'url_password': re.compile(rb'https?://[^\s/:@]+:[^\s/@]+@'),
    'signed_access_url': re.compile(rb'(?:X-Amz-Signature|X-Goog-Signature|access_token)=[A-Za-z0-9%_-]{16,}',re.I),
    'bearer_credential': re.compile(rb'Bearer\s+[A-Za-z0-9_.-]{30,}'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inventory(first,last):
    records = [];omitted = Counter();issues = []
    for directory in sorted((ROOT/'outputs').iterdir()):
        match = re.match(r'w(\d+)(?:_|$)',directory.name)
        if not directory.is_dir() or not match or not first <= int(match.group(1)) <= last:
            continue
        for path in sorted(directory.rglob('*')):
            if not path.is_file():continue
            relative = path.relative_to(ROOT).as_posix()
            if path.is_symlink():
                omitted['symlinks'] += 1;continue
            allowed = path.suffix.lower() in RECORD_SUFFIXES or (
                'code_snapshot' in path.parts and path.suffix.lower() in SNAPSHOT_SUFFIXES)
            if not allowed:
                omitted[path.suffix.lower() or '<no_suffix>'] += 1;continue
            if path.stat().st_size > 5*1024**2:
                issues.append(dict(path=relative,reason='record_exceeds_5_MiB'));continue
            before = path.stat();data = path.read_bytes();after = path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                raise RuntimeError(f'Record changed during export: {relative}')
            if path.suffix.lower() not in {'.png','.pdf'}:
                for name,pattern in SECRET_PATTERNS.items():
                    if pattern.search(data):issues.append(dict(path=relative,reason=name))
                if path.suffix.lower()=='.csv':
                    header = data.splitlines()[0] if data.splitlines() else b''
                    if re.search(rb'cell_id|cell_barcode|clone_id|patient_id',header,re.I):
                        issues.append(dict(path=relative,reason='cell_or_patient_level_csv'))
            records.append(dict(path=relative,bytes=len(data),sha256=digest(data)))
    if issues:
        print(json.dumps(dict(status='REVIEW_REQUIRED',issues=issues),indent=2))
        raise SystemExit(2)
    return records,dict(omitted)


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--first-phase',type=int,default=5)
    parser.add_argument('--last-phase',type=int,default=13);parser.add_argument('--manifest',required=True)
    parser.add_argument('--stage',action='store_true');args = parser.parse_args()
    manifest_path = (ROOT/args.manifest).resolve()
    if not manifest_path.is_relative_to(ROOT/'docs') or manifest_path.exists():
        raise ValueError('Use a NEW manifest under docs; existing exports are preserved')
    records,omitted = inventory(args.first_phase,args.last_phase)
    result = dict(status='RESEARCH_RECORD_EXPORT_READY',created_utc=datetime.now(timezone.utc).isoformat(),
        first_phase=args.first_phase,last_phase=args.last_phase,files=len(records),
        total_bytes=sum(r['bytes'] for r in records),records=records,
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        publication_scope='experiment records, aggregate results, logs, tests, figures and exact code snapshots',
        omitted_file_types=omitted,raw_cell_matrices_included=False,model_weights_included=False,
        sensitivity_scan='credential patterns and cell-level CSV headers; manual review also required')
    manifest_path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    if args.stage:
        # Hashes are rechecked immediately before this explicit, non-destructive staging.
        for record in records:
            if digest((ROOT/record['path']).read_bytes())!=record['sha256']:
                raise RuntimeError('Record changed after export inventory')
        for offset in range(0,len(records),200):
            subprocess.run(['git','add','-f','--',*[r['path'] for r in records[offset:offset+200]]],cwd=ROOT,check=True)
        subprocess.run(['git','add','--',str(manifest_path.relative_to(ROOT))],cwd=ROOT,check=True)
    print(json.dumps({k:result[k] for k in ('status','files','total_bytes','omitted_file_types')},indent=2))


if __name__=='__main__':main()
