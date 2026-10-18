"""Audit the document-only reorganization and immutable research archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from organize_research_docs import LINK,ROOT,local_target


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def navigation_errors():
    errors=[]
    paths=[ROOT/'README.md',ROOT/'HANDOFF.md']
    paths+=list((ROOT/'docs/current').glob('*.md'))
    paths+=list((ROOT/'docs').rglob('README.md'))+list((ROOT/'PLANS').rglob('README.md'))
    for path in set(paths):
        for match in LINK.finditer(path.read_text()):
            target,_=local_target(path.parent,match.group(2))
            if target is not None and not (ROOT/target).exists():
                errors.append(dict(path=path.relative_to(ROOT).as_posix(),target=target))
    return errors


def validate(*,full_archive=False):
    manifest=json.loads((ROOT/'docs/organization_manifest_20261017.json').read_text())
    errors=navigation_errors();categories={};aliases=0
    for row in manifest['records']:
        path=ROOT/row['path'];categories[row['status']]=categories.get(row['status'],0)+1
        if not path.is_file():
            errors.append(dict(path=row['path'],reason='missing_canonical_document'));continue
        if digest(path)!=row['organized_sha256']:
            errors.append(dict(path=row['path'],reason='organized_document_changed'))
        if row['status']=='byte_exact_front_page_snapshot' and digest(path)!=row['original_sha256']:
            errors.append(dict(path=row['path'],reason='historical_front_page_not_byte_exact'))
        if row['compatibility_alias']:
            aliases+=1
            old=ROOT/row['original_path']
            if not old.exists() or '兼容入口' not in old.read_text():
                errors.append(dict(path=row['original_path'],reason='missing_compatibility_alias'))
    for name,expected in manifest['protected_sha256'].items():
        if not (ROOT/name).is_file() or digest(ROOT/name)!=expected:
            errors.append(dict(path=name,reason='protected_protocol_or_scientific_code_changed'))
    count=0
    if full_archive:
        archive=json.loads((ROOT/'docs/research_archive_20261017.json').read_text())
        for row in archive['records']:
            count+=1
            if not (ROOT/row['path']).is_file() or digest(ROOT/row['path'])!=row['sha256']:
                errors.append(dict(path=row['path'],reason='immutable_experiment_record_changed'))
    return dict(status='DOCUMENT_ORGANIZATION_VERIFIED' if not errors else 'DOCUMENT_ORGANIZATION_REVIEW_REQUIRED',
        moved_documents=len(manifest['records']),categories=categories,compatibility_aliases=aliases,
        protected_files=len(manifest['protected_sha256']),archive_records_verified=count,
        current_navigation_valid=not navigation_errors(),errors=errors,
        training_launched=False,scientific_results_changed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--full-archive',action='store_true')
    parser.add_argument('--output');args=parser.parse_args();result=validate(full_archive=args.full_archive)
    if args.output:
        output=(ROOT/args.output).resolve()
        if not output.is_relative_to(ROOT/'docs') or output.exists():raise ValueError('Use a new docs audit output')
        output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if result['errors']:raise SystemExit(2)
