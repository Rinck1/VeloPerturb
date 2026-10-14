"""Restore official Day0 clone calls by cell barcode; count coverage, not outcomes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sys
import time

from veloroute.artifacts import Run, save_csv, save_json, sha256
from w3_6b_fetch_author_metadata import download, metadata_slot

DAY0_URL = 'https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM9044nnn/GSM9044664/suppl/GSM9044664_D0_Clonocluster.rds'
DAY0_BYTES = 1062913437


def mapping_reasons(labels, assignments):
    result = Counter()
    for row in labels:
        local = assignments.get(row['cell_barcode'])
        if local is None:
            result['unassigned'] += 1
        elif int(local['n_barcodes']) != 1:
            result['multiple_barcodes'] += 1
        elif local['clone_barcode'] != row['author_clone']:
            result['single_clone_mismatch'] += 1
        else:
            result['single_clone_exact'] += 1
    return dict(result)


def official_cohort(source, target, minimum=3):
    source_count = Counter(r['author_clone'] for r in source if r['quant_barcode_match'])
    target_count = Counter(); groups = defaultdict(set)
    for row in target:
        if row['quant_barcode_match'] in (True, 'True') and row['valid_author_group'] in (True, 'True'):
            target_count[row['author_clone']] += 1
            groups[row['author_clone']].add(row['bc_group'])
    conflict = [clone for clone, labels in groups.items() if len(labels) != 1]
    if conflict:
        raise ValueError('Official clone group conflicts; cannot silently choose a label')
    return [dict(clone=clone, group=next(iter(groups[clone])),
                 source_cells=source_count[clone], target_cells=target_count[clone])
            for clone in sorted(source_count.keys() & target_count.keys())
            if source_count[clone] >= minimum and target_count[clone] >= minimum]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True)
    args = ap.parse_args(); start = time.monotonic()
    root = Path('/data/yuchang/veloroute_ucheck_20260915')
    target = Path('outputs/w3_6b_metadata_20261014_v1/author_cell_labels.csv')
    target_prov = json.loads((target.parent / 'provenance.json').read_text())
    if target_prov['status'] != 'complete' or sha256(target) != target_prov['outputs'][target.name]:
        raise ValueError('Official Day21 label artifact changed')
    source_assign = root / 'clones/SRR33960310/cell_clone_assignments.csv'
    target_assign = root / 'clones/SRR33960308/cell_clone_assignments.csv'
    source_rows = root / 'quant_merlin310/af_quant/alevin/quants_mat_rows.txt'
    c = dict(url=DAY0_URL, expected_bytes=DAY0_BYTES, metadata_only=True,
        minimum_cells_per_author_side=3, direct_cell_barcode_join=True, labels_used_to_choose_mapping=False)
    with Run(args.output, stage='W3_6b_author_clone_mapping_audit', kind='engineering', config=c,
             inputs=[Path(__file__), Path(__file__).with_name('w3_6b_fetch_author_metadata.py'),
                     target, source_assign, target_assign, source_rows]) as run:
        with target.open() as stream:
            day21 = list(csv.DictReader(stream))
        with source_assign.open() as stream:
            local0 = {r['cell_barcode']: r for r in csv.DictReader(stream)}
        with target_assign.open() as stream:
            local21 = {r['cell_barcode']: r for r in csv.DictReader(stream)}
        reasons = mapping_reasons(day21, local21)
        save_json(run.directory / 'day21_local_mapping_reasons.json', reasons)
        artifact = Path('/data/yuchang/veloroute_w3_20261014/merlin_author/GSM9044664_D0_Clonocluster.rds')
        download(artifact, run.directory / 'status.json', url=DAY0_URL, expected_bytes=DAY0_BYTES)
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/w3-rdata'))
        import rdata
        if rdata.__version__ != '1.1.0':
            raise ValueError('Unexpected parser version')
        save_json(run.directory / 'status.json', dict(status='parsing_Day0_metadata'))
        parsed = rdata.parser.parse_file(artifact)
        metadata = metadata_slot(parsed, rdata)
        if 'Barcode' not in metadata.columns:
            raise ValueError('Author Day0 object lacks clone calls')
        quant = set(source_rows.read_text().splitlines())
        rows = []
        for cell, record in metadata.iterrows():
            match = re.search(r'([ACGT]{16})(?:-\d+)?$', str(cell))
            barcode = match.group(1) if match else ''
            clone = str(record['Barcode'])
            if not re.fullmatch('[ACGT]{15}', clone):
                raise ValueError('Unexpected official clone syntax; inspect before changing mapping')
            rows.append(dict(author_cell_id=str(cell), cell_barcode=barcode, author_clone=clone,
                quant_barcode_match=barcode in quant))
        if len({r['cell_barcode'] for r in rows}) != len(rows):
            raise ValueError('Official source cell IDs are not unique after parsing')
        save_csv(run.directory / 'author_source_clones.csv', rows)
        cohort = official_cohort(rows, day21, minimum=3)
        save_csv(run.directory / 'official_metadata_cohort.csv', cohort,
                 fields=['clone', 'group', 'source_cells', 'target_cells'])
        summary = dict(status='AUTHOR_MAPPING_COVERAGE_AUDITED', day21_local_mapping_reasons=reasons,
            day0_local_mapping_reasons=mapping_reasons(rows, local0), official_day0_cells=len(rows),
            source_quant_barcode_matches=sum(r['quant_barcode_match'] for r in rows),
            both_author_sides_minimum3_clones=len(cohort),
            official_group_counts=dict(Counter(r['group'] for r in cohort)),
            source_expression_QC_applied=False, future_expression_used=False, biological_increment_tested=False,
            author_rds_path=str(artifact), author_rds_sha256=sha256(artifact),
            runtime_seconds=time.monotonic()-start)
        save_json(run.directory / 'summary.json', summary)
        save_json(run.directory / 'status.json', dict(status='complete'))
        (run.directory / 'RESULTS.md').write_text('# Author clone mapping coverage\n\n' + json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
