"""Freeze the usable source/official-label overlap; do not fit a fate model."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from veloroute.artifacts import Run, save_csv, save_json, sha256


def verified_csv(root, name):
    root = Path(root)
    prov = json.loads((root / 'provenance.json').read_text())
    if prov['status'] != 'complete' or sha256(root / name) != prov['outputs'][name]:
        raise ValueError('Input artifact was not complete/hash-bound')
    with (root / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='outputs/w3_6a_source_screen_20261014_v1')
    ap.add_argument('--labels', default='outputs/w3_6b_metadata_20261014_v1')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    source = verified_csv(args.source, 'cohort.csv')
    labels = verified_csv(args.labels, 'author_cell_labels.csv')
    source = {r['clone_id']: int(r['source_cells']) for r in source}
    kept = [r for r in labels if all(r[k] == 'True' for k in ('clone_exact_match', 'quant_barcode_match', 'valid_author_group'))]
    counts = Counter(r['author_clone'] for r in kept)
    group = {}
    for r in kept:
        clone = r['author_clone']
        if clone in group and group[clone] != r['bc_group']:
            raise ValueError('Conflicting author group assignments')
        group[clone] = r['bc_group']
    config = dict(source=args.source, labels=args.labels, minimum_source_cells=3, minimum_target_cells=3,
                  registered_classes=5, registered_folds=10)
    with Run(args.output, stage='W3_6b_official_label_source_cohort', kind='engineering', config=config,
             inputs=[Path(__file__), Path(args.source) / 'cohort.csv', Path(args.labels) / 'author_cell_labels.csv']) as run:
        rows = [dict(clone_id=k, source_cells=source.get(k, 0), exact_author_target_cells=counts[k],
                     author_group=group[k], eligible=source.get(k, 0) >= 3 and counts[k] >= 3) for k in sorted(counts)]
        primary = Counter(r['author_group'] for r in rows if r['eligible'])
        relaxed = Counter(r['author_group'] for r in rows if r['source_cells'] >= 3)
        classes = [f'Barcode group {i}' for i in range(1, 6)]
        feasible = all(primary[g] >= 10 for g in classes)
        result = dict(status='COHORT_READY_FOR_PROTOCOL_REVIEW' if feasible else 'INSUFFICIENT_REGISTERED_FIVE_CLASS_COHORT',
            eligible_clones=sum(primary.values()), primary_group_counts=dict(primary),
            target_ge1_sensitivity_clones=sum(relaxed.values()), target_ge1_group_counts=dict(relaxed),
            missing_classes=[g for g in classes if primary[g] == 0],
            fate_model_trained=False, labels_redefined=False,
            next_dependency='Audit original Day0/Day21 author barcode mapping; do not merge/drop classes to force success')
        save_csv(run.directory / 'clone_overlap.csv', rows)
        save_json(run.directory / 'summary.json', result)
        (run.directory / 'RESULTS.md').write_text('# Official-label source cohort\n\n' + json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
