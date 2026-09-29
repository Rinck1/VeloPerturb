"""Read-only MeRLin source metadata join; never reads target expression."""
import argparse
import csv
from collections import Counter
from pathlib import Path

from veloroute.artifacts import Run, save_csv, save_json


def audit(split_path, quant_barcodes, assignments_path):
    with Path(split_path).open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['condition'] == 'ctrl']
    sources = {r['cell_barcode'].removesuffix('-1'): r for r in rows}
    if not rows or len(sources) != len(rows):
        raise ValueError('Need nonempty uniquely identified control cells')
    samples = sorted({r['sample_id'] for r in rows})
    if len(samples) != 1:
        raise ValueError('Separate source libraries before barcode matching')
    with Path(quant_barcodes).open() as stream:
        quant = {line.strip().removesuffix('-1') for line in stream if line.strip()}
    with Path(assignments_path).open(newline='') as stream:
        assignments = {r['cell_barcode']: r for r in csv.DictReader(stream)}
    shared = set(sources) & set(assignments)
    known = [b for b in shared if sources[b]['clone_known'].lower() == 'true']
    agreement = []
    for minimum_reads in (1, 2, 3, 5):
        subset = [b for b in known if int(assignments[b]['clone_reads']) >= minimum_reads]
        exact = sum(sources[b]['clone_id'] == assignments[b]['clone_barcode'] for b in subset)
        agreement.append(dict(minimum_reads=minimum_reads, cells=len(subset), exact_matches=exact,
                              exact_fraction=exact/len(subset) if subset else None))
    distances = Counter(sum(x != y for x, y in zip(sources[b]['clone_id'], assignments[b]['clone_barcode']))
                        for b in known if len(sources[b]['clone_id']) == len(assignments[b]['clone_barcode']))
    return dict(status='SOURCE_METADATA_AUDIT_ONLY', source_samples=samples,
        published_control_cells=len(sources), quantified_barcodes=len(quant),
        control_barcodes_in_quant=len(set(sources) & quant),
        published_control_coverage=len(set(sources) & quant)/len(sources),
        shared_assigned_barcodes=len(shared), shared_known_clones=len(known),
        clone_agreement=agreement, clone_hamming_counts=dict(sorted(distances.items())),
        source_expression_identity_verified=False, test_expression_read=False,
        conclusion='Barcode overlap alone does not establish sample identity; verify source RNA concordance and run/library mapping.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True)
    parser.add_argument('--quant-barcodes', required=True)
    parser.add_argument('--assignments', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    paths = [args.split, args.quant_barcodes, args.assignments]
    with Run(args.output, stage='merlin_source_metadata_alignment', kind='engineering', config=vars(args), inputs=paths) as run:
        report = audit(*paths)
        save_json(run.directory/'summary.json', report)
        save_csv(run.directory/'clone_agreement.csv', report['clone_agreement'])
        (run.directory/'RESULTS.md').write_text(
            '# MeRLin source metadata alignment\n\n'
            f"Control coverage: {report['control_barcodes_in_quant']}/{report['published_control_cells']}.\n\n"
            f"Clone agreement: {report['clone_agreement'][0]['exact_fraction']:.4f}.\n\n"
            'This is metadata QC, not a velocity/fate experiment. Source expression identity and library mapping remain to verify.\n')
    print(report)


if __name__ == '__main__':
    main()
