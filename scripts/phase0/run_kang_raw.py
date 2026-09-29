"""Audited Kang BAM acquisition; never substitute expression for unspliced RNA."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys

from veloroute.artifacts import Run, load_config, save_json
from veloroute.download import download_ranges, get_range, remote_info


def audit(path, output, dependency_path, maximum=10000, partial=False):
    sys.path.insert(0, str(dependency_path))
    import pysam
    tags, lengths, bc_lengths, umi_lengths = Counter(), Counter(), Counter(), Counter()
    examples, count = [], 0
    with pysam.AlignmentFile(str(path), 'rb', ignore_truncation=partial) as bam:
        header = bam.header.to_dict()
        for read in itertools.islice(bam, maximum):
            count += 1
            fields = dict(read.tags)
            tags.update(fields.keys())
            lengths[read.query_length] += 1
            if 'CR' in fields: bc_lengths[len(fields['CR'])] += 1
            if 'UR' in fields: umi_lengths[len(fields['UR'])] += 1
            if len(examples) < 3:
                examples.append({k: fields[k] for k in ('CR','CB','UR','UB') if k in fields})
    result = dict(records=count, scope='bounded_prefix_not_full_file', tag_counts=dict(tags),
        query_length_histogram=dict(lengths), raw_barcode_lengths=dict(bc_lengths),
        raw_umi_lengths=dict(umi_lengths), tag_examples=examples, header=header,
        candidate_recoverable=bool(tags['CR'] and tags['UR']), su_ready=False)
    save_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/veloroute_kang_20260914.yaml')
    parser.add_argument('--condition', choices=['ctrl','stim'], required=True)
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--mirror', choices=['ena','ncbi'], default='ena')
    args = parser.parse_args()
    config = load_config(args.config)
    row = next(x for x in config['raw'] if x['condition'] == args.condition)
    row = dict(row)
    if args.mirror == 'ncbi': row['url'] = row['ncbi_url']
    root = Path(config['root'])
    run_root = root/'raw'/args.condition
    deps = root/'python_deps'
    if args.probe_only:
        with Run(run_root/'probe', stage='Kang_BAM_prefix_audit', kind='engineering',
                 config=row, inputs=[args.config]) as run:
            size, etag = remote_info(row['url'])
            if size != row['bytes']: raise ValueError('ENA byte count changed')
            path = run.directory/'prefix.partial.bam'
            with path.open('xb') as f:
                f.write(get_range(row['url'], 0, (8 << 20)-1, etag))
            result = audit(path, run.directory/'audit.json', deps, partial=True)
            (run.directory/'RESULTS.md').write_text('# Kang BAM prefix audit\n\n'+json.dumps(result, indent=2))
            print(json.dumps({k:v for k,v in result.items() if k != 'header'}), flush=True)
    else:
        destination = run_root/row['filename']
        download_ranges(row['url'], destination, run_root/('download' if args.mirror == 'ena' else 'download_ncbi'), workers=4)
        with Run(run_root/'validation', stage='Kang_BAM_validation', kind='engineering',
                 config=row, inputs=[destination, args.config]) as run:
            sys.path.insert(0, str(deps))
            import pysam
            pysam.quickcheck(str(destination))
            md5 = hashlib.md5()
            with destination.open('rb') as f:
                for block in iter(lambda: f.read(8 << 20), b''): md5.update(block)
            if md5.hexdigest() != row['md5']: raise ValueError('ENA submitted BAM MD5 mismatch')
            result = audit(destination, run.directory/'audit.json', deps)
            result.update(md5=md5.hexdigest(), integrity='ENA_size_MD5_and_BAM_quickcheck')
            save_json(run.directory/'summary.json', result)
            (run.directory/'RESULTS.md').write_text('# Kang BAM validated\n\n'+json.dumps(result, indent=2))
            print(json.dumps({k:v for k,v in result.items() if k != 'header'}), flush=True)


if __name__ == '__main__':
    main()
