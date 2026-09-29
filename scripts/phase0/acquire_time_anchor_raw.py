"""Acquire one manifest-fixed RENGE day on the dedicated data disk, CPU only.

No expression/guide labels are read, no quantification or model fitting is done.
Every run is downloaded, SRA-validated, extracted and compressed sequentially.
Existing files and interrupted attempts are never overwritten or removed.
"""
import argparse
import gzip
from pathlib import Path
import shutil

from veloroute.artifacts import Run, read_csv, save_csv, save_json
from veloroute.download import download_ranges
from veloroute.raw import audit_fastq_pair, plan_raw, run_command

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/data/yuchang/veloroute_timeanchors_20260914')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', required=True, type=int, choices=[2, 3])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not DATA_ROOT.is_dir() or shutil.disk_usage(DATA_ROOT).free < 250 * (1 << 30):
        raise ValueError('Dedicated data directory and at least 250 GiB free are required')
    tasks = plan_raw(read_csv(args.manifest), args.day,
                     ROOT/'tools/sratoolkit.3.4.1-ubuntu64/bin', DATA_ROOT/'raw')
    config = dict(day=args.day, cpu_only=True, gpu_count=0, tasks=tasks,
                  confirmation='SEALED', raw_acquisition_only=True,
                  no_deletion=True, one_run_at_a_time=True)
    with Run(args.output, stage='time_anchor_raw_acquisition', kind='engineering',
             config=config, inputs=[args.manifest]) as run:
        save_json(run.directory/'tasks.json', tasks)
        rows = []
        for task in tasks:
            accession = task['run']
            save_json(run.directory/'status.json', dict(status='running', run=accession,
                                                       completed_runs=len(rows)))
            record = DATA_ROOT/'records'/f'day{args.day}'/accession
            sra = DATA_ROOT/'raw'/'sra'/accession/f'{accession}.sra'
            url = f'https://sra-pub-run-odp.s3.amazonaws.com/sra/{accession}/{accession}'
            download_ranges(url, sra, record/'download', workers=4,
                            validator=ROOT/'tools/sratoolkit.3.4.1-ubuntu64/bin/vdb-validate')
            fastq = DATA_ROOT/'raw'/'fastq'/accession
            scratch = DATA_ROOT/'raw'/'scratch'/accession
            fastq.mkdir(parents=True, exist_ok=False)
            scratch.mkdir(parents=True, exist_ok=False)
            run_command(task['extract'], record/'extract', timeout_seconds=6*3600)
            reads = [fastq/f'{accession}_{i}.fastq' for i in (1, 2, 3)]
            if not all(p.is_file() for p in reads):
                raise ValueError('Expected index, barcode/UMI and library FASTQ files')
            barcode_path = Path('/data/dataset/perturbation_v1/renge_2023')/f'day{args.day}'/'barcodes.tsv.gz'
            with gzip.open(barcode_path, 'rt') as stream:
                barcodes = [line.strip() for line in stream]
            layout = audit_fastq_pair(reads[1], reads[2], whitelist=barcodes, max_records=1000000)
            with Run(record/'layout', stage='raw_layout_audit', kind='engineering',
                     config=dict(max_records=1000000), inputs=[reads[1], reads[2], barcode_path]) as audit:
                save_json(audit.directory/'layout.json', layout)
                (audit.directory/'RESULTS.md').write_text('# FASTQ layout\n\nBounded prefix only; chemistry not approved.\n')
            compressor = shutil.which('pigz') or shutil.which('gzip')
            if not compressor:
                raise ValueError('A gzip-compatible compressor is required')
            for index, source in enumerate(reads, start=1):
                options = ['-p', '4'] if Path(compressor).name == 'pigz' else []
                run_command([compressor, *options, '-k', str(source)],
                            record/f'compress_{index}', timeout_seconds=6*3600)
                run_command([compressor, '-t', str(source)+'.gz'],
                            record/f'gzip_test_{index}', timeout_seconds=2*3600)
            rows.append(dict(day=args.day, run=accession, library_type=task['library_type'],
                             status='SRA_validated_extracted_compressed', records_examined=layout['records_examined'],
                             raw_record=str(record), su_ready=False))
            save_json(run.directory/'completed_runs.json', rows)
        save_csv(run.directory/'metrics.csv', rows)
        save_json(run.directory/'summary.json', dict(day=args.day, raw_runs_complete=len(rows),
                                                   su_ready=False, confirmation='SEALED'))
        save_json(run.directory/'status.json', dict(status='complete', completed_runs=len(rows)))
        (run.directory/'RESULTS.md').write_text(
            '# Time-anchor raw acquisition\n\nFour separate libraries acquired; no S/U quantification or scientific GO. '
            'Original SRA and uncompressed FASTQs retained. GPU usage: zero.\n')


if __name__ == '__main__':
    main()
