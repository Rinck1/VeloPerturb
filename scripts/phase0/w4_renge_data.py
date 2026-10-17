"""W4 RENGE technical-only acquisition, joint USA quantification, and QC.

No condition expression summaries, embeddings, velocity fits, or effects are
computed. Day4 reproduction uses the SAME TWO GEX runs as the historical release.
New outputs are immutable and confined to the dedicated data disk.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

import numpy as np

from veloroute.artifacts import Run, load_config, read_csv, save_csv, save_json, sha256
from veloroute.download import download_ranges, remote_info
from veloroute.preprocess import audit_fastq_pair, check_layout, import_sample, lines
from veloroute.raw import run_command

ROOT = Path(__file__).resolve().parents[2]
ENA = ('https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA879051'
       '&result=read_run&fields=run_accession,experiment_accession,sample_accession,'
       'experiment_title,read_count,base_count&format=tsv')


def configuration(path):
    extra = load_config(path)
    base = load_config(extra['base_config'])
    for key in ('geometry', 'expected_orientation', 'read_length', 'barcode_read_index', 'cdna_read_index'):
        if extra[key] != base[key]:
            raise ValueError(f'W4 changed historical chemistry: {key}')
    base.update(extra)
    if not 1 <= base['threads'] <= 8 or not 1 <= base['extract_threads'] <= 8:
        raise ValueError('CPU budget exceeded')
    base['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    base['script_sha256'] = sha256(__file__)
    base['seed'] = None
    base['argv'] = sys.argv
    return base


def stamp(config, start, **kwargs):
    return dict(git_commit=config['git_commit'], seed=None, argv=sys.argv,
                elapsed_seconds=time.monotonic() - start,
                script_sha256=config['script_sha256'], **kwargs)


def rows_for_day(config, day):
    if day == 4:
        rows = [dict(r, raw_cache_root=str((ROOT/'data/renge/raw').resolve()))
                for r in read_csv(config['historical_run_manifest']) if int(r['day']) == day]
    else:
        rows = [r for r in read_csv(config['run_manifest']) if int(r['day']) == day]
    if len(rows) != 4 or sum(r['library_type'] == 'mRNA' for r in rows) != 2:
        raise ValueError('Exactly two GEX and two guide runs required')
    return rows


def audit(config, output):
    start = time.monotonic()
    rows = read_csv(config['run_manifest'])
    with Run(output, stage='w4_renge_technical_inventory', kind='engineering', config=config,
             inputs=[config['run_manifest'], config['base_config']]) as run:
        with urlopen(ENA, timeout=60) as stream:
            text = stream.read().decode()
        (run.directory/'ena_PRJNA879051.tsv').write_text(text)
        public = {r['run_accession']: r for r in csv.DictReader(io.StringIO(text), delimiter='\t')}
        inventory = []
        for row in rows:
            acc = row['run_accession']; remote = public[acc]
            for key in ('experiment_accession', 'sample_accession'):
                if row[key] != remote[key]:
                    raise ValueError(f'ENA metadata mismatch: {acc} {key}')
            for value in (row['geo_sample'], 'day'+row['day'], 'from '+row['library_type']):
                if value not in remote['experiment_title']:
                    raise ValueError(f'Unexpected day/library: {acc}')
            raw = Path(row['raw_cache_root'])
            sra = raw/'sra'/acc/f'{acc}.sra'
            size = sra.stat().st_size if sra.exists() else remote_info(row['source_url'])[0]
            if row['sra_bytes'] and size != int(row['sra_bytes']):
                raise ValueError(f'SRA byte count changed: {acc}')
            r1, r2 = [raw/'fastq'/acc/f'{acc}_{i}.fastq.gz' for i in (2, 3)]
            inventory.append(dict(**row, actual_sra_bytes=size, cached_sra=sra.is_file(),
                                  cached_barcode_fastq=r1.is_file(), cached_cdna_fastq=r2.is_file(),
                                  additional_download_bytes=0 if sra.is_file() else size))
        pending = sum(r['additional_download_bytes'] for r in inventory)
        if pending > config['download_limit_bytes']:
            raise ValueError('Additional dataset download exceeds the authorized 50 GB limit')
        save_csv(run.directory/'run_inventory.csv', inventory)
        result = stamp(config, start, status='TECHNICAL_INVENTORY_COMPLETE',
                       additional_download_bytes=pending, ena_url=ENA,
                       ena_sha256=sha256(run.directory/'ena_PRJNA879051.tsv'),
                       run_manifest_sha256=sha256(config['run_manifest']),
                       data_free_bytes=shutil.disk_usage('/data/yuchang').free,
                       home_free_bytes=shutil.disk_usage(ROOT).free,
                       replication_effects_read=False)
        save_json(run.directory/'summary.json', result)
        (run.directory/'RESULTS.md').write_text('# W4 RENGE raw inventory\n\n'+json.dumps(result, indent=2)+'\n')
    return result


def acquire(config, output, day):
    start = time.monotonic(); rows = rows_for_day(config, day)
    sizes = [remote_info(r['source_url'])[0] for r in rows]
    if sum(sizes) > config['download_limit_bytes']:
        raise ValueError('Dataset download budget exceeded')
    if shutil.disk_usage('/data/yuchang').free < config['minimum_free_gib_before_extract'] * (1 << 30):
        raise ValueError('Insufficient data-disk free space')
    with Run(output, stage='w4_renge_raw_acquisition', kind='engineering', config=config,
             inputs=[config['run_manifest']]) as run:
        completed = []
        for row, size in zip(rows, sizes):
            acc = row['run_accession']; raw = Path(row['raw_cache_root'])
            sra = raw/'sra'/acc/f'{acc}.sra'
            if sra.exists():
                if sra.stat().st_size != size:
                    raise ValueError(f'Partial existing SRA, preserved: {sra}')
                run_command([str((ROOT/config['sra_bin']/'vdb-validate').resolve()), str(sra)],
                            run.directory/f'validate_{acc}', timeout_seconds=600)
            else:
                download_ranges(row['source_url'], sra, run.directory/f'download_{acc}', workers=4,
                                validator=ROOT/config['sra_bin']/'vdb-validate')
            fastq = raw/'fastq'/acc; scratch = raw/'scratch'/acc
            gz = [fastq/f'{acc}_{i}.fastq.gz' for i in (1, 2, 3)]
            if not all(p.exists() for p in gz):
                fastq.mkdir(parents=True, exist_ok=False); scratch.mkdir(parents=True, exist_ok=False)
                run_command([str((ROOT/config['sra_bin']/'fasterq-dump').resolve()), str(sra),
                             '--split-files', '--include-technical', '--threads', str(config['extract_threads']),
                             '--outdir', str(fastq), '--temp', str(scratch)], run.directory/f'extract_{acc}',
                            timeout_seconds=config['timeout_seconds'])
                for i in (1, 2, 3):
                    run_command(['/usr/bin/pigz', '-p', str(config['threads']), '-k', str(fastq/f'{acc}_{i}.fastq')],
                                run.directory/f'compress_{acc}_{i}', timeout_seconds=config['timeout_seconds'])
            run_command(['/usr/bin/pigz', '-t', '-p', str(config['threads']), *map(str, gz)],
                        run.directory/f'gzip_test_{acc}', timeout_seconds=config['timeout_seconds'])
            completed.append(dict(run=acc, sra_sha256=sha256(sra), size_bytes=size,
                                  fastqs=[dict(path=str(p), sha256=sha256(p), bytes=p.stat().st_size) for p in gz]))
            save_json(run.directory/'progress.json', stamp(config, start, completed_runs=completed))
        result = stamp(config, start, status='RAW_ACQUISITION_COMPLETE', day=day, completed_runs=completed,
                       replication_effects_read=False)
        save_json(run.directory/'summary.json', result)
        (run.directory/'RESULTS.md').write_text('# W4 RENGE acquisition\n\n'+json.dumps(result, indent=2)+'\n')
    return result


def technical_qc(path):
    import anndata as ad
    a = ad.read_h5ad(path, backed='r')
    obs = a.obs.copy()
    result = dict(cells=a.n_obs, genes=a.n_vars, assigned_cells=int(obs.guide_call_pass.sum()),
                  assigned_fraction=float(obs.guide_call_pass.mean()),
                  technical_pass_cells=int(obs.technical_qc_pass.sum()),
                  jointly_pass_cells=int((obs.guide_call_pass & obs.technical_qc_pass).sum()),
                  median_spliced_umi=float(obs.spliced_umi.median()),
                  median_unspliced_umi=float(obs.unspliced_umi.median()),
                  median_U_fraction=float((obs.unspliced_umi/(obs.spliced_umi+obs.unspliced_umi).clip(lower=1)).median()))
    counts = []
    for name, group in obs.groupby('condition', observed=True):
        counts.append(dict(condition=str(name), cells=len(group),
                           technical_pass_cells=int(group.technical_qc_pass.sum()),
                           jointly_pass_cells=int((group.guide_call_pass & group.technical_qc_pass).sum())))
    a.file.close()
    return result, counts


def reproduce(reference, current, minimum):
    import anndata as ad
    old, new = ad.read_h5ad(reference), ad.read_h5ad(current)
    if set(old.obs_names) != set(new.obs_names) or set(old.var_names) != set(new.var_names):
        raise ValueError('Same-input day4 rerun changed barcode or gene universe')
    new = new[old.obs_names, old.var_names]
    result = {}
    for layer in ('spliced', 'unspliced', 'ambiguous'):
        a = np.asarray(old.layers[layer].sum(1)).ravel()
        b = np.asarray(new.layers[layer].sum(1)).ravel()
        correlation = float(np.corrcoef(a, b)[0, 1])
        result[layer] = dict(total_count_correlation=correlation,
                             maximum_total_count_difference=float(np.abs(a-b).max()),
                             matrix_disagreement_entries=int((old.layers[layer]-new.layers[layer]).nnz))
        if correlation < minimum:
            raise ValueError(f'Day4 same-input reproduction failed: {layer} {correlation}')
    return result


def process(config, output, day, prerequisite):
    start = time.monotonic(); rows = rows_for_day(config, day)
    if day != 4:
        if not prerequisite:
            raise ValueError('New days require an independently completed day4 reproduction')
        check = json.loads(Path(prerequisite).read_text())
        if check.get('status') != 'SAME_INPUT_DAY4_REPRODUCTION_PASSED':
            raise ValueError('Day4 reproduction has not passed')
    with Run(output, stage='w4_renge_joint_USA_technical_only', kind='engineering', config=config,
             inputs=[config['base_config'], config['run_manifest'], config['historical_index_hashes']]) as run:
        index = (ROOT/config['index_output']).resolve()
        for relative, expected in json.loads(Path(config['historical_index_hashes']).read_text()).items():
            if sha256(index/relative) != expected:
                raise ValueError(f'Historical index changed: {relative}')
        save_json(run.directory/'index_verified.json', stamp(config, start, verified=True,
                  index_hash_manifest_sha256=sha256(config['historical_index_hashes'])))
        records = []
        old = {r['run_accession']: r for r in json.loads(Path(config['historical_day4_raw_hashes']).read_text())}
        for row in sorted((r for r in rows if r['library_type'] == 'mRNA'), key=lambda r:r['run_accession']):
            acc = row['run_accession']; raw = Path(row['raw_cache_root'])
            r1, r2 = [raw/'fastq'/acc/f'{acc}_{i}.fastq.gz' for i in (2, 3)]
            checksums = [sha256(p) for p in (r1, r2)]
            if day == 4 and checksums != [old[acc]['barcode_sha256'], old[acc]['cdna_sha256']]:
                raise ValueError(f'Day4 input FASTQs changed: {acc}')
            barcode_file = Path(config['published_root'])/f'day{day}'/'barcodes.tsv.gz'
            layout = audit_fastq_pair(r1, r2, whitelist=lines(barcode_file), max_records=config['layout_audit_records'])
            check_layout(layout, config)
            save_json(run.directory/f'layout_{acc}.json', stamp(config, start, **layout,
                       barcode_sha256=checksums[0], cdna_sha256=checksums[1]))
            records.append(dict(run=acc, barcode_fastq=str(r1), cdna_fastq=str(r2),
                                barcode_sha256=checksums[0], cdna_sha256=checksums[1]))
        save_csv(run.directory/'input_fastqs.csv', records)
        whitelist = run.directory/'published_barcodes.txt'
        whitelist.write_text('\n'.join(b.split('-')[0] for b in lines(barcode_file))+'\n')
        quant = run.directory/'quant'
        cmd = [str((ROOT/config['quant_bin']/'simpleaf').resolve()), 'quant', '--index', str(index/'index'),
               '--chemistry', config['geometry'], '--expected-ori', config['expected_orientation'],
               '--reads1', ','.join(r['barcode_fastq'] for r in records),
               '--reads2', ','.join(r['cdna_fastq'] for r in records), '--explicit-pl', str(whitelist),
               '--resolution', 'cr-like', '--threads', str(config['threads']), '--output', str(quant)]
        run_command(cmd, run.directory/'quant_command', timeout_seconds=config['timeout_seconds'],
                    inputs=[whitelist, run.directory/'input_fastqs.csv'],
                    environment={'ALEVIN_FRY_HOME':str((ROOT/config['af_home']).resolve()), 'CUDA_VISIBLE_DEVICES':''})
        imported = run.directory/'import' if day == 4 else Path(config['processed_output'])/f'day{day}'
        import_sample(quant/'af_quant', day, config, imported)
        h5ad = imported/f'day{day}.h5ad'
        qc, counts = technical_qc(h5ad)
        save_csv(run.directory/'condition_cell_counts.csv', counts)
        comparison = reproduce(ROOT/'data/renge/processed_release_v1/day4/day4.h5ad', h5ad,
                               config['day4_minimum_total_count_correlation']) if day == 4 else None
        result = stamp(config, start, status='SAME_INPUT_DAY4_REPRODUCTION_PASSED' if day == 4 else 'COUNTS_TECHNICAL_QC_COMPLETE',
                       day=day, h5ad=str(h5ad), h5ad_sha256=sha256(h5ad), qc=qc,
                       same_input_reproduction=comparison, replication_effects_read=False,
                       guide_semantics='published_feature_UMI_dominance_computational_labels',
                       formal_ready=False)
        save_json(run.directory/'summary.json', result)
        (run.directory/'RESULTS.md').write_text('# W4 RENGE technical QC\n\n'+json.dumps(result, indent=2)+'\n')
    return result


def day23_pipeline(config, output, prerequisite):
    """Run serially so download/extraction/quantification stay within eight CPUs."""
    start = time.monotonic()
    if not prerequisite:
        raise ValueError('Need a day4 same-input reproduction summary path')
    path = Path(prerequisite)
    with Run(output, stage='w4_renge_day23_serial_pipeline', kind='engineering', config=config,
             inputs=[config['run_manifest'], config['base_config']]) as run:
        while not path.exists():
            provenance_path = path.parent/'provenance.json'
            if provenance_path.exists() and json.loads(provenance_path.read_text()).get('status') == 'failed':
                raise RuntimeError('Day4 reproduction failed; new-day processing blocked')
            if time.monotonic() - start > config['timeout_seconds']:
                raise TimeoutError('Day4 reproduction never completed')
            save_json(run.directory/'status.json', stamp(config, start, status='waiting_for_day4_reproduction'))
            time.sleep(15)
        for name, operation in [
            ('day3_quantification', lambda: process(config, run.directory/'day3', 3, path)),
            ('day2_raw_acquisition', lambda: acquire(config, run.directory/'day2_raw', 2)),
            ('day2_quantification', lambda: process(config, run.directory/'day2', 2, path)),
        ]:
            save_json(run.directory/'status.json', stamp(config, start, status='running', stage=name))
            operation()
        result = stamp(config, start, status='DAY23_TECHNICAL_COUNTS_COMPLETE',
                       processed_root=config['processed_output'], replication_effects_read=False)
        save_json(run.directory/'summary.json', result)
        save_json(run.directory/'status.json', result)
        (run.directory/'RESULTS.md').write_text('# W4 RENGE day2/day3 pipeline\n\n'+json.dumps(result, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/w4_renge_day23_preprocess.yaml')
    p.add_argument('--mode', choices=['audit', 'acquire', 'process', 'day23'], required=True)
    p.add_argument('--day', type=int, choices=[2,3,4])
    p.add_argument('--output', required=True)
    p.add_argument('--reproduction-summary')
    args = p.parse_args(); os.chdir(ROOT)
    config = configuration(args.config)
    if args.mode == 'audit':
        result = audit(config, args.output)
    elif args.mode == 'acquire':
        if args.day not in (2,3):
            p.error('Acquisition requires day2 or day3')
        result = acquire(config, args.output, args.day)
    elif args.mode == 'process':
        if args.day is None:
            p.error('Processing requires a day')
        result = process(config, args.output, args.day, args.reproduction_summary)
    else:
        result = day23_pipeline(config, args.output, args.reproduction_summary)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
