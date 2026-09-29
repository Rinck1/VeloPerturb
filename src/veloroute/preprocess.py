"""Manifest-checked SRA -> USA -> annotated counts. No model fitting or G1 approval.

One sample's sequencing runs enter ONE mapping/UMI-resolution invocation. Summing
run-level matrices would double-count molecules sequenced on both runs.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io, sparse
from threadpoolctl import threadpool_limits

from .artifacts import Run, load_config, object_hash, read_csv, save_csv, save_json, sha256, utc_now
from .raw import audit_fastq_pair, run_command


def lines(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        return [line.strip() for line in stream if line.strip()]


def load_usa(directory):
    """Read alevin-fry's cell x [S|U|A] matrix, rejecting ambiguous layouts."""
    root = Path(directory)
    meta_path = next((p for p in (root/'quant.json', root/'meta_info.json') if p.is_file()), None)
    if meta_path is None:
        raise ValueError('Missing quant.json/meta_info.json')
    meta = json.loads(meta_path.read_text())
    if meta.get('usa_mode') is not True:
        raise ValueError('Quantification is not USA mode; cannot invent S/U')
    matrix_root = root/'alevin' if (root/'alevin').is_dir() else root
    rows = lines(matrix_root/'quants_mat_rows.txt')
    cols = lines(matrix_root/'quants_mat_cols.txt')
    with threadpool_limits(limits=2):
        matrix = sparse.csr_matrix(io.mmread(matrix_root/'quants_mat.mtx'))
    if matrix.shape[0] != len(rows) or matrix.shape[1] % 3:
        raise ValueError('USA matrix orientation/dimensions disagree with barcode count')
    ng = matrix.shape[1] // 3
    if len(cols) == ng:
        genes = cols
    elif len(cols) == 3*ng and cols[:ng] == cols[ng:2*ng] == cols[2*ng:]:
        genes = cols[:ng]
    elif (len(cols) == 3*ng and cols[ng:2*ng] == [g+'-U' for g in cols[:ng]]
          and cols[2*ng:] == [g+'-A' for g in cols[:ng]]):
        # Verified alevin-fry 0.18.2 schema: S IDs, matching -U IDs, matching -A IDs.
        genes = cols[:ng]
    else:
        raise ValueError('Unsupported USA gene label layout; inspect before importing')
    if int(meta.get('num_genes', -1)) not in (ng, 3*ng):
        raise ValueError('USA gene count disagrees with quant metadata')
    if len(set(rows)) != len(rows) or len(set(genes)) != len(genes):
        raise ValueError('Duplicated raw gene/barcode IDs')
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError('Nonfinite/negative UMI counts')
    return rows, genes, [matrix[:, i*ng:(i+1)*ng].copy() for i in range(3)], meta


def collapse_gene_versions(genes, matrices):
    # Preserve _PAR_Y: it is not the same genomic locus as the X copy.
    normalized = [re.sub(r'\.[0-9]+(?=_PAR_Y$|$)', '', gene) for gene in genes]
    unique = list(dict.fromkeys(normalized))
    lookup = {gene: i for i, gene in enumerate(unique)}
    mapping = sparse.csr_matrix((np.ones(len(genes)),
                                (np.arange(len(genes)), [lookup[g] for g in normalized])),
                               shape=(len(genes), len(unique)))
    return unique, [(m @ mapping).tocsr() for m in matrices]


def call_guides(counts, guide_ids, manifest, config):
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(guide_ids) or values.shape[1] < 2:
        raise ValueError('Guide matrix shape mismatch')
    if not np.isfinite(values).all() or np.any(values < 0) or not np.allclose(values, np.rint(values)):
        raise ValueError('Guide caller requires raw nonnegative integer UMI counts')
    by_id = {row['feature_id']: row for row in manifest}
    if len(set(guide_ids)) != len(guide_ids) or set(guide_ids) != set(by_id):
        raise ValueError('Guide IDs do not match the published 50-guide manifest')
    order = np.argsort(values, axis=1, kind='stable')
    top_index = order[:, -1]
    top = values[np.arange(len(values)), top_index]
    runner = values[np.arange(len(values)), order[:, -2]]
    fraction = top / np.maximum(values.sum(1), 1)
    ratio = top / np.maximum(runner, 1)
    called = ((top >= config['minimum_top_umi']) & (fraction >= config['minimum_top_fraction'])
              & (ratio >= config['minimum_top_runner_ratio']) & (top > runner))
    records = []
    for i, index in enumerate(top_index):
        guide = by_id[guide_ids[index]]
        records.append({'guide_top_feature': guide_ids[index], 'guide_top_umi': float(top[i]),
                        'guide_runner_umi': float(runner[i]), 'guide_top_fraction': float(fraction[i]),
                        'guide_top_runner_ratio': float(ratio[i]), 'guide_call_pass': bool(called[i]),
                        'guide_name': guide['guide_name'] if called[i] else '',
                        'condition': guide['target_from_published_name'] if called[i] else 'unassigned',
                        'control_class': guide['control_class'] if called[i] else 'unassigned',
                        'guide_labels_validated': bool(config['formal_validation_approved'] and called[i])})
    return pd.DataFrame(records)


def import_sample(quant_dir, day, config, output):
    import anndata as ad
    published = Path(config['published_root'])/f'day{day}'
    inputs = [Path(quant_dir)/'quant.json', config['guide_manifest'], config['condition_reservation'],
              published/'matrix.mtx.gz', published/'features.tsv.gz', published/'barcodes.tsv.gz']
    # Old alevin-fry uses meta_info.json.
    if not inputs[0].exists():
        inputs[0] = Path(quant_dir)/'meta_info.json'
    matrix_root = Path(quant_dir)/'alevin' if (Path(quant_dir)/'alevin').is_dir() else Path(quant_dir)
    inputs += [matrix_root/name for name in ('quants_mat.mtx', 'quants_mat_cols.txt', 'quants_mat_rows.txt')]
    with Run(output, stage='usa_import', kind='engineering', config=config, inputs=inputs) as run:
        barcodes, raw_genes, layers, meta = load_usa(quant_dir)
        genes, (s, u, a) = collapse_gene_versions(raw_genes, layers)
        pub_barcodes = lines(published/'barcodes.tsv.gz')
        pub_lookup = {b.split('-')[0]: i for i, b in enumerate(pub_barcodes)}
        canonical = [b.split('-')[0] for b in barcodes]
        if len(pub_lookup) != len(pub_barcodes) or len(set(canonical)) != len(canonical):
            raise ValueError('Barcode collision after suffix removal')
        if not set(canonical) <= set(pub_lookup):
            raise ValueError('Quantified barcodes outside the explicit published cohort')
        features = [row.split('\t') for row in lines(published/'features.tsv.gz')]
        guide_ix = [i for i, f in enumerate(features) if f[2] == 'CRISPR Guide Capture']
        with threadpool_limits(limits=2):
            counts = sparse.csr_matrix(io.mmread(published/'matrix.mtx.gz'))
        if counts.shape != (len(features), len(pub_barcodes)):
            raise ValueError('Published feature matrix shape mismatch')
        guide_counts = counts[guide_ix][:, [pub_lookup[b] for b in canonical]].T.toarray()
        obs = call_guides(guide_counts, [features[i][0] for i in guide_ix],
                          read_csv(config['guide_manifest']), config['guide_call'])
        roles = {r['condition']: r['role'] for r in read_csv(config['condition_reservation'])}
        obs['role'] = [roles.get(c, 'unassigned') for c in obs.condition]
        obs['barcode'] = canonical
        obs['day'] = int(day)
        obs['sample'] = f'GSE213069_day{day}'
        obs.index = pd.Index([f'day{day}:{b}' for b in canonical], name='cell_id')
        expression = s + a
        var_names = {re.sub(r'\.[0-9]+$', '', f[0]): f[1] for f in features if f[2] == 'Gene Expression'}
        var = pd.DataFrame({'gene_symbol': [var_names.get(g, g) for g in genes]}, index=genes)
        total = np.asarray(expression.sum(1)).ravel()
        mito = np.asarray(expression[:, var.gene_symbol.str.startswith('MT-').values].sum(1)).ravel()
        obs['expression_umi'] = total
        obs['detected_genes'] = expression.getnnz(1)
        obs['spliced_umi'] = np.asarray(s.sum(1)).ravel()
        obs['unspliced_umi'] = np.asarray(u.sum(1)).ravel()
        obs['ambiguous_umi'] = np.asarray(a.sum(1)).ravel()
        obs['mito_fraction'] = mito / np.maximum(total, 1)
        qc = config['qc']
        obs['technical_qc_pass'] = ((obs.expression_umi >= qc['minimum_expression_umi'])
                                    & (obs.detected_genes >= qc['minimum_detected_genes'])
                                    & (obs.unspliced_umi >= qc['minimum_unspliced_umi'])
                                    & (obs.mito_fraction <= qc['maximum_mito_fraction']))
        result = ad.AnnData(expression, obs=obs, var=var, layers={'spliced': s, 'unspliced': u, 'ambiguous': a})
        result.uns['veloroute'] = {'stage': 'raw_counts_only', 'day': day,
                                  'expression_policy': 'S+A', 'velocity_spliced_policy': 'strict_S',
                                  'ambiguous_policy': 'retained_separate_never_assigned_to_U',
                                  'guide_call_config_json': json.dumps(config['guide_call'], sort_keys=True),
                                  'qc_config_json': json.dumps(config['qc'], sort_keys=True),
                                  'source_quantification': str(Path(quant_dir).resolve()),
                                  'formal_ready': False}
        result.obsm['guide_counts'] = guide_counts
        result.uns['guide_feature_ids'] = [features[i][0] for i in guide_ix]
        result.write_h5ad(run.directory/f'day{day}.h5ad', compression='gzip')
        obs.to_csv(run.directory/'cell_manifest.csv')
        metrics = {'cells': result.n_obs, 'genes': result.n_vars,
                   'cells_passing_technical_qc': int(obs.technical_qc_pass.sum()),
                   'cells_with_conservative_guide_call': int(obs.guide_call_pass.sum()),
                   'median_spliced_umi': float(np.median(obs.spliced_umi)),
                   'median_unspliced_umi': float(np.median(obs.unspliced_umi)),
                   'median_ambiguous_umi': float(np.median(obs.ambiguous_umi)),
                   'u_positive_genes': int(np.count_nonzero(np.asarray(u.sum(0)))),
                   'formal_ready': False, 'G1': 'NOT_RUN'}
        save_json(run.directory/'summary.json', metrics)
        save_csv(run.directory/'metrics.csv', [{'metric': k, 'value': v} for k, v in metrics.items()])
        (run.directory/'RESULTS.md').write_text('# RENGE USA counts\n\n' + json.dumps(metrics, indent=2)
            + '\n\nS, U and A are retained separately. QC/guide calls are computational screening only; '
              'no expression outcome, velocity fit, or test-condition effect was evaluated. Formal approval remains false.\n')
    return metrics


def check_layout(audit, config):
    if audit['r1_length_histogram'] != {26: audit['records_examined']} and audit['r1_length_histogram'] != {'26': audit['records_examined']}:
        raise ValueError('Unexpected barcode/UMI read length')
    if {str(k): v for k, v in audit['r2_length_histogram'].items()} != {str(config['read_length']): audit['records_examined']}:
        raise ValueError('Unexpected cDNA length; rebuild geometry/reference specification')
    hits = audit['published_barcode_match_fraction_by_zero_based_offset']
    if hits['0'] < config['minimum_barcode_match'] or hits['0'] < config['minimum_barcode_offset_ratio']*hits['1']:
        raise ValueError('Barcode offset not supported; do not guess or trim a leading N')


def run_preprocessing(config_path, output, *, reuse_run=None, days=None):
    config = load_config(config_path)
    execution_days = list(days or config['days'])
    if len(set(execution_days)) != len(execution_days) or not set(execution_days) <= set(config['days']):
        raise ValueError('Execution days must be a nonduplicated subset of configured days')
    previous = Path(reuse_run).resolve() if reuse_run else None
    if previous:
        old_config = load_config(previous/'config.yaml')
        if old_config != config:
            raise ValueError('Resume requires identical preprocessing configuration')
    shared_records = []
    stop = threading.Event()
    root = Path(output).resolve()
    raw, sra_bin, quant_bin = [Path(config[k]).resolve() for k in ('raw_root', 'sra_bin', 'quant_bin')]
    env = {'NCBI_SETTINGS': str(Path('tools/ncbi-config/user-settings.mkfg').resolve()),
           'ALEVIN_FRY_HOME': str(Path(config['af_home']).resolve())}
    deadline = time.monotonic() + config['timeout_seconds']
    run_rows = read_csv(config['run_manifest'])
    selected = [r for r in run_rows if int(r['day']) in execution_days]
    for day in execution_days:
        rows = [r for r in selected if int(r['day']) == day]
        if len(rows) != 4 or sum(r['library_type'] == 'mRNA' for r in rows) != 2:
            raise ValueError('Each day requires the two GEX and two gRNA runs in the accession manifest')

    def state(worker, message):
        save_json(root/f'{worker}_state.json', {'status': message, 'updated_utc': utc_now(), 'pid': os.getpid()})

    def await_ready(predicate, worker, message):
        state(worker, message)
        while not predicate():
            if stop.is_set():
                raise RuntimeError('A parallel preprocessing stage failed; stopping dependent work')
            if time.monotonic() >= deadline:
                raise TimeoutError(message)
            time.sleep(5)

    def command(name, argv, inputs=(), timeout=7200):
        return run_command([str(v) for v in argv], root/name, inputs=inputs, environment=env,
                           timeout_seconds=min(timeout, max(1, deadline-time.monotonic())))

    def reference_worker():
        for job in config['reference_download_jobs']:
            def ready(job=job):
                record = json.loads(Path(job).read_text()) if Path(job).is_file() else {}
                if record.get('status') in {'failed', 'interrupted_or_timeout'}:
                    raise RuntimeError(f'Reference download failed: {job}')
                return record.get('status') == 'complete'
            await_ready(ready, 'reference', f'waiting_for_{job}')
        ref = Path(config['reference_root']).resolve()
        expected = {line.split()[-1].lstrip('*'): line.split()[0] for line in lines(ref/'MD5SUMS')}
        checks = []
        for name in (config['reference_fasta'], config['reference_gtf']):
            digest = hashlib.md5()
            with (ref/name).open('rb') as stream:
                for block in iter(lambda: stream.read(8 << 20), b''):
                    digest.update(block)
            if digest.hexdigest() != expected.get(name):
                raise ValueError(f'Reference MD5 mismatch: {name}')
            checks.append({'file': name, 'md5': digest.hexdigest(), 'sha256': sha256(ref/name)})
        save_json(root/'reference_checksums.json', checks)
        index = Path(config['index_output']).resolve()
        if index.exists():
            if not previous:
                raise FileExistsError('Reference index exists; provide audited --reuse-run')
            old_state = json.loads((previous/'reference_state.json').read_text())
            old_build = json.loads((previous/'build_splici/status.json').read_text())
            if old_state['status'] != 'complete' or old_build['status'] != 'complete':
                raise ValueError('Cannot reuse incomplete reference build')
            if json.loads((previous/'reference_checksums.json').read_text()) != checks:
                raise ValueError('Reference files changed since successful build')
            save_json(root/'index_files.json', {str(p.relative_to(index)): sha256(p) for p in sorted((index/'index').glob('*')) if p.is_file()})
            save_json(root/'reference_reuse.json', {'source_run': str(previous), 'reference_checksums': checks,
                      'original_build_provenance_sha256': sha256(previous/'build_splici/provenance.json')})
            state('reference', 'complete_reused_audited_index')
            return index
        state('reference', 'building_splici_r91')
        command('build_splici', [quant_bin/'simpleaf', 'index', '--fasta', ref/config['reference_fasta'],
                                '--gtf', ref/config['reference_gtf'], '--rlen', config['read_length'],
                                '--ref-type', 'spliced+intronic', '--threads', config['threads'],
                                '--output', index, '--work-dir', index/'work', '--ram-limit-gib', '8'],
                inputs=[ref/config['reference_fasta'], ref/config['reference_gtf'], config_path])
        state('reference', 'complete')
        return index

    def reads_worker():
        records = shared_records
        old_records = {r['run_accession']: r for r in json.loads((previous/'completed_raw_runs.json').read_text())} if previous else {}
        for row in sorted(selected, key=lambda r: (int(r['day']), r['library_type'] != 'mRNA', -int(r['run_accession'][3:]))):
            acc, day = row['run_accession'], int(row['day'])
            sra = raw/'sra'/acc/f'{acc}.sra'
            if acc in old_records:
                old = old_records[acc]
                for path, expected in [(sra, old['sra_sha256']), (old['barcode_fastq'], old['barcode_sha256']),
                                        (old['cdna_fastq'], old['cdna_sha256'])]:
                    if sha256(path) != expected:
                        raise ValueError(f'Completed raw artifact changed: {path}')
                records.append(old)
                save_json(root/'completed_raw_runs.json', records)
                state('reads', f'{acc}_reused_hash_verified')
                continue
            await_ready(lambda: sra.is_file() and not Path(str(sra)+'.lock').exists(), 'reads', f'waiting_for_{acc}_SRA')
            command(f'validate_{acc}', [sra_bin/'vdb-validate', sra])
            fastq_root = raw/'fastq'/acc
            gz_paths = [fastq_root/f'{acc}_{i}.fastq.gz' for i in (1, 2, 3)]
            raw_paths = [fastq_root/f'{acc}_{i}.fastq' for i in (1, 2, 3)]
            # The first GEX run was started before this orchestrator. Wait for its
            # recorded compressor, never consume a partially written .gz file.
            if acc == 'SRR21518308':
                compress_status = Path('outputs/veloroute_raw_day4/compress_SRR21518308/status.json')
                def compression_done():
                    status = json.loads(compress_status.read_text())
                    if status['status'] in {'failed', 'interrupted_or_timeout'}:
                        raise RuntimeError('Existing GEX compression failed')
                    return status['status'] == 'complete'
                await_ready(compression_done, 'reads', 'waiting_for_existing_SRR21518308_compression')
            if not all(p.is_file() for p in gz_paths):
                if not all(p.is_file() for p in raw_paths):
                    if any(p.exists() for p in raw_paths + gz_paths):
                        raise ValueError(f'Partial FASTQs for {acc}; preserve and audit, do not overwrite')
                    free = shutil.disk_usage(raw).free / (1 << 30)
                    if free < config['minimum_free_gib_before_extract']:
                        raise RuntimeError(f'Only {free:.1f} GiB free before extraction; stop safely')
                    command(f'extract_{acc}', [sra_bin/'fasterq-dump', sra, '--split-files', '--include-technical',
                            '--threads', config['extract_threads'], '--outdir', fastq_root, '--temp', raw/'scratch'/acc])
                command(f'compress_{acc}', ['/usr/bin/pigz', '-p', config['threads'], '-6', *raw_paths])
            command(f'gzip_test_{acc}', ['/usr/bin/pigz', '-t', '-p', '4', *gz_paths])
            bc = Path(config['published_root'])/f'day{day}'/'barcodes.tsv.gz'
            r1, r2 = gz_paths[config['barcode_read_index']-1], gz_paths[config['cdna_read_index']-1]
            with Run(root/f'layout_{acc}', stage='fastq_layout', kind='engineering', config=config,
                     inputs=[r1, r2, bc]) as run:
                audit = audit_fastq_pair(r1, r2, whitelist=lines(bc), max_records=config['layout_audit_records'])
                save_json(run.directory/'layout.json', audit)
                check_layout(audit, config)
                save_json(run.directory/'geometry_decision.json', {
                    'run': acc, 'geometry': config['geometry'], 'expected_orientation': config['expected_orientation'],
                    'basis': 'observed_26bp_91bp_offset_0_and_documented_5prime_R2_only_protocol',
                    'scope': 'technical_geometry_only_not_velocity_quality', 'full_pair_scan': False})
            records.append({**row, 'barcode_fastq': str(r1), 'cdna_fastq': str(r2),
                            'sra_sha256': sha256(sra), 'barcode_sha256': sha256(r1), 'cdna_sha256': sha256(r2)})
            save_json(root/'completed_raw_runs.json', records)
            state('reads', f'{acc}_complete')
        state('reads', 'complete')
        return records

    inputs = [config_path, config['run_manifest'], config['guide_manifest'], config['condition_reservation']]
    if previous:
        inputs += [previous/'completed_raw_runs.json', previous/'reference_checksums.json']
    with Run(root, stage='renge_preprocessing', kind='engineering', config=config, inputs=inputs) as run:
        try:
            def guarded(worker):
                try:
                    return worker()
                except BaseException:
                    stop.set()
                    raise
            with ThreadPoolExecutor(max_workers=2) as pool:
                reference_future = pool.submit(guarded, reference_worker)
                reads_future = pool.submit(guarded, reads_worker)
                try:
                    index = reference_future.result()
                    for day in execution_days:
                        def day_ready(day=day):
                            if reads_future.done():
                                reads_future.result()
                            return sum(int(r['day']) == day for r in shared_records) == 4
                        await_ready(day_ready, 'quantification', f'waiting_for_day{day}_audited_runs')
                        day_records = [r for r in shared_records if int(r['day']) == day]
                        day_manifest = root/f'day{day}_run_manifest.csv'
                        save_csv(day_manifest, day_records)
                        selected_gex = sorted([r for r in day_records if r['library_type'] == 'mRNA'], key=lambda r: r['run_accession'])
                        whitelist = root/f'day{day}_barcodes.txt'
                        with whitelist.open('x') as stream:
                            stream.write('\n'.join(b.split('-')[0] for b in lines(Path(config['published_root'])/f'day{day}'/'barcodes.tsv.gz'))+'\n')
                        out = Path(config['quant_output']).resolve()/f'day{day}'
                        if out.exists():
                            raise FileExistsError(f'Quant output exists: {out}; inspect or import completed output explicitly')
                        state('quantification', f'day{day}_mapping_and_USA_joint_two_runs')
                        command(f'quant_day{day}', [quant_bin/'simpleaf', 'quant', '--index', index/'index',
                                '--chemistry', config['geometry'], '--expected-ori', config['expected_orientation'],
                                '--reads1', ','.join(r['barcode_fastq'] for r in selected_gex),
                                '--reads2', ','.join(r['cdna_fastq'] for r in selected_gex),
                                '--explicit-pl', whitelist, '--resolution', 'cr-like', '--threads', config['threads'],
                                '--output', out], inputs=[whitelist, day_manifest, config_path])
                        import_sample(out/'af_quant', day, config, Path(config['processed_output'])/f'day{day}')
                        state('quantification', f'day{day}_counts_complete_formal_QC_pending')
                    reads = reads_future.result()
                except BaseException:
                    stop.set()
                    raise
            save_csv(root/'processed_run_manifest.csv', reads)
            save_json(root/'summary.json', {'status': 'COUNTS_COMPLETE', 'days': execution_days, 'formal_ready': False, 'G1': 'NOT_RUN'})
            (root/'RESULTS.md').write_text('# RENGE data pipeline\n\nCounts generated for configured days.\n\n'
                'Two sequencing runs per GEX library jointly UMI-resolved. S/U/A preserved. '
                'Formal data/guide/velocity approval and G1 remain separate.\n')
        except BaseException as error:
            state('pipeline', f'failed: {type(error).__name__}: {error}')
            (root/'RESULTS.md').write_text(f'# RENGE data pipeline\n\nStopped: {type(error).__name__}: {error}\n\n'
                                          'Completed inputs and failed-stage logs are preserved. No success is inferred.\n')
            raise
