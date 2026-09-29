"""Kang batch2: sample-keyed barcodes, original BAM tags, training-only folds."""
from __future__ import annotations

from collections import Counter
import gzip
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

from .artifacts import Run, object_hash, save_csv, save_json, sha256
from .latent import FrozenSplicingTransform, save_pack
from .preprocess import collapse_gene_versions, load_usa


def published_metadata(root):
    path = Path(root)/'GSE96583_batch2.total.tsne.df.tsv.gz'
    data = pd.read_csv(path, sep='\t', index_col=0, dtype={'ind': str})
    if list(data.columns) != ['tsne1','tsne2','ind','stim','cluster','cell','multiplets']:
        raise ValueError('Unexpected Kang batch2 metadata schema')
    data = data.rename(columns={'ind': 'donor', 'stim': 'condition', 'cell': 'cell_type'})
    data['barcode'] = data.index.astype(str)
    data['cell_id'] = data.condition.astype(str)+':'+data.barcode
    if data.cell_id.duplicated().any() or set(data.condition) != {'ctrl','stim'}:
        raise ValueError('Sample/barcode keys are not unique or conditions changed')
    return data.set_index('cell_id', drop=False)


def freeze_metadata(config, output):
    root = Path(config['published_root'])
    metadata = published_metadata(root)
    singlets = metadata[metadata.multiplets == 'singlet']
    if sorted(singlets.donor.unique()) != sorted(config['donors']):
        raise ValueError('Observed donor IDs differ from registered folds')
    with Run(output, stage='Kang_metadata_and_protocol_freeze', kind='engineering', config=config,
             inputs=[root/'GSE96583_batch2.total.tsne.df.tsv.gz'], seed=config['representation']['seed']) as run:
        rows = singlets.groupby(['donor','cell_type','condition']).size().rename('cells').reset_index()
        save_csv(run.directory/'donor_type_counts.csv', rows.to_dict('records'))
        folds = [dict(heldout=d, training=[t for t in config['donors'] if t != d]) for d in config['donors']]
        save_json(run.directory/'folds.json', folds)
        result = dict(status='FROZEN_BEFORE_EXPRESSION', donors=config['donors'], singlets=len(singlets),
                      expression_read=False, config_hash=object_hash(config), n_outer_folds=len(folds))
        save_json(run.directory/'summary.json', result)
        (run.directory/'RESULTS.md').write_text('# Kang protocol freeze\n\n'+json.dumps(result, indent=2))
    return result


def tagged_fastq(read, permit):
    """Corrected barcode if available; otherwise exact raw match. Always raw UMI.

    Only primary alignments represent molecules. Original sequencing orientation
    is restored (BAM reverse alignments store reverse-complemented sequence).
    Alignment/GX status is not a filter: intronic and unmapped reads are retained.
    """
    if read.is_secondary or read.is_supplementary:
        return None, 'nonprimary'
    if not read.has_tag('CR') or not read.has_tag('UR'):
        return None, 'missing_raw_tags'
    raw_bc, umi = read.get_tag('CR'), read.get_tag('UR')
    bc = read.get_tag('CB').split('-')[0] if read.has_tag('CB') else raw_bc
    if bc not in permit:
        return None, 'outside_published_cohort'
    if len(bc) != 14 or len(umi) != 10 or set(bc+umi)-set('ACGTN'):
        raise ValueError('Kang audited 14bp barcode / 10bp UMI layout changed')
    sequence = read.get_forward_sequence()
    quality = read.get_forward_qualities()
    if not sequence or quality is None or len(sequence) != len(quality):
        raise ValueError('Primary read lacks original cDNA/quality')
    name = read.query_name
    r1 = f'@{name}\n{bc}{umi}\n+\n'+('I'*24)+'\n'
    r2 = f'@{name}\n{sequence}\n+\n'+''.join(chr(q+33) for q in quality)+'\n'
    return (r1, r2), 'retained'


def extract_bam(config, condition, output, *, maximum_records=None):
    root = Path(config['root'])
    sys.path.insert(0, str(root/'python_deps'))
    import pysam
    row = next(r for r in config['raw'] if r['condition'] == condition)
    raw_root = root/'raw'/condition
    validation = json.loads((raw_root/'validation/provenance.json').read_text())
    if validation['status'] != 'complete':
        raise ValueError('Original BAM has not passed full MD5 validation')
    path = raw_root/row['filename']
    # Validate hash binding, not just a status flag left beside a different BAM.
    record = next(r for r in validation['inputs'] if Path(r['path']).name == row['filename'])
    if sha256(path) != record['sha256']:
        raise ValueError('Validated original BAM changed')
    metadata = published_metadata(config['published_root'])
    cohort = metadata[(metadata.condition == condition) & (metadata.multiplets == 'singlet')]
    permit = set(b.split('-')[0] for b in cohort.barcode)
    with Run(output, stage='Kang_original_BAM_to_FASTQ', kind='engineering',
             config=dict(condition=condition, maximum_records=maximum_records,
                         geometry='1{b[14]u[10]}2{r:}', barcode='CB_else_exact_CR', umi='UR',
                         target_expression_not_inspected=True), inputs=[path]) as run:
        counts, lengths = Counter(), Counter()
        with (run.directory/'barcodes.txt').open('x') as f:
            f.write('\n'.join(sorted(permit))+'\n')
        with pysam.AlignmentFile(str(path), 'rb', threads=4) as bam, \
             (run.directory/'barcode.fastq').open('x', buffering=8 << 20) as r1, \
             (run.directory/'cdna.fastq').open('x', buffering=8 << 20) as r2:
            for i, read in enumerate(bam):
                if maximum_records is not None and i >= maximum_records: break
                value, reason = tagged_fastq(read, permit)
                counts[reason] += 1
                if value is not None:
                    r1.write(value[0]); r2.write(value[1]); lengths[read.query_length] += 1
                if (i+1) % 1000000 == 0:
                    save_json(run.directory/'status.json', dict(status='running', scanned=i+1, counts=dict(counts)))
        if not counts['retained'] or counts['missing_raw_tags'] > .001*sum(counts.values()):
            raise ValueError('Insufficient molecular-tag recovery; do not proceed to USA')
        result = dict(status='FASTQ_READY' if maximum_records is None else 'PREFIX_ONLY_NOT_FULL',
                      condition=condition, counts=dict(counts), cdna_lengths=dict(lengths),
                      all_alignment_regions_included=True, reverse_orientation_restored=True)
        save_json(run.directory/'summary.json', result)
        (run.directory/'RESULTS.md').write_text('# Kang tagged BAM extraction\n\n'+json.dumps(result, indent=2))
    return result


def gene_symbols(gtf_path):
    genes = {}
    with gzip.open(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'): continue
            fields = line.rstrip().split('\t')
            if fields[2] != 'gene': continue
            attributes = dict(re.findall(r'(\w+) "([^"]+)"', fields[8]))
            gene = re.sub(r'\.[0-9]+(?=_PAR_Y$|$)', '', attributes['gene_id'])
            genes[gene] = attributes.get('gene_name', gene)
    return genes


def sample_keys(metadata, condition, barcodes):
    """Resolve quant barcodes to published cell IDs using each cohort's real suffix.

    Published suffixes are not uniformly '-1' (batch2 contains 313 '-11' stim
    singlets); the permit list keyed bare barcodes, so the mapping must be
    rebuilt from the metadata instead of assuming one suffix.
    """
    cohort = metadata[metadata.condition == condition]
    resolved = {}
    for cell_id, barcode in zip(cohort.cell_id, cohort.barcode):
        bare = barcode.split('-')[0]
        if resolved.setdefault(bare, cell_id) != cell_id:
            raise ValueError('Bare published barcodes are not unique within one condition')
    keys = [resolved.get(b.split('-')[0]) for b in barcodes]
    if any(k is None for k in keys) or len(set(keys)) != len(keys):
        raise ValueError('USA barcodes do not align uniquely to published sample+barcode keys')
    return keys


def import_counts(config, condition, quant, output):
    import anndata as ad
    barcodes, genes, matrices, _ = load_usa(Path(quant)/'af_quant')
    genes, (s,u,a) = collapse_gene_versions(genes, matrices)
    metadata = published_metadata(config['published_root'])
    keys = sample_keys(metadata, condition, barcodes)
    obs = metadata.loc[keys].copy()
    if not (obs.multiplets == 'singlet').all():
        raise ValueError('Non-singlet entered the explicitly singlet permit list')
    gtf = Path(config['preprocessing']['reference_gtf'])
    symbols = gene_symbols(gtf)
    names = [symbols.get(g,g) for g in genes]
    mito = np.array([n.startswith('MT-') for n in names])
    expression = s+a
    total = np.asarray(expression.sum(1)).ravel()
    detected = np.asarray((expression > 0).sum(1)).ravel()
    mt = np.asarray(expression[:, mito].sum(1)).ravel()/np.maximum(total,1)
    s_depth, u_depth = [np.asarray(x.sum(1)).ravel() for x in (s,u)]
    q = config['qc']
    passed = ((total >= q['minimum_expression_umis']) & (detected >= q['minimum_detected_genes'])
              & (mt <= q['maximum_mitochondrial_fraction']) & (s_depth > 0))
    if condition == 'ctrl': passed &= u_depth >= q['minimum_source_unspliced_umis']
    obs['technical_qc_pass'] = passed
    obs['spliced_umis'], obs['unspliced_umis'] = s_depth, u_depth
    obs['mitochondrial_fraction'] = mt
    with Run(output, stage='Kang_USA_import', kind='engineering', config=config,
             inputs=[Path(quant)/'af_quant/quant.json', gtf]) as run:
        data = ad.AnnData(expression.astype('float32'), obs=obs,
                          var=pd.DataFrame({'gene_symbol': names}, index=genes))
        data.layers.update(spliced=s.astype('float32'), unspliced=u.astype('float32'), ambiguous=a.astype('float32'))
        data.uns['veloroute'] = dict(dataset='Kang_GSE96583_batch2', condition=condition,
            raw_bam_md5=next(r['md5'] for r in config['raw'] if r['condition'] == condition),
            S_U_A_separate=True, expression='S_plus_A', future_U_used_for_target_QC=False)
        data.write_h5ad(run.directory/f'{condition}.h5ad', compression='gzip')
        groups = obs.groupby(['donor','cell_type']).agg(cells=('cell_id','size'), passed=('technical_qc_pass','sum')).reset_index()
        save_csv(run.directory/'qc_by_donor_type.csv', groups.to_dict('records'))
        result = dict(status='KANG_SU_COUNTS_READY', condition=condition, cells=len(obs), passed=int(passed.sum()),
                      genes=len(genes), median_source_U=float(np.median(u_depth)) if condition == 'ctrl' else None)
        save_json(run.directory/'summary.json', result)
        (run.directory/'RESULTS.md').write_text('# Kang USA counts\n\n'+json.dumps(result, indent=2))
    return result


def normalized_log_sparse(s, target_sum=10000.):
    depth = np.asarray(s.sum(1)).ravel()
    if np.any(depth <= 0): raise ValueError('Zero spliced depth')
    result = s.astype('float32').multiply(target_sum/depth[:,None]).tocsr()
    result.data = np.log1p(result.data)
    return result


def prepare_kang_fold(config, donor, output):
    import anndata as ad
    if donor not in config['donors']: raise ValueError('Unregistered outer donor')
    root = Path(config['root'])
    paths = [root/'processed'/side/f'{side}.h5ad' for side in ('ctrl','stim')]
    data = [ad.read_h5ad(p) for p in paths]
    if not data[0].var_names.equals(data[1].var_names): raise ValueError('Kang S/U gene orders differ')
    types = [config['cell_types']['primary']]+config['cell_types']['secondary']
    selected = [a[a.obs.technical_qc_pass & a.obs.cell_type.isin(types)].copy() for a in data]
    # Type eligibility is based on each TRAINING donor's counts only. Test cohorts
    # are kept even if low power; evaluation labels them insufficient explicitly.
    registered = config['cell_types']['minimum_cells_per_donor_side']
    eligible = []
    for cell_type in types:
        okay = [all(sum((a.obs.donor == d) & (a.obs.cell_type == cell_type)) >= registered for a in selected)
                for d in config['donors'] if d != donor]
        if sum(okay) >= 4: eligible.append(cell_type)
    if config['cell_types']['primary'] not in eligible:
        raise ValueError('Primary cell type lacks training-donor power; no silent substitution')
    selected = [a[a.obs.cell_type.isin(eligible)].copy() for a in selected]
    training = [a[a.obs.donor != donor] for a in selected]
    sfit = sparse.vstack([a.layers['spliced'] for a in training]).tocsr()
    ids = np.concatenate([a.obs_names.to_numpy(dtype=str) for a in training])
    if any(x.startswith('stim:') is False and x.startswith('ctrl:') is False for x in ids):
        raise ValueError('Missing sample-qualified cell IDs')
    rp = config['representation']
    logs = normalized_log_sparse(sfit, rp['target_sum'])
    variance = np.asarray(logs.power(2).mean(0)-np.square(logs.mean(0))).ravel()
    candidates = np.flatnonzero(np.asarray((sfit > 0).sum(0)).ravel() >= 10)
    keep = candidates[np.argsort(-variance[candidates], kind='stable')[:rp['selected_genes']]]
    if len(keep) != rp['selected_genes']: raise ValueError('Insufficient training genes')
    pca = PCA(rp['pca_components'], svd_solver='randomized', random_state=rp['seed'])
    pca.fit(logs[:,keep].toarray())
    transform = FrozenSplicingTransform(tuple(training[0].var_names), keep, pca.mean_, pca.components_,
        np.zeros(len(keep)), np.zeros(len(keep),dtype=bool), 1., np.zeros(rp['pca_components']),
        np.zeros((rp['pca_components']+1,rp['pca_components'])), object_hash(sorted(ids)), rp['target_sum'],
        estimator='not_estimated_in_pack_GFG_decoder_JVP_computed_online')
    with Run(output, stage='Kang_training_only_fold', kind='engineering', config={**config,'heldout_donor':donor},
             inputs=paths, seed=rp['seed']) as run:
        transform.save(run.directory/'transform.npz')
        fingerprint = sha256(run.directory/'transform.npz')
        save_csv(run.directory/'fit_cell_ids.csv', [dict(cell_id=x, role='train') for x in ids])
        all_labels = []
        for role in ('train','validation'):
            for side, sample in zip(('source','target'), selected):
                mask = sample.obs.donor != donor if role == 'train' else sample.obs.donor == donor
                part = sample[mask]
                if len(part) == 0: raise ValueError('Empty registered fold side')
                s = part.layers['spliced'].tocsr()
                depth = np.asarray(s.sum(1)).ravel()
                sf = s[:,keep].toarray()*(rp['target_sum']/depth[:,None])
                z = ((np.log1p(sf)-transform.mean)@transform.components.T).astype('float32')
                labels = np.array([d+'|'+t for d,t in zip(part.obs.donor,part.obs.cell_type)])
                all_labels.extend(labels.tolist())
                arrays = dict(z=z, cell_ids=part.obs_names.to_numpy(dtype=str), conditions=labels)
                meta = dict(side=side, role=role, task=config['task'], dataset='Kang_GSE96583_batch2',
                    sample_condition='ctrl' if side == 'source' else 'stim', kind='engineering',
                    heldout_donor=donor, training_donors=[d for d in config['donors'] if d != donor],
                    transform_hash=fingerprint, fit_ids_hash=transform.fit_ids_hash, estimator=transform.estimator,
                    role_semantics='outer_heldout' if role == 'validation' else 'fit', formal_ready=False,
                    grouping_only_no_donor_type_tokens=True)
                if side == 'source':
                    uf = part.layers['unspliced'][:,keep].toarray()*(rp['target_sum']/depth[:,None])
                    arrays.update(velocity=np.zeros_like(z), u_features=(np.log1p(uf)@transform.components.T).astype('float32'),
                                  u_predicted=np.zeros_like(z), depth=depth)
                save_pack(run.directory/f'{role}_{side}.npz', arrays, meta)
                if side == 'source':
                    save_pack(run.directory/f'{role}_gene_source.npz', dict(gene_us=np.concatenate((uf,sf),1).astype('float32'),
                        cell_ids=arrays['cell_ids'], gene_ids=np.array(transform.gene_ids)[keep]),
                        {**meta, 'side':'source_gene_US', 'source_hash':sha256(run.directory/f'{role}_source.npz'),
                         'normalization':'source_S_library_10000_unlogged_US_no_neighbor_smoothing'})
                else:
                    save_pack(run.directory/f'{role}_target_genes.npz', dict(gene_logspliced=np.log1p(sf).astype('float32'),
                        gene_ids=np.array(transform.gene_ids)[keep], cell_ids=arrays['cell_ids'], conditions=labels),
                        {**meta,'side':'target_gene_space','space':'log1p_S_normalized_by_S_library'})
        labels = np.array(sorted(set(all_labels)))
        # A single perturbation has one fixed token; donor/type IDs NEVER enter c.
        save_pack(run.directory/'conditions.npz', dict(conditions=labels, embeddings=np.ones((len(labels),1),dtype='float32')),
                  dict(source='registered_constant_IFNB_token_not_ESM', frozen=True, kind='engineering'))
        result = dict(status='KANG_FOLD_READY', heldout_donor=donor, eligible_cell_types=eligible,
                      transform_hash=fingerprint, source_velocity='GFG_online_only_no_residual_proxy',
                      outer_target_evaluated=False, no_test_cells_in_fit=True)
        save_json(run.directory/'summary.json', result)
        (run.directory/'RESULTS.md').write_text('# Kang frozen fold\n\n'+json.dumps(result,indent=2))
    return result
