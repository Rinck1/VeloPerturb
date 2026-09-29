"""Build exploratory gain-probe packs from packaged S/U datasets. Not pre-registered."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from veloroute.artifacts import Run, save_json, sha256
from veloroute.contracts import FitScope
from veloroute.latent import FrozenSplicingTransform, save_pack


def load_su(path):
    import anndata as ad
    data = ad.read_h5ad(path)
    for layer in ('spliced', 'unspliced'):
        if layer not in data.layers:
            raise ValueError(f'Missing {layer} layer: {path}')
    s = sparse.csr_matrix(data.layers['spliced']).astype('float32')
    u = sparse.csr_matrix(data.layers['unspliced']).astype('float32')
    genes = tuple(str(g) for g in data.var_names)
    if len(set(genes)) != len(genes):
        raise ValueError('Duplicated gene names; make unique before preparing')
    return data, s, u, genes


def lung_groups(data):
    day = data.obs['day'].astype(str)
    transitions = {str(d): str(d+1) for d in range(2, 13)}
    transitions['13'] = '15'
    train_days = [str(d) for d in range(2, 10)]
    validation_days = ['11', '12', '13']
    groups = {}
    for label in train_days+validation_days:
        source = np.flatnonzero((day == label).to_numpy())
        target = np.flatnonzero((day == transitions[label]).to_numpy())
        if len(source) < 40 or len(target) < 40:
            raise ValueError(f'Day {label} transition lacks cells')
        groups[label] = dict(source=source, target=target)
    return groups, train_days, validation_days


def pathway_groups(data, source_cluster, target_cluster, labels, column='clusters_fine'):
    cluster = data.obs[column].astype(str)
    groups = {}
    for label in labels:
        source = np.flatnonzero((cluster == source_cluster[label]).to_numpy())
        target = np.flatnonzero((cluster == target_cluster[label]).to_numpy())
        if len(source) < 40 or len(target) < 40:
            raise ValueError(f'Pathway {label} lacks cells ({len(source)}/{len(target)})')
        groups[label] = dict(source=source, target=target)
    return groups


def build(config, dataset, h5ad, output, swap=False):
    data, s, u, genes = load_su(h5ad)
    if dataset == 'lung':
        if swap:
            raise ValueError('Lung day transitions do not support condition swap')
        groups, train_labels, validation_labels = lung_groups(data)
        task = 'lung_regeneration_day_transition'
    elif dataset == 'pancreas_pathways':
        train_labels = ['Alpha', 'Beta']
        validation_labels = ['Delta', 'Epsilon']
        source_cluster = dict(Alpha='Pre-Alpha', Beta='Pre-Beta', Delta='Fev+ Delta', Epsilon='Fev+ Epsilon')
        target_cluster = dict(Alpha='Alpha', Beta='Beta', Delta='Delta', Epsilon='Epsilon')
        groups = pathway_groups(data, source_cluster, target_cluster, train_labels+validation_labels)
        task = 'pancreas_endocrinogenesis_pathway'
    elif dataset == 'bonemarrow_pathways':
        train_labels = ['Ery', 'Mono']
        validation_labels = ['Mega', 'Dendritic']
        source_cluster = dict(Ery='Ery_1', Mono='Mono_1', Mega='Precursors', Dendritic='CLP')
        target_cluster = dict(Ery='Ery_2', Mono='Mono_2', Mega='Mega', Dendritic='DCs')
        groups = pathway_groups(data, source_cluster, target_cluster, train_labels+validation_labels,
            column='clusters')
        task = 'bonemarrow_maturation_pathway'
    elif dataset == 'lung_pairs':
        if swap:
            raise ValueError('Lung day transitions do not support condition swap')
        day = data.obs['day'].astype(str)
        transitions = {'2': '3', '4': '5', '6': '7', '8': '9', '10': '11', '12': '13'}
        train_labels = ['2', '4', '6', '8']
        validation_labels = ['10', '12']
        groups = {}
        for label in train_labels+validation_labels:
            source = np.flatnonzero((day == label).to_numpy())
            target = np.flatnonzero((day == transitions[label]).to_numpy())
            if len(source) < 40 or len(target) < 40:
                raise ValueError(f'Day {label} transition lacks cells')
            groups[label] = dict(source=source, target=target)
        task = 'lung_regeneration_day_transition_nonoverlap'
    elif dataset == 'dentate_pairs':
        train_labels = ['nb_gi_35']
        validation_labels = ['nb_gi_12', 'gi_gm_12']
        spec = {'nb_gi_35': ('35', 'Neuroblast', 'Granule immature'),
                'nb_gi_12': ('12', 'Neuroblast', 'Granule immature'),
                'gi_gm_12': ('12', 'Granule immature', 'Granule mature')}
        age = data.obs['age(days)'].astype(str)
        cluster = data.obs['clusters'].astype(str)
        groups = {}
        for label in train_labels+validation_labels:
            group_age, source_name, target_name = spec[label]
            source = np.flatnonzero(((age == group_age) & (cluster == source_name)).to_numpy())
            target = np.flatnonzero(((age == group_age) & (cluster == target_name)).to_numpy())
            if len(source) < 40 or len(target) < 40:
                raise ValueError(f'Dentate group {label} lacks cells ({len(source)}/{len(target)})')
            groups[label] = dict(source=source, target=target)
        task = 'dentate_gyrus_age_transition_nonoverlap'
    elif dataset == 'dentate_ages':
        train_labels = ['nb_gi_35', 'gi_gm_35']
        validation_labels = ['nb_gi_12', 'gi_gm_12']
        spec = {'nb_gi_35': ('35', 'Neuroblast', 'Granule immature'),
                'gi_gm_35': ('35', 'Granule immature', 'Granule mature'),
                'nb_gi_12': ('12', 'Neuroblast', 'Granule immature'),
                'gi_gm_12': ('12', 'Granule immature', 'Granule mature')}
        age = data.obs['age(days)'].astype(str)
        cluster = data.obs['clusters'].astype(str)
        groups = {}
        for label in train_labels+validation_labels:
            group_age, source_name, target_name = spec[label]
            source = np.flatnonzero(((age == group_age) & (cluster == source_name)).to_numpy())
            target = np.flatnonzero(((age == group_age) & (cluster == target_name)).to_numpy())
            if len(source) < 40 or len(target) < 40:
                raise ValueError(f'Dentate group {label} lacks cells ({len(source)}/{len(target)})')
            groups[label] = dict(source=source, target=target)
        task = 'dentate_gyrus_age_transition'
    else:
        raise ValueError('Unknown dataset task')
    if swap:
        train_labels, validation_labels = validation_labels, train_labels
    cell_ids = np.asarray([f'{dataset}:{i}' for i in range(data.n_obs)], dtype=str)
    parts = {}
    for role, labels in (('train', train_labels), ('validation', validation_labels)):
        for side in ('source', 'target'):
            index = np.concatenate([groups[label][side] for label in labels])
            conditions = np.concatenate([np.full(len(groups[label][side]), label) for label in labels])
            order = np.argsort(index, kind='stable')
            parts[role, side] = dict(index=index[order], conditions=conditions[order])
    fit_index = np.concatenate([parts['train', side]['index'] for side in ('source', 'target')])
    fit_index = np.unique(fit_index)
    fit_ids = cell_ids[fit_index]
    scope = FitScope(frozenset(fit_ids.tolist()))
    transform = FrozenSplicingTransform.fit(s[fit_index], u[fit_index], gene_ids=genes, cell_ids=fit_ids,
        scope=scope, n_genes=config['n_genes'], n_components=config['n_components'], seed=config['seed'])
    with Run(output, stage='exploratory_gain_probe_prepare', kind='engineering',
             config=dict(**config, dataset=dataset, task=task, h5ad=str(h5ad)),
             inputs=[h5ad]) as run:
        transform.save(run.directory/'transform.npz')
        fingerprint = sha256(run.directory/'transform.npz')
        labels = train_labels+validation_labels
        save_pack(run.directory/'conditions.npz',
            dict(conditions=np.asarray(labels),
                 embeddings=np.eye(len(labels), dtype='float32')),
            dict(source='declared_frozen_one_hot_for_exploratory_gain_screen', frozen=True,
                 kind='engineering'))
        for role, role_labels in (('train', train_labels), ('validation', validation_labels)):
            for side in ('source', 'target'):
                part = parts[role, side]
                enc = transform.transform(s[part['index']], u[part['index']], gene_ids=genes)
                meta = dict(side=side, role=role, kind='exploratory_gain_probe', task=task,
                    dataset=dataset, transform_hash=fingerprint, fit_ids_hash=transform.fit_ids_hash,
                    condition_labels=role_labels,
                    role_semantics='fit' if role == 'train' else 'exploratory_holdout',
                    formal_ready=False, individual_fate_claim=False)
                if side == 'source':
                    enc['depth'] = np.asarray(s[part['index']].sum(1)).ravel().astype('float32')
                    enc['cell_ids'] = cell_ids[part['index']]
                    enc['conditions'] = part['conditions']
                    source_path = run.directory/f'{role}_{side}.npz'
                    save_pack(source_path, enc, meta)
                    from veloroute.latent import normalized_counts
                    sn, un = normalized_counts(s[part['index']], u[part['index']], transform.target_sum)
                    gene_us = np.concatenate([un[:, transform.selected], sn[:, transform.selected]], 1).astype('float32')
                    gp_meta = dict(side='source_gene_US', role=role, kind='exploratory_gain_probe', task=task,
                        dataset=dataset, transform_hash=fingerprint,
                        source_hash=sha256(source_path),
                        normalization='source_S_library_10000_unlogged_US_no_neighbor_smoothing')
                    save_pack(run.directory/f'{role}_gene_source.npz',
                        dict(gene_us=gene_us, gene_ids=np.asarray(genes)[transform.selected],
                             cell_ids=cell_ids[part['index']]), gp_meta)
                    continue
                enc = {k: enc[k] for k in ('z',)}
                enc['cell_ids'] = cell_ids[part['index']]
                enc['conditions'] = part['conditions']
                save_pack(run.directory/f'{role}_{side}.npz', enc, meta)
        result = dict(status='GAIN_PROBE_PACKS_READY', dataset=dataset, task=task,
            train_conditions=train_labels, validation_conditions=validation_labels,
            cells=dict(train_source=len(parts['train', 'source']['index']),
                       train_target=len(parts['train', 'target']['index']),
                       validation_source=len(parts['validation', 'source']['index']),
                       validation_target=len(parts['validation', 'target']['index'])),
            transform_hash=fingerprint, outer_outcomes_read=False)
        save_json(run.directory/'summary.json', result)
        (run.directory/'RESULTS.md').write_text('# Exploratory gain-probe packs\n\n'+json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['lung', 'lung_pairs', 'pancreas_pathways', 'bonemarrow_pathways', 'dentate_ages', 'dentate_pairs'])
    parser.add_argument('--h5ad', required=True)
    parser.add_argument('--config', default='configs/veloroute_gain_probe_20260915.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--swap', action='store_true')
    args = parser.parse_args()
    from veloroute.artifacts import load_config
    print(json.dumps(build(load_config(args.config), args.dataset, Path(args.h5ad), Path(args.output),
        swap=args.swap), indent=2))
