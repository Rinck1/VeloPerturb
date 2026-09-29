"""Frozen, training-only S/U transforms for real-data pipeline integration.

The initial estimator is an explicit steady-state residual baseline, not VeloVI
or GFG and not calibrated in days. Its assumptions must pass data diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA

from .artifacts import Run, object_hash, save_csv, save_json, sha256
from .contracts import FitScope, unique_strings


def dense_counts(value):
    value = value.toarray() if sparse.issparse(value) else np.asarray(value)
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError('Expected finite nonnegative cell-by-gene counts')
    return value


def normalized_counts(s, u, target_sum=10000.):
    s, u = dense_counts(s), dense_counts(u)
    if s.shape != u.shape:
        raise ValueError('S/U shape mismatch')
    total = s.sum(1)
    if np.any(total <= 0):
        raise ValueError('Zero spliced library; exclude explicitly by QC before transformation')
    # S-only arms must not gain U indirectly through their library-size divisor.
    scale = target_sum/total
    return s*scale[:, None], u*scale[:, None]


@dataclass(frozen=True)
class FrozenSplicingTransform:
    gene_ids: tuple[str, ...]
    selected: np.ndarray
    mean: np.ndarray
    components: np.ndarray
    gamma: np.ndarray
    supported: np.ndarray
    velocity_scale: float
    u_mean: np.ndarray
    u_coef: np.ndarray
    fit_ids_hash: str
    target_sum: float
    estimator: str = 'steady_state_residual_beta1_tail_ols_v1'

    @classmethod
    def fit(cls, s, u, *, gene_ids, cell_ids, scope, n_genes=2000, n_components=50, seed=0,
            target_sum=10000., upper_quantile=.8, minimum_u_cells=10):
        ids = scope.validate(cell_ids)
        genes = unique_strings(gene_ids, 'genes')
        s, u = normalized_counts(s, u, target_sum)
        if s.shape != (len(ids), len(genes)):
            raise ValueError('Fit counts and metadata mismatch')
        if not 0 < upper_quantile < 1 or minimum_u_cells < 1:
            raise ValueError('Invalid estimator support parameters')
        logs = np.log1p(s)
        # Feature selection uses only training spliced abundance.
        variance = logs.var(0)
        candidates = np.flatnonzero((s > 0).sum(0) >= min(10, len(ids)))
        selected = candidates[np.argsort(-variance[candidates], kind='stable')[:n_genes]]
        if n_components > min(len(selected), len(ids)-1):
            raise ValueError('Insufficient training cells/genes for declared PCA dimensions')
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=seed)
        z = pca.fit_transform(logs[:, selected])
        sf, uf = s[:, selected], u[:, selected]
        tail = sf >= np.quantile(sf, upper_quantile, axis=0)
        denominator = (sf*sf*tail).sum(0)
        gamma = (sf*uf*tail).sum(0)/np.maximum(denominator, 1e-8)
        supported = ((uf > 0).sum(0) >= minimum_u_cells) & (denominator > 1e-8) & (gamma > 0)
        gamma = np.where(supported, gamma, 0)
        velocity_gene = np.where(supported, (uf - sf*gamma)/(1+sf), 0.)
        projected = velocity_gene @ pca.components_.T
        velocity_scale = float(np.sqrt(np.mean(projected**2)))
        if not np.isfinite(velocity_scale) or velocity_scale < 1e-10:
            raise ValueError('No supported nonzero velocity; report diagnostic failure, do not invent signal')
        u_latent = np.log1p(uf) @ pca.components_.T
        design = np.column_stack((np.ones(len(z)), z))
        penalty = np.eye(design.shape[1]); penalty[0, 0] = 0
        coef = np.linalg.solve(design.T@design + penalty, design.T@u_latent)
        return cls(genes, selected, pca.mean_, pca.components_, gamma, supported, velocity_scale,
                   u_latent.mean(0), coef, object_hash(sorted(ids)), target_sum)

    def transform(self, s, u, *, gene_ids):
        if tuple(gene_ids) != self.gene_ids:
            raise ValueError('Frozen gene order mismatch')
        s, u = normalized_counts(s, u, self.target_sum)
        s, u = s[:, self.selected], u[:, self.selected]
        z = (np.log1p(s)-self.mean) @ self.components.T
        # Chain rule for log1p. No PCA mean subtraction for vectors.
        vg = np.where(self.supported, (u-s*self.gamma)/(1+s), 0.)
        velocity = (vg @ self.components.T) / self.velocity_scale
        u_features = np.log1p(u) @ self.components.T - self.u_mean
        u_predicted = np.column_stack((np.ones(len(z)), z)) @ self.u_coef - self.u_mean
        return {'z': z.astype(np.float32), 'velocity': velocity.astype(np.float32),
                'u_features': u_features.astype(np.float32), 'u_predicted': u_predicted.astype(np.float32)}

    def decode(self, z):
        z = np.asarray(z)
        if z.ndim != 2 or z.shape[1] != len(self.components) or not np.isfinite(z).all():
            raise ValueError('Invalid latent states')
        return z @ self.components + self.mean

    def save(self, path):
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream, **{k: np.asarray(v) for k, v in vars(self).items()})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            values = {k: data[k] for k in data.files}
        values['gene_ids'] = tuple(values['gene_ids'].tolist())
        for k in ('velocity_scale', 'target_sum'):
            values[k] = float(values[k])
        for k in ('fit_ids_hash', 'estimator'):
            values[k] = str(values[k])
        return cls(**values)


def save_pack(path, arrays, metadata):
    import json
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays, metadata_json=np.array(json.dumps(metadata, sort_keys=True)))


def load_pack(path, *, expected_side=None):
    import json
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata_json']))
        arrays = {k: data[k] for k in data.files if k != 'metadata_json'}
    if expected_side and metadata.get('side') != expected_side:
        raise ValueError(f'Expected {expected_side}-only data pack')
    if expected_side == 'source':
        allowed = {'z', 'velocity', 'u_features', 'u_predicted', 'cell_ids', 'conditions', 'depth'}
    elif expected_side == 'target':
        allowed = {'z', 'cell_ids', 'conditions'}
    else:
        allowed = set(arrays)
    if set(arrays) != allowed:
        raise ValueError('Unknown or missing pack fields; future-input leakage guard')
    n = len(arrays['z'])
    if any(len(v) != n for v in arrays.values()) or n == 0:
        raise ValueError('Pack length mismatch/empty pack')
    unique_strings(arrays['cell_ids'], 'pack cell IDs')
    for key, value in arrays.items():
        if value.dtype.kind in 'fc' and not np.isfinite(value).all():
            raise ValueError(f'Nonfinite pack field: {key}')
    return arrays, metadata


def prepare_fold(source_path, target_path, config, output, *, kind='engineering'):
    import anndata as ad
    if kind not in {'engineering', 'development', 'synthetic'}:
        raise ValueError('Prepare confirmation only through predeclared sealed packs, not a confirmation fit')
    with Run(output, stage='prepare_fold', kind=kind, config=config, inputs=[source_path, target_path], seed=config['seed']) as run:
        datasets = [ad.read_h5ad(p) for p in (source_path, target_path)]
        if kind == 'synthetic' and any(a.uns.get('veloroute', {}).get('data_origin') != 'veloroute_synthetic_counts_v1' for a in datasets):
            raise ValueError('Synthetic pipeline requires explicitly synthetic fixtures')
        if any(not a.var_names.equals(datasets[0].var_names) for a in datasets):
            raise ValueError('Day4/day5 gene order must be identical; align explicit reference first')
        for a, expected_day in zip(datasets, (4, 5)):
            if set(a.obs.day.astype(int)) != {expected_day} or any(k not in a.layers for k in ('spliced', 'unspliced')):
                raise ValueError('ER-short requires day4/day5 S/U')
            if kind == 'development' and not a.uns.get('veloroute', {}).get('formal_ready', False):
                raise ValueError('Data/QC/guide approval missing; engineering is not formal readiness')
        selected = [a[a.obs.technical_qc_pass & a.obs.guide_call_pass].copy() for a in datasets]
        training = [a[a.obs.role == 'train'] for a in selected]
        ids = np.concatenate([a.obs_names.to_numpy() for a in training])
        scope = FitScope(frozenset(ids))
        s = sparse.vstack([a.layers['spliced'] for a in training]).tocsr()
        u = sparse.vstack([a.layers['unspliced'] for a in training]).tocsr()
        transform = FrozenSplicingTransform.fit(s, u, gene_ids=training[0].var_names,
                    cell_ids=ids, scope=scope, **config)
        transform.save(run.directory/'transform.npz')
        fingerprint = sha256(run.directory/'transform.npz')
        save_csv(run.directory/'fit_cell_ids.csv', [{'cell_id': v, 'role': 'train'} for v in ids])
        for role in ('train', 'validation', 'confirmation'):
            for side, a, day in zip(('source', 'target'), selected, (4, 5)):
                part = a[a.obs.role == role]
                if len(part) == 0:
                    raise ValueError(f'No QC-passing {role} {side} cells; cannot silently omit split')
                if side == 'source':
                    arrays = transform.transform(part.layers['spliced'], part.layers['unspliced'], gene_ids=part.var_names)
                    arrays['depth'] = np.asarray(part.layers['spliced'].sum(1)).ravel()
                else:
                    # Future U is neither transformed nor exported to the model.
                    spliced = dense_counts(part.layers['spliced'])
                    normalized, _ = normalized_counts(spliced, np.zeros_like(spliced), transform.target_sum)
                    arrays = {'z': ((np.log1p(normalized[:, transform.selected])-transform.mean)
                                    @ transform.components.T).astype(np.float32)}
                arrays.update(cell_ids=part.obs_names.to_numpy(dtype=str), conditions=part.obs.condition.to_numpy(dtype=str))
                metadata = {'side': side, 'role': role, 'day': day, 'task': 'ER-short', 'kind': kind,
                            'transform_hash': fingerprint, 'estimator': transform.estimator,
                            'fit_ids_hash': transform.fit_ids_hash, 'formal_ready': kind == 'development'}
                save_pack(run.directory/f'{role}_{side}.npz', arrays, metadata)
                if side == 'target':
                    save_pack(run.directory/f'{role}_target_genes.npz', {
                        'gene_logspliced': np.log1p(normalized[:, transform.selected]).astype(np.float32),
                        'gene_ids': np.array(transform.gene_ids)[transform.selected],
                        'cell_ids': arrays['cell_ids'], 'conditions': arrays['conditions']},
                        {**metadata, 'side': 'target_gene_space', 'space': 'log1p_S_normalized_by_S_library'})
        summary = {'status': 'FROZEN_PACKS_CREATED', 'kind': kind, 'estimator': transform.estimator,
                   'units': 'standardized_arbitrary_velocity_not_days', 'train_cells': len(ids),
                   'selected_genes': len(transform.selected), 'supported_velocity_genes': int(transform.supported.sum()),
                   'G1': 'NOT_RUN', 'transform_hash': fingerprint}
        save_json(run.directory/'summary.json', summary)
        (run.directory/'RESULTS.md').write_text('# Frozen data fold\n\n'+str(summary)+'\n\n'
            'No confirmation outcomes evaluated. The simple steady-state estimator is not GFG/VeloVI. '
            'Its support and incremental value remain to be established.\n')
    return summary
