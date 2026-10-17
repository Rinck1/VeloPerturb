"""W4 source-only state and analytic kinetic streams (no neural GFG or JVP).

All count inputs are raw, strict spliced/unspliced layers in the same gene
order. Library factors use ALL spliced genes, before selecting the HVGs.
The derivative holds those source factors fixed. Pooled velocity is a feature,
not the derivative of the mean log-expression state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def checked_counts(x):
    x = sparse.csr_matrix(x, dtype=np.float64)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1]:
        raise ValueError('Counts must be a nonempty cells by genes matrix')
    if not np.isfinite(x.data).all() or (x.data < 0).any():
        raise ValueError('Raw counts must be finite and nonnegative')
    return x


def normalized_pair(s, u, target_sum=10000.):
    s, u = checked_counts(s), checked_counts(u)
    if s.shape != u.shape or not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError('S/U shape or library target mismatch')
    depth = np.asarray(s.sum(1)).ravel()
    if (depth <= 0).any():
        raise ValueError('Zero-depth S cell: apply declared technical QC first')
    factors = target_sum / depth
    return s.multiply(factors[:, None]).tocsr(), u.multiply(factors[:, None]).tocsr()


def fit_gamma_tail(s, u, q=.05, min_cells=10):
    """No-intercept OLS on BOTH low and high S tails; source-only.

    Numerically matches the legacy a1 fit_gamma_tail, without importing its
    unrelated model and GPU dependencies. Unsupported genes have gamma=0.
    """
    s, u = np.asarray(s), np.asarray(u)
    if s.shape != u.shape or not 0 < q < .5:
        raise ValueError('Invalid gamma inputs')
    gamma = np.zeros(s.shape[1], dtype=np.float64)
    for g in range(s.shape[1]):
        sg = s[:, g]
        if np.count_nonzero(sg > 0) < min_cells:
            continue
        low, high = np.quantile(sg, [q, 1-q])
        tail = (sg <= low) | (sg >= high)
        denominator = np.square(sg[tail]).sum()
        if tail.sum() >= min_cells and denominator > 1e-12:
            gamma[g] = sg[tail] @ u[tail, g] / denominator
    return gamma


@dataclass
class SourceTransform:
    genes: np.ndarray
    mean: np.ndarray
    components: np.ndarray
    gamma: np.ndarray
    centers: np.ndarray
    reference_z: np.ndarray
    source_ids: np.ndarray
    gene_names: np.ndarray
    target_sum: float
    seed: int

    @classmethod
    def fit(cls, s, u, *, cell_ids, gene_names, allowed_ids, n_hvg=2000,
            dimensions=50, clusters=8, target_sum=10000., seed=0):
        """No target argument exists. Caller supplies an explicit fit allowlist."""
        ids = np.asarray(cell_ids, dtype=str)
        names = np.asarray(gene_names, dtype=str)
        if len(set(ids)) != len(ids) or not set(ids) <= set(allowed_ids):
            raise ValueError('Duplicate or forbidden source fit cell IDs')
        sn, un = normalized_pair(s, u, target_sum)
        if len(ids) != sn.shape[0] or len(names) != sn.shape[1] or len(set(names)) != len(names):
            raise ValueError('Source row/gene identity mismatch')
        if not dimensions <= n_hvg <= sn.shape[1] or dimensions >= sn.shape[0] or clusters > sn.shape[0]:
            raise ValueError('Insufficient source cells/genes for registered geometry')
        logs = sn.copy(); logs.data = np.log1p(logs.data)
        mu = np.asarray(logs.mean(0)).ravel()
        variance = np.asarray(logs.power(2).mean(0)).ravel() - mu**2
        # Stable ties follow the provided gene order; no target-based HVGs.
        genes = np.argsort(-variance, kind='stable')[:n_hvg]
        logs = logs[:, genes].toarray()
        pca = PCA(n_components=dimensions, svd_solver='full').fit(logs)
        z = pca.transform(logs)
        centers = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit(z).cluster_centers_
        return cls(genes, pca.mean_, pca.components_,
                   fit_gamma_tail(sn[:, genes].toarray(), un[:, genes].toarray()),
                   centers, z, ids, names, float(target_sum), int(seed))

    def normalized(self, s, u):
        if s.shape[1] != len(self.gene_names):
            raise ValueError('Input gene dimension changed')
        sn, un = normalized_pair(s, u, self.target_sum)
        return sn[:, self.genes].toarray(), un[:, self.genes].toarray()

    def state(self, s):
        sn, _ = self.normalized(s, sparse.csr_matrix(s.shape))
        return self.state_normalized(sn)

    def state_normalized(self, sn):
        return (np.log1p(sn) - self.mean) @ self.components.T

    def projected_rate(self, sn, rate):
        """Jacobian of log1p-PCA with FIXED source library factors."""
        return (np.asarray(rate) / (1 + np.asarray(sn))) @ self.components.T

    def velocity(self, s, u):
        sn, un = self.normalized(s, u)
        return self.projected_rate(sn, un - self.gamma*sn)

    def fingerprint(self):
        digest = hashlib.sha256()
        for name in self.__dataclass_fields__:
            a = np.asarray(getattr(self, name))
            digest.update(name.encode()); digest.update(str(a.dtype).encode())
            digest.update(str(a.shape).encode()); digest.update(a.tobytes())
        return digest.hexdigest()

    def save(self, path):
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            values = {k: f[k] for k in cls.__dataclass_fields__}
        values['target_sum'] = float(values['target_sum']); values['seed'] = int(values['seed'])
        return cls(**values)


def group_mean(x, cell_groups, labels):
    cell_groups = np.asarray(cell_groups, dtype=str)
    if len(cell_groups) != len(x) or len(set(labels)) != len(labels):
        raise ValueError('Invalid population identity')
    result = []
    for label in labels:
        keep = cell_groups == label
        if not keep.any():
            raise ValueError(f'Missing population: {label}')
        result.append(np.asarray(x)[keep].mean(0))
    return np.stack(result)


def pooled_counts(x, cell_groups, labels):
    x = checked_counts(x); groups = np.asarray(cell_groups, dtype=str)
    if len(groups) != x.shape[0] or len(set(labels)) != len(labels):
        raise ValueError('Invalid pooled population identity')
    rows = []
    for label in labels:
        keep = groups == label
        if not keep.any():
            raise ValueError(f'Missing population: {label}')
        rows.append(np.asarray(x[keep].sum(0)).ravel())
    return np.stack(rows)


def pseudo_velocity(transform, z, ids, neighbors=30):
    if neighbors >= len(transform.reference_z):
        raise ValueError('Insufficient training source cells for pseudo-S neighbors')
    distances, indices = NearestNeighbors(n_neighbors=neighbors+1).fit(transform.reference_z).kneighbors(z)
    del distances
    result = []
    for row, (identity, candidates) in enumerate(zip(ids, indices)):
        # A training cell may be in its own reference set: exclude by identity.
        keep = candidates[transform.source_ids[candidates] != str(identity)][:neighbors]
        result.append(transform.reference_z[keep].mean(0)-z[row])
    return np.stack(result)


def population_features(transform, s, u, cell_groups, labels, cell_ids, neighbors=30):
    labels = np.asarray(labels, dtype=str)
    sp, up = pooled_counts(s, cell_groups, labels), pooled_counts(u, cell_groups, labels)
    sn, un = transform.normalized(sp, up)
    z = transform.state(s)
    return dict(labels=labels, state=group_mean(z, cell_groups, labels),
                vel_pool=transform.projected_rate(sn, un-transform.gamma*sn),
                vel_mean=group_mean(transform.velocity(s, u), cell_groups, labels),
                direct_U_pool=np.log1p(un) @ transform.components.T,
                pseudo_S=group_mean(pseudo_velocity(transform, z, cell_ids, neighbors), cell_groups, labels),
                pooled_s=sp, pooled_u=up)


def swap_pooled_u(recipient_u, recipient_z, recipient_labels, donor_u, donor_z, donor_labels, *, seed=0):
    """Nearest different TRAINING population; not an exact conditional null.

    Rescales ALL genes to the recipient U total. Reuse is allowed and audited.
    Seed affects distance ties only. Zero-depth donors are ineligible; zero-depth
    recipients remain zero. S is not an argument and cannot be modified.
    """
    ru, du = np.asarray(recipient_u), np.asarray(donor_u)
    rz, dz = np.asarray(recipient_z), np.asarray(donor_z)
    if ru.ndim != 2 or du.ndim != 2 or ru.shape[1] != du.shape[1]:
        raise ValueError('Incompatible pooled U matrices')
    if not np.isfinite(ru).all() or not np.isfinite(du).all() or (ru < 0).any() or (du < 0).any():
        raise ValueError('Invalid U pool')
    if len(ru) != len(recipient_labels) or len(du) != len(donor_labels) or len(rz) != len(ru) or len(dz) != len(du):
        raise ValueError('Population row mismatch')
    rng = np.random.default_rng(seed); result = np.empty_like(ru, dtype=float); audit = []
    for i, label in enumerate(recipient_labels):
        distance = np.linalg.norm(dz-rz[i], axis=1)
        valid = (np.asarray(donor_labels) != label) & (du.sum(1) > 0)
        if not valid.any():
            raise ValueError('No eligible different training U donor')
        nearest = np.flatnonzero(valid & np.isclose(distance, distance[valid].min(), rtol=1e-12, atol=1e-12))
        j = int(rng.choice(nearest)); total = float(ru[i].sum())
        result[i] = du[j] * (total / du[j].sum())
        audit.append(dict(recipient=str(label), donor=str(donor_labels[j]), distance=float(distance[j]),
                          recipient_u_total=total, donor_u_total=float(du[j].sum()),
                          zero_depth_recipient=total == 0, ties=len(nearest)))
    return result, audit


def add_swap_feature(transform, recipients, donors, seed=0):
    swapped, audit = swap_pooled_u(recipients['pooled_u'], recipients['state'], recipients['labels'],
        donors['pooled_u'], donors['state'], donors['labels'], seed=seed)
    sn, un = transform.normalized(recipients['pooled_s'], swapped)
    result = dict(recipients)
    result['vel_pool_Uswap'] = transform.projected_rate(sn, un-transform.gamma*sn)
    return result, audit
