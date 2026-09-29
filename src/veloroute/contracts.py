"""Explicit source-only inference data and training-only transform contracts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .artifacts import object_hash


def finite_matrix(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite 2D array")
    return value


def unique_strings(values, name):
    result = tuple(str(v) for v in values)
    if not result or any(not v for v in result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique nonempty identifiers")
    return result


@dataclass(frozen=True)
class SourceBatch:
    spliced: np.ndarray
    unspliced: np.ndarray
    cell_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    conditions: tuple[str, ...]
    source_day: int
    target_day: int
    task: str

    @classmethod
    def from_mapping(cls, value):
        allowed = set(cls.__dataclass_fields__)
        if set(value) != allowed:
            raise ValueError(f"Source-only contract mismatch; unknown={set(value)-allowed}, missing={allowed-set(value)}")
        return cls(**value)

    def __post_init__(self):
        s = finite_matrix(self.spliced, "source spliced")
        u = finite_matrix(self.unspliced, "source unspliced")
        if s.shape != u.shape or np.any(s < 0) or np.any(u < 0):
            raise ValueError("Raw source S/U must have identical shape and nonnegative counts")
        ids = unique_strings(self.cell_ids, "cell_ids")
        genes = unique_strings(self.gene_ids, "gene_ids")
        if s.shape != (len(ids), len(genes)) or len(self.conditions) != len(ids):
            raise ValueError("Source feature/metadata dimensions do not match")
        if any(not isinstance(c, str) or not c for c in self.conditions):
            raise ValueError("Conditions must be explicitly called nonempty labels")
        tasks = {"ER-short": (4, 5), "ER-long": (3, 5), "de-novo": (2, 5)}
        if self.task not in tasks or (self.source_day, self.target_day) != tasks[self.task]:
            raise ValueError("Task/day mismatch")
        # The source batch has no future expression, future labels, or velocity fit state.


@dataclass(frozen=True)
class FitScope:
    allowed_cell_ids: frozenset[str]

    def validate(self, cell_ids):
        ids = unique_strings(cell_ids, "fit cell IDs")
        invalid = set(ids) - self.allowed_cell_ids
        if invalid:
            raise ValueError(f"Leakage: {len(invalid)} fit cells outside training scope")
        return ids


@dataclass(frozen=True)
class FrozenPCA:
    mean: np.ndarray
    components: np.ndarray
    gene_ids: tuple[str, ...]
    fit_cell_ids_hash: str
    input_space: str

    @classmethod
    def fit(cls, x, *, cell_ids, gene_ids, scope, n_components, input_space):
        x = finite_matrix(x, "PCA input")
        ids = scope.validate(cell_ids)
        genes = unique_strings(gene_ids, "gene IDs")
        if x.shape != (len(ids), len(genes)):
            raise ValueError("PCA matrix/metadata dimensions disagree")
        if not input_space or not 1 <= n_components <= min(x.shape[0] - 1, x.shape[1]):
            raise ValueError("Declare coordinate space and valid PCA component count")
        mean = x.mean(0)
        _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
        components = vt[:n_components].copy()
        mean.setflags(write=False)
        components.setflags(write=False)
        return cls(mean, components, genes, object_hash(sorted(ids)), input_space)

    def transform(self, x, *, gene_ids, input_space):
        x = finite_matrix(x, "PCA transform input")
        if tuple(gene_ids) != self.gene_ids or input_space != self.input_space or x.shape[1] != len(self.gene_ids):
            raise ValueError("Frozen PCA requires identical gene order and coordinate space")
        return (x - self.mean) @ self.components.T

    def inverse_transform(self, z):
        z = finite_matrix(z, "PCA latent input")
        if z.shape[1] != self.components.shape[0]:
            raise ValueError("PCA latent dimension mismatch")
        return z @ self.components + self.mean


def stratified_permutation(values, strata, seed):
    """Strata are explicitly supplied source-side keys; no future statistics are read."""
    values = finite_matrix(values, "permutation values")
    if len(strata) != len(values):
        raise ValueError("Permutation strata length mismatch")
    groups = {}
    for index, key in enumerate(strata):
        if not isinstance(key, tuple) or not key or any(v is None or v == "" for v in key):
            raise ValueError("Each stratum must be an explicit tuple without invented missing labels")
        groups.setdefault(key, []).append(index)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(values))
    for group in groups.values():
        indices[group] = rng.permutation(group)
    return values[indices].copy(), indices

