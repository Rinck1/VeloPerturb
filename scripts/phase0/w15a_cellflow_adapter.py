"""Source-only W15-A CellFlow conditions; official CellFlow remains unmodified.

The interpolation time tau is never used to build biological-time covariates.
No function constructing a kinetic token accepts target counts or target state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from w4_streams import SourceTransform, checked_counts
from w5_scntseq_repair import regularized_gamma


ROOT = Path(__file__).resolve().parents[2]
CELLFLOW_ROOT = Path('/data/yuchang/veloroute_kang_20260914/external/CellFlow')
ARMS = ('cf_static', 'cf_kin', 'cf_Ssum', 'cf_kin_shuf', 'cf_directU', 'cf_kin_cell')
NOT_POOLED = ('bio_time', 'bio_delta', 'population_token', 'cell_token')


def array_hash(*values):
    result = hashlib.sha256()
    for value in values:
        if sparse.issparse(value):
            value = sparse.csr_matrix(value)
            items = (value.data, value.indices, value.indptr, np.array(value.shape))
        else:
            items = (np.asarray(value),)
        for item in items:
            result.update(str(item.dtype).encode()); result.update(str(item.shape).encode())
            result.update(item.tobytes())
    return result.hexdigest()


@dataclass(frozen=True)
class SourceCohort:
    condition: str
    time_hours: float
    replicate: str
    s: sparse.csr_matrix
    u: sparse.csr_matrix
    cell_ids: np.ndarray

    def __post_init__(self):
        s, u = checked_counts(self.s), checked_counts(self.u)
        ids = np.asarray(self.cell_ids, dtype=str)
        if s.shape != u.shape or s.shape[0] != len(ids) or len(set(ids)) != len(ids):
            raise ValueError('Source cohort has inconsistent counts or cell IDs')
        if not np.isfinite(self.time_hours) or self.time_hours < 0:
            raise ValueError('Biological source time must be finite and nonnegative')
        object.__setattr__(self, 's', s); object.__setattr__(self, 'u', u)
        object.__setattr__(self, 'cell_ids', ids)

    @property
    def key(self):
        return f'{self.replicate}|{self.condition}|{self.time_hours:g}h'

    def fingerprint(self):
        return array_hash(self.s, self.u, self.cell_ids, self.key)


def select_source_cohort(s, u, cell_ids, source_indices, *, allowed_source_ids,
                         condition, time_hours, replicate):
    """Select exactly the declared source rows, before computing any statistics."""
    rows = np.asarray(source_indices)
    if rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows):
        raise ValueError('Source indices must be a nonempty integer vector')
    if len(set(rows.tolist())) != len(rows) or rows.min() < 0 or rows.max() >= len(cell_ids):
        raise ValueError('Duplicate or out-of-range source indices')
    ids = np.asarray(cell_ids, dtype=str)[rows]
    if not set(ids) <= set(allowed_source_ids):
        raise ValueError('Source selection includes a forbidden target cell')
    return SourceCohort(condition, float(time_hours), replicate,
                        sparse.csr_matrix(s)[rows], sparse.csr_matrix(u)[rows], ids)


def fit_source_transform(cohorts, gene_names, *, allowed_source_ids,
                         n_hvg=2000, dimensions=50, clusters=8, seed=0):
    """Caller supplies training SOURCE cohorts only, not destination snapshots."""
    if not cohorts:
        raise ValueError('At least one training source cohort is required')
    s, u = [sparse.vstack([getattr(c, key) for c in cohorts], format='csr') for key in ('s', 'u')]
    ids = np.concatenate([c.cell_ids for c in cohorts])
    transform = SourceTransform.fit(s, u, cell_ids=ids, gene_names=gene_names,
        allowed_ids=allowed_source_ids, n_hvg=n_hvg, dimensions=dimensions,
        clusters=clusters, seed=seed)
    sn, un = transform.normalized(s.sum(0), u.sum(0))
    transform.gamma, _ = regularized_gamma(sn[0], un[0])
    return transform


def pooled_tokens(transform, cohort):
    """Pool RAW counts first; normalize S and U with the SAME all-gene S depth."""
    sn, un = transform.normalized(cohort.s.sum(0), cohort.u.sum(0))
    return dict(kin=transform.projected_rate(sn, un-transform.gamma*sn)[0],
                Ssum=(np.log1p(sn) @ transform.components.T)[0],
                directU=(np.log1p(un) @ transform.components.T)[0])


def source_tokens(transform, cohort, reference=None, *, neighbors=64):
    """All references must also be observed source-side populations.

    A historical earliest-time reference or same-time non-targeting reference is
    allowed. A future reference is rejected. No implicit target-derived reference.
    """
    if reference is not None and reference.time_hours > cohort.time_hours:
        raise ValueError('Reference population comes from the future')
    current = pooled_tokens(transform, cohort)
    baseline = (pooled_tokens(transform, reference) if reference is not None else
                {key: np.zeros_like(value) for key, value in current.items()})
    population = {key: value-baseline[key] for key, value in current.items()}
    z = transform.state(cohort.s)
    count = min(int(neighbors), len(z))
    if count < 1:
        raise ValueError('At least one source neighbor is required')
    indices = NearestNeighbors(n_neighbors=count).fit(z).kneighbors(z, return_distance=False)
    adjacency = sparse.csr_matrix((np.ones(indices.size),
        (np.repeat(np.arange(len(z)), count), indices.ravel())), shape=(len(z), len(z)))
    depth = np.asarray(adjacency @ np.asarray(cohort.s.sum(1)).ravel())[:, None]
    if (depth <= 0).any():
        raise ValueError('Source neighborhood has zero spliced depth')
    factor = transform.target_sum/depth
    sn = (adjacency @ cohort.s[:, transform.genes]).toarray()*factor
    un = (adjacency @ cohort.u[:, transform.genes]).toarray()*factor
    local = transform.projected_rate(sn, un-transform.gamma*sn)-baseline['kin']
    return dict(z=z, population=population, cell_kin=local, source_ids=cohort.cell_ids.copy(),
                source_key=cohort.key, source_hash=cohort.fingerprint(),
                reference_key=None if reference is None else reference.key,
                reference_policy='zero_no_control' if reference is None else 'observed_source_reference')


@dataclass(frozen=True)
class TokenNormalizer:
    mean: dict
    scale: dict
    training_source_keys: tuple

    @classmethod
    def fit(cls, packs, *, allowed_training_source_keys):
        if not packs or len({p['source_key'] for p in packs}) != len(packs):
            raise ValueError('Training token packs must be nonempty and unique')
        keys = tuple(p['source_key'] for p in packs)
        if not set(keys) <= set(allowed_training_source_keys):
            raise ValueError('Normalization contains a forbidden source cohort')
        values = {key: np.stack([p['population'][key] for p in packs])
                  for key in ('kin', 'Ssum', 'directU')}
        values['cell_kin'] = np.concatenate([p['cell_kin'] for p in packs])
        return cls({k: v.mean(0) for k, v in values.items()},
                   {k: np.maximum(v.std(0), 1e-5) for k, v in values.items()}, keys)

    def apply(self, kind, value):
        return (np.asarray(value)-self.mean[kind])/self.scale[kind]

    def fingerprint(self):
        return array_hash(*(self.mean[k] for k in sorted(self.mean)),
                          *(self.scale[k] for k in sorted(self.scale)), self.training_source_keys)


def biological_fourier(hours):
    """Four frozen radian frequencies {1,2,4,8} on log1p(hours / 1 hour)."""
    value = np.asarray(hours, dtype=float)
    if not np.isfinite(value).all() or (value < 0).any():
        raise ValueError('Biological time must be finite and nonnegative')
    phase = np.log1p(value)[..., None]*np.array([1., 2., 4., 8.])
    return np.concatenate([np.sin(phase), np.cos(phase)], axis=-1)


def effective_hours(hours, calibration=24.):
    return calibration*(-np.expm1(-np.asarray(hours)/calibration))/(-np.expm1(-1.))


def time_schedule(delta_hours, training_durations=(24.,), *, kernel='decay'):
    """Observed spans integrate to tau=1; unseen spans keep the 24h CONDITION."""
    delta_hours = float(delta_hours)
    training_durations = np.asarray(training_durations, dtype=float)
    if not np.isfinite(delta_hours) or delta_hours < 0 or not len(training_durations):
        raise ValueError('Invalid biological prediction span')
    if kernel not in ('decay', 'linear'):
        raise ValueError('Unknown preregistered clock')
    if delta_hours == 0:
        return dict(condition_delta_hours=24., tau_end=0., requested_delta_hours=0., kernel=kernel)
    if np.isclose(training_durations, delta_hours).any():
        condition, tau = delta_hours, 1.
    else:
        if not np.isclose(training_durations, 24.).any():
            raise ValueError('Unseen-span rule requires a trained 24-hour reference')
        condition = 24.
        tau = float(effective_hours(delta_hours)/24. if kernel == 'decay' else delta_hours/24.)
    return dict(condition_delta_hours=condition, tau_end=tau,
                requested_delta_hours=delta_hours, kernel=kernel)


def condition_inputs(pack, normalizer, arm, *, time_hours, condition_delta_hours,
                     perturbation_embedding=None, shuffled_population_token=None,
                     cell_permutation=None):
    """Every arm has IDENTICAL keys/shapes, including both zeroed plugin slots."""
    if arm not in ARMS and arm != 'cf_kin_cell_shuf':
        raise ValueError('Unknown CellFlow arm')
    n, d = pack['z'].shape
    group, local = np.zeros(d), np.zeros((n, d))
    if arm in ('cf_kin', 'cf_Ssum', 'cf_directU'):
        kind = {'cf_kin':'kin', 'cf_Ssum':'Ssum', 'cf_directU':'directU'}[arm]
        group = normalizer.apply(kind, pack['population'][kind])
    elif arm == 'cf_kin_shuf':
        if shuffled_population_token is None:
            raise ValueError('Group shuffle must be constructed across eligible source conditions')
        group = np.asarray(shuffled_population_token)
    elif arm in ('cf_kin_cell', 'cf_kin_cell_shuf'):
        local = normalizer.apply('cell_kin', pack['cell_kin'])
        if arm.endswith('_shuf'):
            indices = np.asarray(cell_permutation)
            if indices.shape != (n,) or set(indices.tolist()) != set(range(n)):
                raise ValueError('Cell shuffle must be a permutation of this source cohort')
            local = local[indices]
    perturbation = (np.ones(1) if perturbation_embedding is None else np.asarray(perturbation_embedding))
    if perturbation.ndim == 1:
        perturbation = perturbation[None, :]
    result = dict(perturbation=np.broadcast_to(perturbation, (n, *perturbation.shape)).copy(),
        bio_time=np.broadcast_to(biological_fourier(time_hours), (n,1,8)).copy(),
        bio_delta=np.broadcast_to(biological_fourier(condition_delta_hours), (n,1,8)).copy(),
        population_token=np.broadcast_to(group, (n,1,d)).copy(), cell_token=local[:,None,:].copy())
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite condition input')
    return {k: np.asarray(v, dtype='float32') for k,v in result.items()}


def shuffle_group_tokens(packs, cohorts, normalizer, *, seed):
    """Cross-condition permutation at fixed biological time; min THREE conditions."""
    if len(packs) != len(cohorts) or any(p['source_key'] != c.key for p,c in zip(packs, cohorts)):
        raise ValueError('Source pack/cohort identity mismatch')
    result = np.stack([normalizer.apply('kin', p['population']['kin']) for p in packs])
    original = result.copy(); mapping = np.arange(len(packs)); rng = np.random.default_rng(seed)
    for time in sorted({c.time_hours for c in cohorts}):
        rows = np.array([i for i,c in enumerate(cohorts) if c.time_hours == time])
        conditions = np.array([cohorts[i].condition for i in rows])
        if len(set(conditions)) < 3:
            raise ValueError('Population shuffle needs at least three conditions at each time')
        for _ in range(10000):
            order = rng.permutation(len(rows))
            if np.all(conditions[order] != conditions):
                break
        else:
            raise ValueError('Cannot construct a cross-condition permutation for these unequal groups')
        mapping[rows] = rows[order]; result[rows] = original[rows[order]]
    return result, mapping


def align_conditions_to_source(conditions, source_indices, *, source_size):
    """Keep particle labels with source OT resampling; also accepts JAX arrays."""
    if conditions is None:
        return None
    result = {}
    for key, value in conditions.items():
        if value.shape[0] == source_size:
            result[key] = value[source_indices]
        elif value.shape[0] == 1:
            result[key] = value
        else:
            raise ValueError('Condition batch axis is neither source-size nor shared-size one')
    return result


def plugin_network_kwargs(dimension=50, condition_dim=32):
    """Native per-input MLP blocks; zero-valued inputs remain trainable in all arms."""
    layers = {name: [] for name in ('perturbation', *NOT_POOLED)}
    for name in ('population_token', 'cell_token'):
        layers[name] = [dict(layer_type='mlp', dims=(64, condition_dim), act_last_layer=False)]
    return dict(covariates_not_pooled=NOT_POOLED, layers_before_pool=layers)


def add_cellflow_paths():
    """Project-local overlay only. This does not pip-install or patch official code."""
    paths = [ROOT/'tools/w15a-cellflow-deps',
             Path('/data/yuchang/veloroute_kang_20260914/cellflow_deps'), CELLFLOW_ROOT/'src']
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return paths


def predict_cellflow_tau(solver, x, condition, *, tau_end=1., per_particle=False,
                         rtol=1e-5, atol=1e-5, **kwargs):
    """Integrate native inference field to arbitrary tau WITHOUT editing CellFlow.

    Native OTFM.predict hard-codes t1=1 and shared conditions. This wrapper only
    changes the integration endpoint and optionally vmaps per-source conditions.
    """
    if tau_end < 0 or not np.isfinite(tau_end):
        raise ValueError('Interpolation endpoint must be finite and nonnegative')
    if tau_end == 0:
        return np.asarray(x).copy()
    import diffrax
    import jax
    import jax.numpy as jnp
    if solver.condition_encoder_mode != 'deterministic' or getattr(solver, 'guidance', None) is not None:
        raise ValueError('W15-A wrapper currently requires deterministic unguided conditions')
    condition = {k:jnp.asarray(v) for k,v in condition.items()}
    noise = jnp.zeros((1, solver.vf.condition_embedding_dim))
    vf = solver._base_velocity()
    options = dict(dt0=None, solver=diffrax.Tsit5(),
                   stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol))
    options.update(kwargs)
    def solve(params, state, cond):
        if per_particle:
            cond = {k:v[None] for k,v in cond.items()}
        return diffrax.diffeqsolve(diffrax.ODETerm(vf), t0=0., t1=float(tau_end), y0=state,
            args=(params, cond, noise), **options).ys[0]
    fn = jax.jit(jax.vmap(solve, in_axes=(None, 0, 0 if per_particle else None)))
    return np.asarray(fn(solver.vf_state_inference.params, jnp.asarray(x), condition))


def save_source_manifest(path, packs, normalizer, *, transform):
    """Small reproducibility record, with no destination values or hashes needed."""
    result = dict(source_packs=[dict(key=p['source_key'], source_hash=p['source_hash'],
        source_ids_hash=array_hash(p['source_ids'])) for p in packs],
        token_normalizer_hash=normalizer.fingerprint(), source_transform_hash=transform.fingerprint(),
        target_input=False, biological_fourier_frequencies=[1,2,4,8])
    Path(path).write_text(json.dumps(result, indent=2)+'\n')
    return result
