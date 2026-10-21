"""L3 double-stream mixture scaffold; no official CellFlow backbone is claimed.

The replaceable backbone contract is ``backbone(x_tau, tau, condition) -> h``.
Only the source state and encoded kinetic input meet at the router.  OT indices
are produced externally from S-state geometry and then reindex source attributes.
Real-data training is intentionally gated on the L2 cf_static freeze.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


ARMS = ("moe_static", "moe_kin", "moe_kin_shuf", "moe_directU")


@dataclass(frozen=True)
class MoEConfig:
    dimensions: int = 50
    experts: int = 8
    expert_width: int = 128
    kinetic_width: int = 64
    kinetic_embedding: int = 16
    router_width: int = 64
    balance_coefficient: float = 0.01


class KineticEncoder(nn.Module):
    """E_v has a single argument: kappa, never raw source x or conditions."""

    width: int = 64
    embedding: int = 16

    @nn.compact
    def __call__(self, kappa):
        return nn.Dense(self.embedding, name="embedding")(
            nn.silu(nn.Dense(self.width, name="hidden")(kappa)))


class Router(nn.Module):
    experts: int = 8
    width: int = 64

    @nn.compact
    def __call__(self, source_x, kinetic_embedding):
        inputs = jnp.concatenate((source_x, kinetic_embedding), axis=-1)
        logits = nn.Dense(self.experts, name="logits")(
            nn.silu(nn.Dense(self.width, name="hidden")(inputs)))
        return jax.nn.softmax(logits, axis=-1)


class ExpertHead(nn.Module):
    dimensions: int = 50
    width: int = 128

    @nn.compact
    def __call__(self, hidden):
        return nn.Dense(self.dimensions, name="velocity")(
            nn.silu(nn.Dense(self.width, name="hidden")(hidden)))


def _repeat_batch(tree, repeats):
    return jax.tree_util.tree_map(lambda value: jnp.repeat(value, repeats, axis=0), tree)


def _batch_tau(tau, batch):
    tau = jnp.asarray(tau)
    if tau.ndim == 0:
        return jnp.full((batch, 1), tau)
    if tau.shape == (batch,):
        return tau[:, None]
    if tau.shape != (batch, 1):
        raise ValueError("tau must be scalar, [batch], or [batch, 1]")
    return tau


def state_condition_only(condition):
    """Named L2 token slots cannot silently leak kinetics into L3 experts.

    A production official-backbone wrapper must additionally construct its input
    from the explicit perturbation/time whitelist. Anonymous dense conditions in
    scaffold tests do not establish data provenance on real experiments.
    """
    forbidden = {"population_token", "cell_token", "kappa", "kinetic_embedding",
                 "velocity", "directU", "direct_U", "unspliced", "U"}
    if isinstance(condition, Mapping) and forbidden.intersection(condition):
        raise ValueError("Kinetic/U-derived conditions are forbidden in the L3 state stream")
    return condition


class KineticMoE(nn.Module):
    """S-only state stream plus independent kappa-to-router stream.

    ``backbone`` is an injected Flax module, not an imitation CellFlow model.
    The official hidden-feature adapter is still a separate integration task.
    Conditions passed here must contain only perturbation and biological time;
    the L2 population/cell kinetic token inputs must not be forwarded.
    """

    backbone: nn.Module
    config: MoEConfig = MoEConfig()

    def setup(self):
        c = self.config
        self.kinetic_encoder = KineticEncoder(c.kinetic_width, c.kinetic_embedding)
        self.router = Router(c.experts, c.router_width)
        self.expert_heads = tuple(ExpertHead(c.dimensions, c.expert_width,
                                           name=f"expert_{i}") for i in range(c.experts))

    def encode_kinetics(self, kappa):
        return self.kinetic_encoder(kappa)

    def route(self, source_x, kappa):
        return self.router(source_x, self.kinetic_encoder(kappa))

    def velocities(self, x_tau, tau, condition):
        """No kappa/m/U argument can enter the state stream."""
        state_condition_only(condition)
        hidden = self.backbone(x_tau, _batch_tau(tau, len(x_tau)), condition)
        return jnp.stack([head(hidden) for head in self.expert_heads], axis=1)

    def candidate_velocities(self, candidate_states, tau, condition):
        """Each expert evolves its own state, not a mixed velocity trajectory."""
        state_condition_only(condition)
        batch, experts, dimensions = candidate_states.shape
        if (experts, dimensions) != (self.config.experts, self.config.dimensions):
            raise ValueError("Candidate states differ from registered K/dimensions")
        hidden = self.backbone(candidate_states.reshape(batch * experts, dimensions),
                               _repeat_batch(_batch_tau(tau, batch), experts),
                               _repeat_batch(condition, experts))
        hidden = hidden.reshape(batch, experts, -1)
        return jnp.stack([head(hidden[:, k]) for k, head in enumerate(self.expert_heads)], axis=1)

    def __call__(self, source_x, kappa, x_tau, tau, condition):
        return dict(q=self.route(source_x, kappa),
                    velocities=self.velocities(x_tau, tau, condition))


def arm_input(arm, kappa, *, direct_u=None, permutation=None):
    """All arms use the same module/parameters; inputs alone differ.

    Apply a source-cohort permutation BEFORE external OT source reindexing.
    Upstream constructs direct_u without gamma and freezes training-only scaling.
    This is HOST-side cohort preprocessing; permutation validation is deliberately
    not jittable. Compiled train steps consume its already prepared array.
    """
    if arm not in ARMS:
        raise ValueError(f"Unregistered L3 arm: {arm}")
    value = jnp.asarray(kappa)
    if value.ndim != 2:
        raise ValueError("Kinetic inputs must be [source, dimensions]")
    if arm == "moe_static":
        return jnp.zeros_like(value)
    if arm == "moe_kin":
        return value
    if arm == "moe_directU":
        if direct_u is None or np.shape(direct_u) != value.shape:
            raise ValueError("direct_U must match the kinetic input shape")
        return jnp.asarray(direct_u)
    if permutation is None:
        raise ValueError("Shuffled arm requires an explicit source-cohort permutation")
    permutation = np.asarray(permutation)
    if permutation.ndim != 1 or not np.array_equal(np.sort(permutation), np.arange(len(value))):
        raise ValueError("Shuffled indices must be a full source-cohort permutation")
    return value[permutation]


class PairedBatch(NamedTuple):
    source_x: Any
    kinetic_input: Any
    target_x: Any
    condition: Any = None


def align_ot_indices(source_x, kinetic_input, target_x, source_indices, target_indices,
                     *, condition=None):
    """Carry kappa along OT's source indices, including duplicated source draws.

    This helper does not compute an OT cost and cannot access router parameters.
    The official CellFlow adapter owns state-only OT and supplies both indices.
    """
    source_x, kinetic_input, target_x = map(jnp.asarray, (source_x, kinetic_input, target_x))
    if len(source_x) != len(kinetic_input):
        raise ValueError("Source states and kinetic attributes must be aligned")
    si, ti = np.asarray(source_indices), np.asarray(target_indices)
    if si.ndim != 1 or si.shape != ti.shape or len(si) == 0:
        raise ValueError("OT indices must be equally-sized nonempty vectors")
    if not np.issubdtype(si.dtype, np.integer) or not np.issubdtype(ti.dtype, np.integer):
        raise ValueError("OT indices must be integers")
    if si.min() < 0 or si.max() >= len(source_x) or ti.min() < 0 or ti.max() >= len(target_x):
        raise ValueError("OT index outside its source/target batch")
    return align_ot_indices_jit(source_x, kinetic_input, target_x, si, ti, condition=condition)


def align_ot_indices_jit(source_x, kinetic_input, target_x, source_indices, target_indices,
                         *, condition=None):
    """Pure-JAX pairing core for compiled official OT training steps.

    Index *values* must already be valid outputs of the state-only OT sampler;
    host-side data/index auditing uses align_ot_indices. JAX indexing can clamp
    out-of-range values, so this core is not an index-validation substitute.
    Shape/dtype checks are compile-time only; no tracer is converted to NumPy.
    """
    source_x, kinetic_input, target_x, si, ti = map(jnp.asarray,
        (source_x, kinetic_input, target_x, source_indices, target_indices))
    if len(source_x) != len(kinetic_input):
        raise ValueError("Source states and kinetic attributes must be aligned")
    if si.ndim != 1 or si.shape != ti.shape or not si.shape[0]:
        raise ValueError("OT indices must be equally-sized nonempty vectors")
    if not jnp.issubdtype(si.dtype, jnp.integer) or not jnp.issubdtype(ti.dtype, jnp.integer):
        raise ValueError("OT indices must be integers")
    aligned_condition = None
    if condition is not None:
        state_condition_only(condition)
        if any(leaf.shape[0] != len(source_x) for leaf in jax.tree_util.tree_leaves(condition)):
            raise ValueError("Every condition leaf must align with the source batch")
        aligned_condition = jax.tree_util.tree_map(lambda value: jnp.asarray(value)[si], condition)
    return PairedBatch(source_x[si], kinetic_input[si], target_x[ti], aligned_condition)


def flow_matching_loss(q, expert_velocities, target_velocity, balance_coefficient=0.01):
    """Exact E_k squared error, NOT squared error of the mixed velocity."""
    q, expert_velocities, target_velocity = map(jnp.asarray, (q, expert_velocities, target_velocity))
    if expert_velocities.shape[:2] != q.shape or target_velocity.shape != expert_velocities.shape[::2]:
        raise ValueError("Expected q[B,K], velocities[B,K,D], target[B,D]")
    per_expert = jnp.sum(jnp.square(expert_velocities - target_velocity[:, None]), axis=-1)
    matching = jnp.mean(jnp.sum(q * per_expert, axis=1))
    average_q = jnp.mean(q, axis=0)
    balance = jnp.sum(jax.scipy.special.xlogy(average_q, average_q * q.shape[1]))
    return matching + balance_coefficient * balance, dict(matching=matching, balance=balance,
        effective_experts=jnp.exp(-jnp.sum(jax.scipy.special.xlogy(average_q, average_q))))


def paired_flow_matching_loss(model, variables, batch, tau, *, condition=None):
    """Trainable complete L3 objective on an externally OT-aligned mini-batch.

    tau is the FM interpolation clock, never the biological time covariate. The
    source state/kappa are fixed while only x_tau enters the expert state stream.
    """
    if condition is None:
        condition = batch.condition
    if condition is None:
        raise ValueError("Source-aligned state-only conditions are required")
    tau = _batch_tau(tau, len(batch.source_x))
    target_velocity = batch.target_x - batch.source_x
    x_tau = batch.source_x + tau * target_velocity
    result = model.apply(variables, batch.source_x, batch.kinetic_input, x_tau, tau, condition)
    return flow_matching_loss(result["q"], result["velocities"], target_velocity,
                               model.config.balance_coefficient)


def integrate_mixture(model, variables, source_x, kinetic_input, condition, *,
                      tau_end=1.0, steps=32, q_intervention="none"):
    """RK4 candidates with one source-side route call and frozen route weights.

    This solver is a tested scaffold.  Official L2 solver selection/configuration
    must be inherited before real-data L3 training; no ODE accuracy is assumed here.
    """
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("RK4 steps must be a positive integer")
    if q_intervention not in ("none", "cohort_q", "uniform_q"):
        raise ValueError("Unknown frozen routing intervention")
    source_x = jnp.asarray(source_x)
    q = model.apply(variables, source_x, kinetic_input, method=model.route)
    if q_intervention == "cohort_q":
        q = jnp.broadcast_to(q.mean(axis=0), q.shape)
    elif q_intervention == "uniform_q":
        q = jnp.full_like(q, 1 / q.shape[1])
    candidates = jnp.repeat(source_x[:, None, :], model.config.experts, axis=1)
    dt = jnp.asarray(tau_end) / steps

    def velocity(x, tau):
        return model.apply(variables, x, tau, condition, method=model.candidate_velocities)

    def advance(x, step):
        tau = step * dt
        k1 = velocity(x, tau)
        k2 = velocity(x + dt * k1 / 2, tau + dt / 2)
        k3 = velocity(x + dt * k2 / 2, tau + dt / 2)
        k4 = velocity(x + dt * k3, tau + dt)
        updated = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return updated, q

    candidates, q_history = jax.lax.scan(advance, candidates, jnp.arange(steps))
    return dict(candidates=candidates, q=q, q_history=q_history, routing_evaluations=1)


def weighted_energy(candidates, q, target):
    """Exact empirical-mixture V-statistic, including same-source k != l terms.

    It is NOT the expectation of finite-N one-expert-draw V-statistic energy.
    Independent expert draws for the two within-source factors give that equality.
    This CPU diagnostic materializes distances and is not the production scorer.
    """
    candidates, q, target = map(lambda value: np.asarray(value, dtype=np.float64),
                                (candidates, q, target))
    if candidates.ndim != 3 or q.shape != candidates.shape[:2] or target.ndim != 2:
        raise ValueError("Invalid weighted empirical-mixture dimensions")
    if target.shape[1] != candidates.shape[-1] or not len(target) or not len(q):
        raise ValueError("Empty or incompatible target distribution")
    if not np.isfinite(candidates).all() or not np.isfinite(target).all() or not np.isfinite(q).all():
        raise ValueError("Nonfinite distributions")
    if (q < 0).any() or not np.allclose(q.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Router weights must be row-stochastic")
    flat = candidates.reshape(-1, candidates.shape[-1])
    weights = q.ravel() / len(q)
    distance = lambda x, y: np.linalg.norm(x[:, None] - y[None], axis=-1)
    cross = weights @ distance(flat, target).mean(axis=1)
    within = weights @ distance(flat, flat) @ weights
    target_within = distance(target, target).mean()
    return float(2 * cross - within - target_within)


def sampled_energy(candidates, q, target, rng, *, independent_within=True):
    """Unbiased expert-assignment Monte Carlo when independent_within=True.

    Source/target empirical supports remain fixed; this is not biological sampling.
    A single unbiased draw may be negative. The naive False branch is retained
    only to demonstrate its finite-N positive diagonal bias in the unit tests.
    """
    candidates, q, target = map(np.asarray, (candidates, q, target))
    n, k = q.shape
    choose = lambda: np.array([rng.choice(k, p=p) for p in q])
    a = candidates[np.arange(n), choose()]
    b = candidates[np.arange(n), choose()] if independent_within else a
    distance = lambda x, y: np.linalg.norm(x[:, None] - y[None], axis=-1)
    cross = (distance(a, target).mean() + distance(b, target).mean()) / 2
    return float(2 * cross - distance(a, b).mean() - distance(target, target).mean())


def require_l2_freeze(configuration, *, configuration_path=None, adapter_path=None):
    """Verify independent frozen L2 scientific config AND official adapter bytes.

    The freeze record is separate from the scientific config to avoid a circular
    self-hash. Actual L2 freezing/publishing still owns checking its git commit.
    Both file paths are explicit so callers cannot bypass the gate with booleans.
    """
    required = ("cf_static_frozen", "freeze_commit", "configuration_sha256",
                "official_backbone_adapter", "official_backbone_adapter_sha256",
                "backbone", "optimizer", "solver")
    if not configuration.get("cf_static_frozen") or any(not configuration.get(k) for k in required):
        raise ValueError("Real-data L3 awaits the frozen L2 cf_static configuration and official hidden adapter")
    if configuration_path is None or adapter_path is None:
        raise ValueError("Frozen L2 configuration and adapter files are required for hash verification")
    for path, field in ((configuration_path, "configuration_sha256"),
                        (adapter_path, "official_backbone_adapter_sha256")):
        path = Path(path)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != configuration[field]:
            raise ValueError(f"Frozen L2 file missing or SHA256 changed: {path}")
    saved = l2_scientific_sections(json.loads(Path(configuration_path).read_text()))
    if any(saved.get(key) != configuration[key] for key in ("backbone", "optimizer", "solver")):
        raise ValueError("L2 freeze metadata disagrees with the hashed scientific configuration")
    return configuration


def l2_scientific_sections(saved):
    """Read L2's actual flat scientific config without rewriting trained files.

    The current L2 selector writes backbone/optimizer/solver only in its separate
    freeze_record. This read-only mapping matches those already recorded fields.
    Nested test/future records are accepted directly, with the same hash check.
    """
    if all(key in saved for key in ("backbone", "optimizer", "solver")):
        return {key: saved[key] for key in ("backbone", "optimizer", "solver")}
    try:
        return dict(backbone={key: saved[key] for key in
            ("width", "depth", "condition_dim", "time_width", "time_depth", "time_freqs", "pooling")},
            optimizer={key: saved[key] for key in ("learning_rate", "steps", "batch_size")},
            solver=dict(name="official_OTFlowMatching", probability_path="ConstantNoiseFlow_sigma0",
                matching="official_balanced_Sinkhorn_mean_scale_cost", epsilon=saved["ot_epsilon"],
                max_iterations=saved["ot_max_iterations"], prediction="Tsit5_PID_rtol_atol_1e-5",
                primary_time="decay_h(delta)/h(24)", sensitivity="linear_delta/24"))
    except KeyError as error:
        raise ValueError("Incomplete flat L2 scientific configuration") from error


def kinetic_sensitivity(model, variables, source_x, original_kappa, modified_kappa):
    """Inference-only U-audit summaries after upstream recomputes kappa.

    The upstream adapter owns actual raw-U replacement and S-fixed recomputation;
    this function does not pretend that arbitrary kappa corruption is U removal.
    Endpoint-energy changes are scored separately on the frozen full predictions.
    """
    original = model.apply(variables, original_kappa, method=model.encode_kinetics)
    modified = model.apply(variables, modified_kappa, method=model.encode_kinetics)
    numerator = jnp.sum(original * modified, axis=-1)
    denominator = jnp.linalg.norm(original, axis=-1) * jnp.linalg.norm(modified, axis=-1)
    cosine = jnp.where(denominator > 0, numerator / jnp.maximum(denominator, 1e-12), jnp.nan)
    q_original = model.apply(variables, source_x, original_kappa, method=model.route)
    q_modified = model.apply(variables, source_x, modified_kappa, method=model.route)
    tv = jnp.mean(jnp.abs(q_original - q_modified).sum(axis=-1) / 2)
    return dict(embedding_median_cosine=jnp.nanmedian(cosine),
                mean_route_tv=tv, u_insensitive=tv < 0.01)
