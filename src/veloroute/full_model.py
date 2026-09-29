"""Full deployable VeloRoute. Delayed teachers and future cells are NOT members."""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .full_components import (DynamicsEncoder, FrozenReferenceField, IntrinsicField, ModeController,
                              ReliabilityGate, SetConditionEncoder, StateEncoder, sparse_mode_probabilities)
from .model import ModelConfig, SharedExpertField, SourceRouter


@dataclass(frozen=True)
class FullConfig:
    state_dim: int = 50
    esm_dim: int = 2560
    representation_dim: int = 64
    condition_dim: int = 256
    encoder_hidden: int = 256
    hidden_dim: int = 768
    router_hidden_dim: int = 256
    time_hidden_dim: int = 128
    residual_blocks: int = 6
    expert_rank: int = 64
    max_experts: int = 8
    initial_active: int = 1
    top_k: int = 2
    attention_heads: int = 8
    attention_head_dim: int = 64
    reference_neighbors: int = 30
    prune_patience: int = 3
    minimum_mode_usage: float = .01
    delta_u_given_s: float = 0.
    support_start: float = 4.
    support_end: float = 5.
    maximum_coverage_distance: float = 5.
    initial_noise: float = .01
    use_state_encoder: bool = True
    use_dynamics: bool = True
    # Legacy checkpoints retain frozen fallback semantics. New static-router
    # runs explicitly select the trained field while disabling velocity input.
    static_uses_trained_field: bool = False
    use_intrinsic: bool = True
    use_gate: bool = True
    use_noise: bool = True
    use_adaptive_modes: bool = True
    dynamics_backend: str = 'baseline'
    joint_dynamics: bool = False
    gfg_genes: int = 2000
    gfg_hidden: tuple = (256, 512, 512, 256)
    gfg_gene_dim: int = 8
    gfg_codes: int = 32

    def __post_init__(self):
        if self.dynamics_backend not in {'baseline', 'gfg'} or (self.joint_dynamics and self.dynamics_backend != 'gfg'):
            raise ValueError('Joint dynamics requires the GFG backend')
        for name in ('state_dim', 'esm_dim', 'representation_dim', 'condition_dim', 'encoder_hidden',
                     'hidden_dim', 'router_hidden_dim', 'time_hidden_dim', 'residual_blocks', 'expert_rank',
                     'max_experts', 'initial_active', 'top_k', 'attention_heads', 'attention_head_dim',
                     'reference_neighbors', 'prune_patience'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'Invalid dimension/count: {name}')
        if not self.initial_active <= self.max_experts or not self.top_k <= self.max_experts:
            raise ValueError('Active/top-k mode count exceeds maximum')
        for name in ('minimum_mode_usage', 'delta_u_given_s', 'support_start', 'support_end',
                     'maximum_coverage_distance', 'initial_noise'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f'Nonfinite configuration: {name}')
        if self.support_end <= self.support_start or self.initial_noise <= 0 or self.maximum_coverage_distance <= 0:
            raise ValueError('Invalid time support, noise or coverage policy')


@dataclass
class SourceContext:
    z0: torch.Tensor
    representation: torch.Tensor
    velocity: torch.Tensor
    rho: torch.Tensor
    condition: torch.Tensor
    static_condition: torch.Tensor
    probabilities: torch.Tensor
    gate_features: torch.Tensor
    gate_valid: torch.Tensor


@dataclass
class FullPrediction:
    endpoint: torch.Tensor
    modes: torch.Tensor
    probabilities: torch.Tensor
    gate: torch.Tensor
    times: tuple[float, ...]
    trajectory: tuple[torch.Tensor, ...]
    contributions: dict[str, torch.Tensor]


class FullVeloRoute(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or FullConfig()
        c = self.config
        self.state_encoder = StateEncoder(c.state_dim, c.encoder_hidden)
        if c.dynamics_backend == 'gfg':
            from .gfg import GFGDynamics
            self.dynamics = GFGDynamics(c)
        else:
            self.dynamics = DynamicsEncoder(c.state_dim, c.representation_dim, c.encoder_hidden)
        self.condition_encoder = SetConditionEncoder(c.esm_dim, c.condition_dim, c.attention_heads, c.attention_head_dim)
        field_config = ModelConfig(state_dim=c.state_dim, velocity_dim=c.state_dim, condition_dim=c.condition_dim,
            hidden_dim=c.hidden_dim, router_hidden_dim=c.router_hidden_dim, time_hidden_dim=c.time_hidden_dim,
            residual_blocks=c.residual_blocks, expert_rank=c.expert_rank, n_experts=c.max_experts, top_k=c.top_k)
        self.field = SharedExpertField(field_config)
        self.router = SourceRouter(ModelConfig(**{**asdict(field_config), 'velocity_dim': c.representation_dim}))
        self.intrinsic = IntrinsicField(c.state_dim)
        self.gate = ReliabilityGate()
        self.reference = FrozenReferenceField(c.state_dim, c.reference_neighbors)
        initial = c.initial_active if c.use_adaptive_modes else c.max_experts
        self.mode_controller = ModeController(c.max_experts, initial, c.prune_patience, c.minimum_mode_usage)
        self.noise_raw = nn.Parameter(torch.full((c.max_experts, c.state_dim), math.log(math.expm1(c.initial_noise))))
        # A real frozen fallback needs its own field AND condition encoder snapshots.
        self.static_field = copy.deepcopy(self.field).requires_grad_(False)
        self.static_condition_encoder = copy.deepcopy(self.condition_encoder).requires_grad_(False)
        self.register_buffer('fallback_ready', torch.tensor(False))
        self.register_buffer('condition_memory', torch.empty(0, c.condition_dim))

    @property
    def active_modes(self):
        return self.mode_controller.active

    def static_rate(self, z, t, condition):
        # The frozen snapshot uses its base field only: no expert or intrinsic term.
        field = self.static_field
        time = torch.as_tensor(t, device=z.device, dtype=z.dtype)
        if time.ndim == 0:
            time = time.expand(len(z))
        h = field.blocks(field.input(torch.cat((z, field.time(time), condition), -1)))
        return field.base(h)

    @torch.no_grad()
    def snapshot_static(self):
        self.static_field.load_state_dict(self.field.state_dict())
        self.static_condition_encoder.load_state_dict(self.condition_encoder.state_dict())
        self.static_field.requires_grad_(False)
        self.static_condition_encoder.requires_grad_(False)
        self.fallback_ready.fill_(True)

    def freeze_encoders(self):
        self.state_encoder.requires_grad_(False)
        if not self.config.joint_dynamics:
            self.dynamics.requires_grad_(False)
        for parameter in list(self.state_encoder.parameters())+list(self.dynamics.parameters()):
            parameter.grad = None

    def encode_state(self, spliced):
        return self.state_encoder(spliced) if self.config.use_state_encoder else spliced

    @torch.no_grad()
    def refresh_reference(self, spliced, unspliced, tokens, mask, *, cell_ids, scope):
        # Called only with declared training IDs. No condition reaches dynamics.
        z = self.encode_state(spliced)
        if self.config.use_dynamics:
            batch = getattr(self.dynamics, 'reference_batch_size', len(spliced))
            v = torch.cat([self.dynamics(spliced[i:i+batch], unspliced[i:i+batch])[1]
                           for i in range(0, len(spliced), batch)])
        else:
            # Static arms still need a reference-field tensor with the same
            # shape as the PCA state, but must not touch the dynamics module.
            v = torch.zeros_like(z)
        self.reference.fit(z, v, cell_ids=cell_ids, scope=scope)
        e = self.condition_encoder(tokens, mask)
        self.condition_memory = torch.unique(e.detach(), dim=0)

    def make_context(self, spliced, unspliced, tokens, mask, *, velocity_override=None, background=None, time_start=4.):
        with torch.no_grad():
            z = self.encode_state(spliced)
            e_static = self.static_condition_encoder(tokens, mask)
        if self.config.use_dynamics:
            if self.config.joint_dynamics and torch.is_grad_enabled():
                r, v, rho = self.dynamics(spliced, unspliced)
            else:
                with torch.no_grad():
                    r, v, rho = self.dynamics(spliced, unspliced)
        else:
            # Strict static arm: do not even invoke the S/U encoder.
            r = spliced.new_zeros((len(spliced), self.config.representation_dim))
            v = spliced.new_zeros((len(spliced), self.config.state_dim))
            rho = spliced.new_ones((len(spliced), 1))*10
        if not self.config.use_dynamics:
            r, v, rho = torch.zeros_like(r), torch.zeros_like(v), torch.ones_like(rho)*10
        if background is not None and self.config.use_dynamics:
            r = r-background[0]
            v = v-background[1]
        if velocity_override is not None:
            if velocity_override.shape != v.shape or not torch.isfinite(velocity_override).all():
                raise ValueError('Invalid source-side corrupted velocity')
            v = velocity_override.detach()
        route_r = r if self.config.joint_dynamics else r.detach()
        e = self.condition_encoder(tokens.detach(), mask)
        logits = self.router.logits(z.detach(), route_r, e.detach())
        q = sparse_mode_probabilities(logits, self.active_modes, self.config.top_k)
        with torch.no_grad():
            ref, distance = self.reference(z)
            candidates = self.field(z, time_start, e.detach())
            start_field = (q.detach()[..., None]*candidates).sum(1)
            cosine = F.cosine_similarity(start_field, ref, dim=-1)
            if len(self.condition_memory):
                similarity = F.normalize(e.detach(), dim=-1) @ F.normalize(self.condition_memory, dim=-1).T
                confidence = similarity.max(-1).values.clamp(0, 1)
            else:
                confidence = rho.new_zeros(len(rho))
            # Dynamics uncertainty is emitted as [N, 1] by both the baseline
            # and GFG heads; gate features are one scalar per cell.
            rho_scalar = rho.squeeze(-1)
            features = torch.stack((rho_scalar, cosine,
                                    rho_scalar.new_full(rho_scalar.shape, self.config.delta_u_given_s),
                                    -distance, v.norm(dim=-1), confidence), -1)
            valid = ((v.norm(dim=-1) > 1e-8) & (ref.norm(dim=-1) > 1e-8)
                     & (distance <= self.config.maximum_coverage_distance) & (confidence > 0))
        return SourceContext(z.detach(), route_r, v.detach(), rho.detach(), e, e_static.detach(), q,
                             features.detach(), valid)

    def gate_value(self, context, time):
        if not self.config.use_dynamics:
            return context.rho.new_full((len(context.rho),), float(self.config.static_uses_trained_field))
        if not self.config.use_gate:
            return context.rho.new_ones(len(context.rho))
        t = torch.as_tensor(time, device=context.rho.device, dtype=context.rho.dtype)
        outside = (self.config.support_start-t).clamp_min(0)+(t-self.config.support_end).clamp_min(0)
        features = context.gate_features.clone()
        features[:, 0] += outside
        # Deterministic attenuation enforces increasing caution outside support,
        # regardless of the learned coefficient on uncertainty.
        return self.gate(features, context.gate_valid)*torch.exp(-outside)

    def noise_scale(self):
        return F.softplus(self.noise_raw).clamp_min(1e-5)

    def rate_parts(self, z, time, context, modes, *, epsilon=None, force_gate=None):
        if not bool(self.fallback_ready):
            raise ValueError('Snapshot the trained static fallback before gated prediction')
        g = self.gate_value(context, time).detach()
        if force_gate is not None:
            if force_gate not in (0., 1.):
                raise ValueError('Only explicit all-static/all-dynamic ablation is supported')
            g = torch.full_like(g, force_gate)
        dynamic = self.field.selected(z, time, context.condition, modes)
        intrinsic = self.intrinsic(z, context.z0, context.velocity) if self.config.use_intrinsic else torch.zeros_like(z)
        static = self.static_rate(z, time, context.static_condition)
        noise = torch.zeros_like(z)
        if epsilon is not None and self.config.use_noise:
            noise = self.noise_scale()[modes]*epsilon
        return {'static': (1-g[:, None])*static, 'intrinsic': g[:, None]*intrinsic,
                'perturbation': g[:, None]*dynamic, 'within_mode_noise': g[:, None]*noise}

    @torch.no_grad()
    def predict(self, spliced, unspliced, tokens, mask, *, t0=4., t1=5., max_step=.25,
                generator=None, modes=None, stochastic=True, return_trajectory=False, snapshots=(),
                force_gate=None, background=None):
        context = self.make_context(spliced, unspliced, tokens, mask, background=background, time_start=t0)
        return self.integrate_context(context, t0=t0, t1=t1, max_step=max_step, generator=generator,
            modes=modes, stochastic=stochastic, return_trajectory=return_trajectory,
            snapshots=snapshots, force_gate=force_gate)

    def integrate_context(self, context, *, t0=4., t1=5., max_step=.25, generator=None,
                          modes=None, stochastic=True, return_trajectory=False, snapshots=(), force_gate=None):
        """Differentiable RK4; inference uses predict's no-grad wrapper.

        Reuse one context across experts to keep router/GFG and field graphs live.
        Gate gradients remain isolated for corruption calibration.
        """
        if not len(context.z0) or not all(math.isfinite(x) for x in (t0, t1, max_step)) or t1 <= t0 or max_step <= 0:
            raise ValueError('Need finite increasing times and a bounded positive step')
        count = math.ceil((t1-t0)/max_step)
        if count > 10000 or any(not math.isfinite(t) or not t0 <= t <= t1 for t in snapshots):
            raise ValueError('Invalid intermediate times or excessive step budget')
        if modes is None:
            modes = torch.multinomial(context.probabilities, 1, generator=generator).squeeze(-1)
        if modes.shape != (len(context.z0),) or modes.dtype != torch.long:
            raise ValueError('Need one locked integer mode per source cell')
        if (modes < 0).any() or (modes >= self.config.max_experts).any() or not self.active_modes[modes].all():
            raise ValueError('Cannot sample inactive/retired experts')
        epsilon = None
        if stochastic and self.config.use_noise:
            epsilon = torch.randn(context.z0.shape, device=context.z0.device, dtype=context.z0.dtype, generator=generator)
        times = tuple(sorted(set([t0+i*(t1-t0)/count for i in range(count+1)] + list(snapshots))))
        z = context.z0.clone()
        trajectory = [z.clone()] if return_trajectory else []
        totals = {key: torch.zeros_like(z) for key in ('static', 'intrinsic', 'perturbation', 'within_mode_noise')}
        for left, right in zip(times, times[1:]):
            dt = right-left
            def evaluate(state, time):
                parts = self.rate_parts(state, time, context, modes, epsilon=epsilon, force_gate=force_gate)
                return sum(parts.values()), parts
            a, pa = evaluate(z, left)
            b, pb = evaluate(z+dt*a/2, left+dt/2)
            c, pc = evaluate(z+dt*b/2, left+dt/2)
            d, pd = evaluate(z+dt*c, right)
            z = z + dt*(a+2*b+2*c+d)/6
            for key in totals:
                totals[key] += dt*(pa[key]+2*pb[key]+2*pc[key]+pd[key])/6
            if not torch.isfinite(z).all():
                raise FloatingPointError('Full ODE generated nonfinite states')
            if return_trajectory:
                trajectory.append(z.clone())
        reported_gate = self.gate_value(context, t0)
        if force_gate is not None:
            reported_gate = torch.full_like(reported_gate, force_gate)
        return FullPrediction(z, modes, context.probabilities, reported_gate, times,
                              tuple(trajectory), totals)

    def freeze_retired_experts(self):
        for k, expert in enumerate(self.field.experts):
            if self.mode_controller.retired[k]:
                expert.requires_grad_(False)
                for parameter in expert.parameters():
                    parameter.grad = None


def save_full_checkpoint(path, model, *, metadata):
    with Path(path).open('xb') as stream:
        torch.save({'format': 'veloroute_full_v1', 'config': asdict(model.config), 'model': model.state_dict(),
                    'metadata': metadata, 'reference_fit_ids_hash': model.reference.fit_ids_hash,
                    'dynamics_fit_ids_hash': getattr(model.dynamics, 'fit_ids_hash', None)}, stream)


def load_full_checkpoint(path, *, map_location='cpu'):
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get('format') != 'veloroute_full_v1':
        raise ValueError('Not a full-model checkpoint')
    model = FullVeloRoute(FullConfig(**payload['config'])).to(map_location)
    state = payload['model']
    for name in ('positions', 'velocities'):
        setattr(model.reference, name, torch.empty_like(state[f'reference.{name}']))
    model.condition_memory = torch.empty_like(state['condition_memory'])
    model.load_state_dict(state, strict=True)
    model.reference.fit_ids_hash = payload['reference_fit_ids_hash']
    if hasattr(model.dynamics, 'fit_ids_hash'):
        model.dynamics.fit_ids_hash = payload.get('dynamics_fit_ids_hash')
    model.freeze_encoders()
    model.freeze_retired_experts()
    model.eval()
    return model, payload['metadata']
