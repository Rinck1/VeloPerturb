"""Minimal velocity-conditioned router and shared flow experts.

Velocity is an INFERENCE input to the router only, not an additive vector field.
Input states/velocities are already in audited frozen latent coordinates.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    state_dim: int = 50
    velocity_dim: int = 50
    condition_dim: int = 256
    hidden_dim: int = 256
    router_hidden_dim: int = 256
    time_hidden_dim: int = 128
    residual_blocks: int = 3
    expert_rank: int = 32
    n_experts: int = 2
    top_k: int = 2
    use_velocity: bool = True

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name != "use_velocity" and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if self.top_k > self.n_experts:
            raise ValueError("top_k cannot exceed expert count")


def _check_matrix(x, n, d, name):
    if x.ndim != 2 or x.shape != (n, d) or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError(f"{name} must be finite floating tensor of shape {(n, d)}")


class SourceRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.state_dim + config.velocity_dim + config.condition_dim
        self.network = nn.Sequential(nn.Linear(dim, config.router_hidden_dim), nn.SiLU(),
                                     nn.Linear(config.router_hidden_dim, config.router_hidden_dim), nn.SiLU(),
                                     nn.Linear(config.router_hidden_dim, config.n_experts))

    def logits(self, z0, velocity0, condition):
        cfg, n = self.config, len(z0)
        _check_matrix(z0, n, cfg.state_dim, "source state")
        _check_matrix(velocity0, n, cfg.velocity_dim, "source velocity")
        _check_matrix(condition, n, cfg.condition_dim, "condition embedding")
        velocity = velocity0 if cfg.use_velocity else torch.zeros_like(velocity0)
        return self.network(torch.cat((z0, velocity, condition), dim=-1))

    def forward(self, z0, velocity0, condition):
        logits = self.logits(z0, velocity0, condition)
        selected = logits.topk(self.config.top_k, dim=-1).indices
        sparse_logits = torch.full_like(logits, -torch.inf).scatter(1, selected, logits.gather(1, selected))
        return sparse_logits.softmax(-1)


class PhysicalTimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        # Four physical periods in days, each with eight harmonics: 32 sine/cosine pairs.
        frequencies = torch.tensor([h / period for period in (1., 2., 4., 8.) for h in range(1, 9)])
        self.register_buffer("frequencies", frequencies)
        self.projection = nn.Sequential(nn.Linear(64, width), nn.SiLU())

    def forward(self, t):
        angles = 2 * math.pi * t[:, None] * self.frequencies[None, :]
        return self.projection(torch.cat((angles.sin(), angles.cos()), -1))


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.block = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                                   nn.Linear(width, width))

    def forward(self, x):
        return x + self.block(x) / math.sqrt(2.)


class LowRankExpert(nn.Module):
    def __init__(self, width, rank, output_dim):
        super().__init__()
        self.down = nn.Linear(width, rank, bias=False)  # random, NEVER simultaneously zero
        self.up = nn.Linear(rank, output_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, h):
        return self.up(F.gelu(self.down(h)))


class SharedExpertField(nn.Module):
    """Fields consume only z, physical time, c. No U/velocity/reference-field path."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.time = PhysicalTimeEmbedding(config.time_hidden_dim)
        self.input = nn.Sequential(nn.Linear(config.state_dim + config.time_hidden_dim + config.condition_dim,
                                             config.hidden_dim), nn.LayerNorm(config.hidden_dim), nn.SiLU())
        self.blocks = nn.Sequential(*(ResidualBlock(config.hidden_dim) for _ in range(config.residual_blocks)))
        self.base = nn.Linear(config.hidden_dim, config.state_dim)
        self.experts = nn.ModuleList(LowRankExpert(config.hidden_dim, config.expert_rank, config.state_dim)
                                     for _ in range(config.n_experts))

    def forward(self, z, t, condition):
        n = len(z)
        _check_matrix(z, n, self.config.state_dim, "field state")
        _check_matrix(condition, n, self.config.condition_dim, "field condition")
        t = torch.as_tensor(t, dtype=z.dtype, device=z.device)
        if t.ndim == 0:
            t = t.expand(n)
        if t.shape != (n,) or not torch.isfinite(t).all():
            raise ValueError("Physical time must be scalar or one finite time per cell")
        h = self.blocks(self.input(torch.cat((z, self.time(t), condition), -1)))
        return self.base(h)[:, None, :] + torch.stack([expert(h) for expert in self.experts], dim=1)

    def selected(self, z, t, condition, modes):
        if modes.shape != (len(z),) or modes.dtype != torch.long:
            raise ValueError("Need one integer locked expert per source cell")
        if (modes < 0).any() or (modes >= self.config.n_experts).any():
            raise ValueError("Expert index out of range")
        fields = self(z, t, condition)
        return fields[torch.arange(len(z), device=z.device), modes]


@dataclass
class Prediction:
    endpoint: torch.Tensor
    modes: torch.Tensor
    probabilities: torch.Tensor
    times: tuple[float, ...]
    trajectory: tuple[torch.Tensor, ...]


class VeloRoute(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.router = SourceRouter(self.config)
        self.field = SharedExpertField(self.config)

    @torch.no_grad()
    def predict(self, z0, velocity0, condition, *, t0=4., t1=5., max_step=.25,
                generator=None, modes=None, return_trajectory=False):
        if len(z0) == 0 or not all(math.isfinite(x) for x in (t0, t1, max_step)) or t1 <= t0 or max_step <= 0:
            raise ValueError("Need nonempty sources, increasing physical times and a positive integration step")
        probabilities = self.router(z0, velocity0, condition)
        if modes is None:
            modes = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
        else:
            modes = modes.detach().clone()
        # Mode sampled ONCE outside the RK4 loop. No averaging of expert vector fields.
        steps = math.ceil((t1-t0) / max_step)
        if steps > 10000:
            raise ValueError("Integration request exceeds bounded step budget")
        dt = (t1-t0) / steps
        z = z0.detach().clone()
        trajectory = [z.clone()] if return_trajectory else []
        for index in range(steps):
            t = t0 + index*dt
            field = lambda state, time: self.field.selected(state, time, condition, modes)
            a = field(z, t)
            b = field(z+dt*a/2, t+dt/2)
            c = field(z+dt*b/2, t+dt/2)
            d = field(z+dt*c, t+dt)
            z = z + dt*(a+2*b+2*c+d)/6
            if not torch.isfinite(z).all():
                raise FloatingPointError("Nonfinite ODE state")
            if return_trajectory:
                trajectory.append(z.clone())
        return Prediction(z, modes, probabilities, tuple(t0+i*dt for i in range(steps+1)), tuple(trajectory))


def validate_responsibilities(weights, n, k):
    _check_matrix(weights, n, k, "responsibilities")
    if (weights < 0).any() or not torch.allclose(weights.sum(-1), torch.ones(n, device=weights.device, dtype=weights.dtype), atol=1e-5):
        raise ValueError("Responsibilities must be normalized nonnegative rows")


def expert_regression_loss(model, z, physical_time, condition, target_rate, responsibilities):
    """Average PER-EXPERT errors; never regress the expert-average vector field."""
    validate_responsibilities(responsibilities, len(z), model.config.n_experts)
    _check_matrix(target_rate, len(z), model.config.state_dim, "target rate")
    fields = model.field(z.detach(), physical_time, condition.detach())
    errors = (fields-target_rate.detach()[:, None, :]).square().mean(-1)
    return (responsibilities.detach()*errors).sum(-1).mean()


def flow_matching_loss(model, z0, z1, condition, responsibilities, *, t0, t1, generator=None):
    if t1 <= t0 or z0.shape != z1.shape:
        raise ValueError("Invalid paired transition interval/dimensions")
    s = torch.rand((len(z0), 1), device=z0.device, dtype=z0.dtype, generator=generator)
    z = (1-s)*z0 + s*z1
    return expert_regression_loss(model, z, t0+s[:, 0]*(t1-t0), condition,
                                  (z1-z0)/(t1-t0), responsibilities)


def routing_loss(model, z0, velocity0, condition, responsibilities):
    validate_responsibilities(responsibilities, len(z0), model.config.n_experts)
    # Dense CE teaches all logits, including currently non-top-k experts. No gradient to inputs/experts.
    logits = model.router.logits(z0.detach(), velocity0.detach(), condition.detach())
    return -(responsibilities.detach()*logits.log_softmax(-1)).sum(-1).mean()


def save_checkpoint(path, model, *, metadata):
    from pathlib import Path
    with Path(path).open("xb") as stream:
        torch.save({"format_version": 1, "config": asdict(model.config),
                    "state_dict": model.state_dict(), "metadata": metadata}, stream)


def load_checkpoint(path, *, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unknown checkpoint format")
    model = VeloRoute(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(map_location), payload["metadata"]
