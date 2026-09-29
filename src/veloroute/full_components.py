"""Extended-model components. No future data enters any deployment component."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .artifacts import object_hash
from .contracts import FitScope
from .model import _check_matrix


class StateEncoder(nn.Module):
    """PCA-coordinate denoiser, not an unconstrained new latent coordinate system."""
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.dim = dim
        self.network = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, spliced):
        _check_matrix(spliced, len(spliced), self.dim, 'PCA spliced')
        return spliced + self.network(spliced)

    def reconstruction_loss(self, spliced, *, mask_ratio=.3, generator=None, anchor_weight=.1):
        if not 0 < mask_ratio < 1:
            raise ValueError('Mask ratio must lie strictly between zero and one')
        target = spliced.detach()
        mask = torch.rand(target.shape, device=target.device, generator=generator) < mask_ratio
        mask[~mask.any(-1), 0] = True  # Ensure supervision without always masking coordinate zero.
        prediction = self(target.masked_fill(mask, 0))
        masked = (prediction-target).square()[mask].mean()
        anchor = (self(target)-target).square().mean()
        return masked + anchor_weight*anchor


class DynamicsEncoder(nn.Module):
    """S/U -> representation, velocity and scalar RMS error. Condition is not accepted."""
    def __init__(self, dim, representation=64, hidden=256):
        super().__init__()
        self.dim = dim
        self.network = nn.Sequential(nn.Linear(2*dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                     nn.SiLU(), nn.Linear(hidden, representation))
        self.velocity_head = nn.Linear(representation, dim)
        self.uncertainty_head = nn.Linear(representation, 1)

    def forward(self, spliced, unspliced):
        _check_matrix(spliced, len(spliced), self.dim, 'PCA spliced')
        _check_matrix(unspliced, len(spliced), self.dim, 'projected unspliced')
        r = self.network(torch.cat((spliced, unspliced), -1))
        return r, self.velocity_head(r), F.softplus(self.uncertainty_head(r)).squeeze(-1)

    def reference_loss(self, spliced, unspliced, reference):
        _, velocity, rho = self(spliced.detach(), unspliced.detach())
        _check_matrix(reference, len(velocity), self.dim, 'frozen reference velocity')
        error = (velocity-reference.detach()).square().mean(-1)
        # rho is one scalar per cell; no accidental broadcasting against a D-vector.
        calibration_target = (error.detach()+1e-12).sqrt()
        return error.mean() + .1*(rho-calibration_target).square().mean()


class SetConditionEncoder(nn.Module):
    """Permutation-invariant protein-set attention; padding and time have no signal."""
    def __init__(self, input_dim=2560, output_dim=256, heads=8, head_dim=64):
        super().__init__()
        self.input_dim, self.heads, self.head_dim = input_dim, heads, head_dim
        self.project = nn.Linear(input_dim, output_dim)
        self.qkv = nn.Linear(output_dim, 3*heads*head_dim)
        self.output = nn.Linear(heads*head_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, tokens, mask):
        if (tokens.ndim != 3 or tokens.shape[2] != self.input_dim or tokens.shape[1] < 1
                or mask.shape != tokens.shape[:2] or mask.dtype != torch.bool or not mask.any(-1).all()):
            raise ValueError('Need a nonempty protein token set per cell and a boolean padding mask')
        if not torch.isfinite(tokens[mask]).all():
            raise ValueError('Nonfinite protein embedding')
        x = self.project(tokens.masked_fill(~mask[..., None], 0))
        n, length, _ = x.shape
        q, k, v = self.qkv(x).reshape(n, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask[:, None, None, :], -torch.inf)
        attention = scores.softmax(-1) @ v
        h = self.norm(x+self.output(attention.transpose(1, 2).reshape(n, length, -1)))
        return (h*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)


class FrozenReferenceField(nn.Module):
    """Training-only kNN memory; querying never fits neighbors on held-out data."""
    def __init__(self, dim, neighbors=30):
        super().__init__()
        self.dim, self.neighbors = dim, neighbors
        self.register_buffer('positions', torch.empty(0, dim))
        self.register_buffer('velocities', torch.empty(0, dim))
        self.register_buffer('distance_scale', torch.ones(()))
        self.fit_ids_hash = None

    @torch.no_grad()
    def fit(self, positions, velocities, *, cell_ids, scope: FitScope):
        ids = scope.validate(cell_ids)
        _check_matrix(positions, len(ids), self.dim, 'reference positions')
        _check_matrix(velocities, len(ids), self.dim, 'reference velocities')
        if len(ids) < 2:
            raise ValueError('Reference coverage needs at least two training cells')
        self.positions, self.velocities = positions.detach().clone(), velocities.detach().clone()
        k = min(self.neighbors, len(ids)-1)
        means = []
        for start in range(0, len(ids), 256):
            d = torch.cdist(self.positions[start:start+256], self.positions)
            d[torch.arange(len(d), device=d.device), torch.arange(start, start+len(d), device=d.device)] = torch.inf
            means.append(d.topk(k, largest=False).values.mean(-1))
        self.distance_scale = torch.cat(means).median().clamp_min(1e-5)
        self.fit_ids_hash = object_hash(sorted(ids))

    @torch.no_grad()
    def forward(self, positions):
        _check_matrix(positions, len(positions), self.dim, 'reference query')
        if not len(self.positions):
            return torch.zeros_like(positions), positions.new_full((len(positions),), 1e6)
        fields, distances = [], []
        for query in positions.detach().split(256):
            d, ix = torch.cdist(query, self.positions).topk(min(self.neighbors, len(self.positions)), largest=False)
            weights = 1/(d+1e-5)
            fields.append((self.velocities[ix]*weights[..., None]).sum(1)/weights.sum(1, keepdim=True))
            distances.append(d.mean(-1)/self.distance_scale)
        return torch.cat(fields), torch.cat(distances)


class IntrinsicField(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.alpha_network = nn.Sequential(nn.Linear(dim, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.alpha_network[-1].weight)
        nn.init.constant_(self.alpha_network[-1].bias, math.log(math.expm1(.1)))
        self.correction = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, z, z0, velocity):
        alpha = F.softplus(self.alpha_network(z0.detach()))
        return alpha*F.normalize(velocity.detach(), dim=-1) + self.correction(z)


class ReliabilityGate(nn.Module):
    FEATURE_NAMES = ('rho', 'start_reference_cosine', 'training_delta_u_given_s',
                     'negative_training_knn_distance', 'velocity_norm', 'condition_retrieval_confidence')

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(6, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.constant_(self.network[-1].bias, -2.)
        self.uncertainty_penalty = nn.Parameter(torch.zeros(()))

    def logits(self, features):
        _check_matrix(features, len(features), 6, 'source-only gate whitelist')
        x = features.detach().clone()
        rho = x[:, 0].clamp_min(0)
        x[:, 0] = 0
        # Enforce monotonic nonincreasing trust in uncertainty, including extrapolation.
        return self.network(x).squeeze(-1)-F.softplus(self.uncertainty_penalty)*rho

    def forward(self, features, valid):
        if valid.shape != (len(features),) or valid.dtype != torch.bool:
            raise ValueError('Gate validity must be a source-side boolean per cell')
        return self.logits(features).sigmoid()*valid.to(features.dtype)

    def calibration_loss(self, features, trusted):
        if trusted.shape != (len(features),) or not ((trusted == 0) | (trusted == 1)).all():
            raise ValueError('Calibration labels are 1=trusted, 0=corrupted')
        return F.binary_cross_entropy_with_logits(self.logits(features), trusted.float())


class ModeController(nn.Module):
    """Prune only after every declared training condition reports low usage."""
    def __init__(self, maximum, initial=1, patience=3, minimum_usage=.01):
        super().__init__()
        if not 1 <= initial <= maximum or patience < 1 or not 0 <= minimum_usage < 1:
            raise ValueError('Invalid adaptive mode policy')
        self.patience, self.minimum_usage = patience, minimum_usage
        self.register_buffer('active', torch.arange(maximum) < initial)
        self.register_buffer('retired', torch.zeros(maximum, dtype=torch.bool))
        self.register_buffer('low_epochs', torch.zeros(maximum, dtype=torch.long))

    @torch.no_grad()
    def activate_next(self):
        candidates = torch.where(~self.active & ~self.retired)[0]
        if len(candidates):
            self.active[candidates[0]] = True
            return int(candidates[0])
        return None

    @torch.no_grad()
    def review(self, probabilities, conditions, declared_conditions):
        if set(conditions) != set(declared_conditions) or len(conditions) != len(probabilities):
            raise ValueError('Adaptive K needs a complete training-condition review, not one batch')
        if probabilities.shape[1] != len(self.active) or not torch.isfinite(probabilities).all():
            raise ValueError('Invalid mode usage')
        use = torch.stack([probabilities[[c == label for c in conditions]].mean(0) for label in sorted(set(conditions))])
        low = (use < self.minimum_usage).all(0) & self.active
        self.low_epochs = torch.where(low, self.low_epochs+1, torch.zeros_like(self.low_epochs))
        retire = (self.low_epochs >= self.patience) & self.active
        # Never retire every expert. Preserve the most used currently active mode.
        best = use.mean(0).masked_fill(~self.active, -1).argmax()
        retire[best] = False
        self.active[retire] = False
        self.retired[retire] = True
        return torch.where(retire)[0].tolist()


def sparse_mode_probabilities(logits, active, top_k):
    if not active.any():
        raise ValueError('At least one mode must remain active')
    masked = logits.masked_fill(~active[None, :], -torch.inf)
    selected = masked.topk(min(top_k, int(active.sum())), dim=-1).indices
    values = torch.full_like(masked, -torch.inf).scatter(1, selected, masked.gather(1, selected))
    return values.softmax(-1)


def mode_regularization(logits, active, *, entropy_weight=.01, balance_weight=.01, sparse_weight=.001):
    p = logits[:, active].softmax(-1)
    entropy = -(p*p.clamp_min(1e-8).log()).sum(-1).mean()
    mean = p.mean(0)
    balance = (mean*(mean.clamp_min(1e-8).log()+math.log(p.shape[1]))).sum()
    sparsity = (1-p.square().sum(-1)).mean()
    return entropy_weight*entropy + balance_weight*balance + sparse_weight*sparsity


def corrupt_source(spliced, unspliced, velocity, method, *, generator=None):
    """Corrupt only the velocity pathway; pairing/target values never enter here."""
    if method == 'shift':
        if len(velocity) < 2:
            raise ValueError('Need two cells for a nonidentity shift')
        return unspliced.roll(1, 0), velocity.roll(1, 0)
    if method == 'gaussian':
        noise = torch.randn(velocity.shape, device=velocity.device, generator=generator)
        return unspliced, velocity + noise*velocity.norm(dim=-1, keepdim=True)/math.sqrt(velocity.shape[1])
    if method == 'zero_u':
        return torch.zeros_like(unspliced), None
    raise ValueError('Unknown corruption; use shift, gaussian, or zero_u')
