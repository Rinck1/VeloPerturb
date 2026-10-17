"""Small, source-trained kinetic-feature denoiser; no GFG decoder or JVP.

The residual is bounded in training-source RMS-standardized coordinates.
Source-cell half estimates are repeat measurements of population features,
not ground-truth velocities or independent biological replicates. The same
architecture is available for direct-U and S-only capacity controls.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import sparse
import torch
from torch import nn


FEATURES = ('vel_pool', 'direct_U_pool', 'pooled_S')


class KineticCalibrator(nn.Module):
    def __init__(self, scale, *, hidden=16, maximum_relative_correction=.25):
        super().__init__()
        scale = torch.as_tensor(scale, dtype=torch.float32).detach().clone()
        if scale.ndim != 1 or not len(scale) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('Finite positive source feature scales required')
        if hidden < 1 or not 0 < maximum_relative_correction < 1:
            raise ValueError('Invalid hidden width or correction bound')
        self.register_buffer('scale', scale)
        self.down = nn.Linear(len(scale), hidden, bias=False)
        self.up = nn.Linear(hidden, len(scale), bias=False)
        nn.init.zeros_(self.up.weight)
        self.maximum_relative_correction = float(maximum_relative_correction)

    def forward(self, velocity):
        if velocity.ndim != 2 or velocity.shape[1] != len(self.scale):
            raise ValueError('Expected population-by-feature velocity matrix')
        normalized = velocity / self.scale
        amplitude = torch.linalg.vector_norm(normalized, dim=1, keepdim=True) / math.sqrt(len(self.scale))
        residual = self.maximum_relative_correction * amplitude * torch.tanh(
            self.up(torch.nn.functional.silu(self.down(normalized))))
        return velocity + self.scale * residual


def pooled_feature_values(transform, summed_s, summed_u, full_s_depth):
    """Accept HVG sums but normalize with depth from ALL strict-S genes."""
    s, u = np.atleast_2d(summed_s), np.atleast_2d(summed_u)
    depth = np.asarray(full_s_depth).reshape(-1)
    if (s.shape != u.shape or s.shape[1] != len(transform.genes) or len(depth) != len(s)
            or not np.isfinite(s).all() or not np.isfinite(u).all() or not np.isfinite(depth).all()
            or (s < 0).any() or (u < 0).any() or (depth <= 0).any()):
        raise ValueError('Invalid pooled counts or full-S depth')
    sn, un = s * (transform.target_sum / depth[:, None]), u * (transform.target_sum / depth[:, None])
    return dict(vel_pool=transform.projected_rate(sn, un-transform.gamma*sn),
                direct_U_pool=np.log1p(un) @ transform.components.T,
                pooled_S=transform.state_normalized(sn))


def source_partition_pairs(transform, s, u, groups, labels, *, partitions, seed):
    """Randomly partition source CELLS, splitting S and U together by identity.

    Repeated partitions reuse cells. They enlarge the optimization bank, never
    the biological sample count. No target or perturbation embedding is read.
    """
    s, u = sparse.csr_matrix(s), sparse.csr_matrix(u)
    groups, labels = np.asarray(groups), list(labels)
    if (s.shape != u.shape or s.shape[0] != len(groups) or s.shape[1] != len(transform.gene_names)
            or partitions < 1 or not labels or len(set(labels)) != len(labels)):
        raise ValueError('Invalid source partition specification')
    if any(not np.isfinite(x.data).all() or (x.data < 0).any() for x in (s, u)):
        raise ValueError('Source counts must be finite and nonnegative')
    sh, uh = s[:, transform.genes], u[:, transform.genes]
    depth = np.asarray(s.sum(1)).ravel()
    if (depth <= 0).any():
        raise ValueError('Apply source technical QC before splitting cells')
    rng = np.random.default_rng(seed)
    left, right, identities = {k: [] for k in FEATURES}, {k: [] for k in FEATURES}, []
    for label in labels:
        rows = np.flatnonzero(groups == label)
        if len(rows) < 2:
            raise ValueError(f'At least two source cells required for {label}')
        for _ in range(partitions):
            a, b = np.split(rng.permutation(rows), [len(rows)//2])
            for rows_half, destination in ((a, left), (b, right)):
                values = pooled_feature_values(transform, np.asarray(sh[rows_half].sum(0)),
                    np.asarray(uh[rows_half].sum(0)), [depth[rows_half].sum()])
                for key in FEATURES:
                    destination[key].append(values[key][0])
            identities.append(label)
    return ({k: np.asarray(v) for k, v in left.items()},
            {k: np.asarray(v) for k, v in right.items()}, np.asarray(identities))


def fit_calibrator(left, right, config, *, seed=0, device='cpu'):
    a, b = np.asarray(left), np.asarray(right)
    if (a.shape != b.shape or a.ndim != 2 or not len(a) or not a.shape[1]
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('Finite paired training-source feature matrices required')
    for key in ('steps', 'batch_size', 'hidden'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'Invalid {key}')
    if (not math.isfinite(config['learning_rate']) or config['learning_rate'] <= 0
            or not math.isfinite(config['weight_decay']) or config['weight_decay'] < 0
            or not math.isfinite(config['anchor_weight']) or config['anchor_weight'] < 0):
        raise ValueError('Invalid optimizer or anchoring configuration')
    torch.manual_seed(seed)
    scale = np.sqrt(np.mean(np.concatenate((a, b))**2, axis=0))
    scale = np.maximum(scale, 1e-6)
    model = KineticCalibrator(scale, hidden=config['hidden'],
        maximum_relative_correction=config['maximum_relative_correction']).to(device)
    a, b = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in (a, b)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    rng = np.random.default_rng(seed)
    history = []
    with torch.no_grad():
        initial = float((((a-b)/model.scale)**2).mean())
    for step in range(config['steps']):
        selected = torch.as_tensor(rng.integers(len(a), size=config['batch_size']), device=device)
        x, y = a[selected], b[selected]
        fx, fy = model(x), model(y)
        data_loss = .5 * (((fx-y)/model.scale).square().mean() + ((fy-x)/model.scale).square().mean())
        anchor = .5 * (((fx-x)/model.scale).square().mean() + ((fy-y)/model.scale).square().mean())
        loss = data_loss + config['anchor_weight']*anchor
        if not torch.isfinite(loss):
            raise ArithmeticError('Nonfinite kinetic-head objective')
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 0 or (step+1) % 100 == 0 or step+1 == config['steps']:
            history.append(dict(step=step+1, loss=float(loss.detach()),
                denoising_loss=float(data_loss.detach()), anchor=float(anchor.detach())))
    model.eval()
    with torch.no_grad():
        final = .5 * ((((model(a)-b)/model.scale)**2).mean() + (((model(b)-a)/model.scale)**2).mean())
    return model, dict(initial_pair_mse=initial, final_pair_mse=float(final), history=history,
        parameters=sum(p.numel() for p in model.parameters()), head_condition_input=False,
        head_future_supervision=False, biological_replicates_added=0,
        training_objective='source_half_population_feature_stability_not_true_velocity_accuracy')


def apply_calibrator(model, values):
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Finite feature matrix required')
    with torch.no_grad():
        return model(torch.as_tensor(values, dtype=torch.float32, device=model.scale.device)).cpu().numpy()
