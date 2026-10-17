"""Audited population-centered small/18M endpoint mixtures; no GFG/JVP/ODE."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

from w6_local_router import LocalResponseRouter


class StableResidualBlock(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.block = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
            nn.SiLU(), nn.Linear(width, width))
        self.scale = 1/math.sqrt(2*depth)

    def forward(self, x):
        return x+self.scale*self.block(x)


class PopulationResponseRouter(LocalResponseRouter):
    """Forward must receive the entire SOURCE cohort, in training and deployment.

    Inherited forward subtracts the q-weighted population mean with its full
    autograd path. Experts consume S only; local velocity consumes only routing.
    """
    def __init__(self, normalization, config, *, arm, seed):
        if config['architecture'] not in ('small', 'residual_large'):
            raise ValueError('Unknown architecture')
        if config['experts'] < 2 or config['calibration_hours'] <= 0:
            raise ValueError('Need multiple experts and positive hour calibration')
        super().__init__(**normalization, config=config, arm=arm, seed=seed)
        if config['architecture'] == 'residual_large':
            width, depth = config['hidden'], config['residual_blocks']
            if depth < 1: raise ValueError('Positive residual depth required')
            self.backbone = nn.Sequential(nn.Linear(len(self.z_mean), width), nn.LayerNorm(width), nn.SiLU(),
                *[StableResidualBlock(width, depth) for _ in range(depth)], nn.LayerNorm(width))


def independent_source_energy(candidates, q, target):
    """U-statistic over iid SOURCE draws, integrating expert choices exactly.

    Exclude all (k,l) pairs from the same minibatch draw i, not only the same
    flattened candidate (i,k). Different draws may select the same original
    source cell; those pairs remain valid draws from the empirical population.
    Can be negative in a minibatch; do not clamp it.
    """
    b, k, d = candidates.shape; m = len(target)
    if b < 2 or m < 2 or q.shape != (b,k) or target.shape[1] != d:
        raise ValueError('Aligned mixtures with at least two independent draws per side required')
    x = candidates.reshape(-1,d)
    # Avoid squared-norm subtraction cancellation for identical sampled cells.
    distance = lambda a,b:torch.cdist(a,b,compute_mode='donot_use_mm_for_euclid_dist')
    cross = (distance(x,target).mean(1).reshape(b,k)*q).sum()/b
    distances = distance(x,x).reshape(b,k,b,k)
    pair = (distances*q[:,:,None,None]*q[None,None,:,:]).sum((1,3))
    within = (pair.sum()-pair.diagonal().sum())/(b*(b-1))
    yy = distance(target,target)
    within_y = (yy.sum()-yy.diagonal().sum())/(m*(m-1))
    return 2*cross-within-within_y


def make_training_state(model, optimizer, generator, step, history):
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        numpy_rng=generator.bit_generator.state, torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if model.z_mean.device.type == 'cuda' else [],
        step=step, history=history)


def fit_population(groups, normalization, config, *, arm, seed, device, progress=None, checkpoint=None, resume=None):
    model = PopulationResponseRouter(normalization, config, arm=arm, seed=seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    tensors = [{key:torch.as_tensor(group[key], dtype=torch.float32, device=device)
                for key in ('z','velocity','target','global_rate')} for group in groups]
    generator = np.random.default_rng(seed); history = []; start = 0
    if resume is not None:
        model.load_state_dict(resume['model']); optimizer.load_state_dict(resume['optimizer'])
        generator.bit_generator.state = resume['numpy_rng']; torch.set_rng_state(resume['torch_rng'].cpu())
        if device != 'cpu': torch.cuda.set_rng_state_all([s.cpu() for s in resume['cuda_rng']])
        start = resume['step']; history = list(resume['history'])
    began = time.monotonic()
    for step in range(start, config['steps']):
        group = tensors[step % len(tensors)]
        si = torch.as_tensor(generator.integers(len(group['z']), size=config['source_batch']), device=device)
        ti = torch.as_tensor(generator.integers(len(group['target']), size=config['target_batch']), device=device)
        # Entire source population: mean and its parameter gradient match inference.
        full, full_q, raw = model(group['z'], group['velocity'], group['global_rate'], config['calibration_hours'])
        candidates, q = full[si], full_q[si]
        energy = independent_source_energy(candidates, q, group['target'][ti])/math.sqrt(full.shape[-1])
        entropy = -(full_q*full_q.clamp_min(1e-12).log()).sum(1).mean()
        usage = full_q.mean(0)
        negative_information = entropy+(usage*usage.clamp_min(1e-12).log()).sum()
        regularization = config['residual_regularization']*raw.square().mean()
        loss = energy+regularization+config['routing_regularization']*negative_information
        if not torch.isfinite(loss): raise ArithmeticError('Nonfinite population objective')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == start or (step+1)%config.get('log_interval',200) == 0 or step+1 == config['steps']:
            mean_error = ((full*full_q[:,:,None]).sum(1).mean(0)
                -(group['z'].mean(0)+config['calibration_hours']*group['global_rate'])).norm()
            row = dict(step=step+1, loss=float(loss.detach()), energy=float(energy.detach()),
                gradient_norm=float(gradient_norm), mean_anchor_error=float(mean_error.detach()),
                route_entropy=float(entropy.detach()), mode_usage=usage.detach().cpu().tolist(),
                elapsed_seconds=time.monotonic()-began, centering_cells=len(group['z']))
            history.append(row)
            if progress: progress(row)
        if checkpoint and step+1 in config['checkpoint_steps']:
            checkpoint(step+1, model, optimizer, generator, history)
    model.eval()
    return model, dict(history=history, parameters=sum(p.numel() for p in model.parameters()),
        arm=arm, modes=model.modes, preserved_mean=True, population_centering='full_source_differentiable',
        training_energy='independent_source_draw_U_statistic', velocity_consumers=['router'] if arm == 'velocity_router' else [])
