"""One-change recipe bridge; original W6 and W9 remain unchanged."""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from w6_local_router import LocalResponseRouter, weighted_energy
from w9_population_router import independent_source_energy


def model_arm(arm):
    if arm not in ('static_router', 'velocity_router', 'shuffled_velocity_router'):
        raise ValueError('Unknown bridge arm')
    return 'velocity_router' if arm == 'shuffled_velocity_router' else arm


def shuffle_velocity(velocity, seed, cohort_index, offset=100000):
    """Fixed within-SOURCE-cohort row permutation, no target or optimizer RNG."""
    generator = np.random.default_rng(np.random.SeedSequence([offset, seed, cohort_index]))
    return np.asarray(velocity)[generator.permutation(len(velocity))].copy()


def bridge_energy(candidates, q, target, config):
    if config['estimator'] == 'U':
        if config['centering'] != 'full_source' or config['distance_kernel'] != 'direct':
            raise ValueError('Source U-statistic needs full-source centering and direct distance')
        return independent_source_energy(candidates, q, target)
    if config['estimator'] != 'V':
        raise ValueError('Unknown energy estimator')
    if config['distance_kernel'] == 'heuristic':
        return weighted_energy(candidates, q, target)
    if config['distance_kernel'] != 'direct':
        raise ValueError('Unknown distance kernel')
    x = candidates.reshape(-1, candidates.shape[-1]); w = q.reshape(-1)/len(q); w = w/w.sum()
    distance = lambda a,b: torch.cdist(a,b,compute_mode='donot_use_mm_for_euclid_dist')
    return 2*(distance(x,target).mean(1)@w)-w@(distance(x,x)@w)-distance(target,target).mean()


def fit_bridge(groups, normalization, config, *, arm, seed, device, progress=None, checkpoint=None, resume=None):
    if config['centering'] not in ('batch', 'full_source'):
        raise ValueError('Unknown centering scope')
    if config['regularization_scope'] not in ('batch', 'full_source'):
        raise ValueError('Unknown regularization scope')
    if config['centering'] == 'batch' and config['regularization_scope'] != 'batch':
        raise ValueError('Bridge does not combine batch centering with full-source regularizers')
    model = LocalResponseRouter(**normalization, config=config, arm=model_arm(arm), seed=seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    tensors = []
    for index, group in enumerate(groups):
        values = dict(group)
        if arm == 'shuffled_velocity_router':
            values['velocity'] = shuffle_velocity(group['velocity'], seed, index, config.get('shuffle_seed_offset',100000))
        tensors.append({key:torch.as_tensor(values[key],dtype=torch.float32,device=device)
                        for key in ('z','velocity','target','global_rate')})
    generator = np.random.default_rng(seed); history = []; start = 0
    if resume is not None:
        model.load_state_dict(resume['model']); optimizer.load_state_dict(resume['optimizer'])
        generator.bit_generator.state = resume['numpy_rng']; torch.set_rng_state(resume['torch_rng'].cpu())
        if str(device).split(':')[0] == 'cuda': torch.cuda.set_rng_state_all([s.cpu() for s in resume['cuda_rng']])
        start = resume['step']; history = list(resume['history'])
    began = time.monotonic(); interval = config.get('log_interval',300)
    for step in range(start, config['steps']):
        group_index = step % len(tensors); group = tensors[group_index]
        si = torch.as_tensor(generator.integers(len(group['z']),size=config['source_batch']),device=device)
        ti = torch.as_tensor(generator.integers(len(group['target']),size=config['target_batch']),device=device)
        if config['centering'] == 'batch':
            candidates,q,raw = model(group['z'][si],group['velocity'][si],group['global_rate'],config['calibration_hours'])
            reg_raw,reg_q = raw,q
        else:
            full,full_q,raw = model(group['z'],group['velocity'],group['global_rate'],config['calibration_hours'])
            candidates,q = full[si],full_q[si]
            reg_raw,reg_q = (raw,full_q) if config['regularization_scope']=='full_source' else (raw[si],q)
        energy = bridge_energy(candidates,q,group['target'][ti],config)/math.sqrt(candidates.shape[-1])
        entropy = -(reg_q*reg_q.clamp_min(1e-12).log()).sum(1).mean()
        usage = reg_q.mean(0); routing = entropy+(usage*usage.clamp_min(1e-12).log()).sum()
        loss = energy+config['residual_regularization']*reg_raw.square().mean()+config['routing_regularization']*routing
        if not torch.isfinite(loss): raise ArithmeticError('Nonfinite bridge objective')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        # Rotating offset avoids logging only 3/15 cohorts, as every-200 did.
        log_step = (step+1) % interval == ((step+1)//interval) % len(tensors)
        if step == start or log_step or step+1 == config['steps']:
            row = dict(step=step+1,group_index=group_index,loss=float(loss.detach()),energy=float(energy.detach()),
                gradient_norm=float(gradient),route_entropy=float(entropy.detach()),
                mode_usage=usage.detach().cpu().tolist(),elapsed_seconds=time.monotonic()-began)
            history.append(row)
            if progress: progress(row)
        if checkpoint and step+1 in config['checkpoint_steps']:
            checkpoint(step+1,model,optimizer,generator,history)
    model.eval()
    return model,dict(history=history,parameters=sum(p.numel() for p in model.parameters()),arm=arm,
        modes=model.modes,preserved_mean=True,centering=config['centering'],estimator=config['estimator'],
        regularization_scope=config['regularization_scope'],distance_kernel=config['distance_kernel'],
        velocity_consumers=[] if arm=='static_router' else ['router'])
