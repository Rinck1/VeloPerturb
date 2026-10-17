"""Small fixed-route mixture ODE and its parameter/update-matched endpoint control."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

from w6_local_router import LocalResponseRouter
from w10_recipe_model import bridge_energy, model_arm, shuffle_velocity


def elapsed_clock(hours, calibration, kernel):
    """Effective hours since ONE forecast origin; zero is valid for an ODE."""
    if hours < 0 or calibration <= 0 or not math.isfinite(hours + calibration):
        raise ValueError('Finite nonnegative elapsed hours and positive calibration required')
    if kernel == 'linear':
        return float(hours)
    if kernel == 'decay':
        return calibration * (-math.expm1(-hours / calibration)) / (-math.expm1(-1.))
    raise ValueError('Time model supports smooth linear or decay clocks only')


def clock_derivative(hours, calibration, kernel):
    elapsed_clock(hours, calibration, kernel)
    return 1. if kernel == 'linear' else math.exp(-hours / calibration) / (-math.expm1(-1.))


def rk4_step(function, state, time, delta):
    a = function(time, state)
    b = function(time + delta / 2, state + delta * a / 2)
    c = function(time + delta / 2, state + delta * b / 2)
    d = function(time + delta, state + delta * c)
    return state + delta * (a + 2*b + 2*c + d) / 6


class TimeResponseRouter(LocalResponseRouter):
    def __init__(self, normalization, config, *, arm, seed):
        super().__init__(**normalization, config=config, arm=model_arm(arm), seed=seed)
        self.kernel = config['time_kernel']
        self.max_step = config['rk4_step_hours']
        if not math.isfinite(self.max_step) or self.max_step <= 0:
            raise ValueError('Positive finite RK4 step required')
        self.time_projection = nn.Linear(4, config['hidden'], bias=False)
        nn.init.zeros_(self.time_projection.weight)

    def route(self, z, velocity):
        state = (z-self.z_mean)/self.z_scale
        kinetic = ((velocity-self.v_mean)/self.v_scale).clamp(-5,5) if self.arm == 'velocity_router' else torch.zeros_like(z)
        return self.router(torch.cat((state,kinetic),dim=1)).softmax(-1)

    def field(self, positions, elapsed):
        """Expert k sees its OWN evolving position, never the mean of experts."""
        scale = elapsed/self.calibration_hours
        features = positions.new_tensor([scale, math.sin(math.pi*scale),
                                         math.cos(math.pi*scale)-1, scale*scale])
        state = (positions-self.z_mean)/self.z_scale
        hidden = self.backbone[1](self.backbone[0](state) + self.time_projection(features))
        hidden = self.backbone[3](self.backbone[2](hidden))
        raw = torch.stack([expert(hidden[:,k]) for k,expert in enumerate(self.experts)],dim=1).tanh()*self.limit
        return raw*self.z_scale/self.calibration_hours, raw

    def context(self, z, velocity, global_rate):
        return dict(source=z, source_mean=z.mean(0), q=self.route(z,velocity), global_rate=global_rate)

    def integrate(self, context, stop, *, start=0., residual=None, max_step=None):
        if stop < start or start < 0 or not math.isfinite(stop + start):
            raise ValueError('Ordered finite elapsed times required')
        q, mean, rate = [context[k] for k in ('q','source_mean','global_rate')]
        if residual is None:
            if start != 0:
                raise ValueError('Restart requires the residual and ORIGINAL source context')
            residual = (context['source']-mean)[:,None,:].expand(-1,self.modes,-1)
        penalties = []
        def derivative(t, y):
            anchor = mean+elapsed_clock(t,self.calibration_hours,self.kernel)*rate
            field,raw = self.field(y+anchor,t)
            penalties.append(raw.square().mean())
            centered = field-(q[:,:,None]*field).sum(1).mean(0)
            return clock_derivative(t,self.calibration_hours,self.kernel)*centered
        step = self.max_step if max_step is None else max_step
        if not math.isfinite(step) or step <= 0:
            raise ValueError('Positive finite step required')
        count = max(1,math.ceil((stop-start)/step)) if stop > start else 0
        delta = (stop-start)/max(count,1)
        for index in range(count):
            residual = rk4_step(derivative,residual,start+index*delta,delta)
        anchor = mean+elapsed_clock(stop,self.calibration_hours,self.kernel)*rate
        penalty = torch.stack(penalties).mean() if penalties else residual.new_zeros(())
        return residual+anchor, residual, penalty

    def forward(self, z, velocity, global_rate, duration, *, dynamics='ode', max_step=None):
        ctx = self.context(z,velocity,global_rate)
        if dynamics == 'ode':
            positions,_,penalty = self.integrate(ctx,float(duration),max_step=max_step)
        elif dynamics == 'endpoint_time':
            # Same field/parameters and midpoint time, but no evolving-state integration.
            initial = z[:,None,:].expand(-1,self.modes,-1)
            rates,raw = self.field(initial,float(duration)/2)
            rates = rates-(ctx['q'][:,:,None]*rates).sum(1).mean(0)
            positions = initial+elapsed_clock(float(duration),self.calibration_hours,self.kernel)*(global_rate+rates)
            penalty = raw.square().mean()
        else:
            raise ValueError('Unknown time-model dynamics')
        return positions,ctx['q'],penalty


def fit_time(groups, normalization, config, *, arm, seed, dynamics, device,
             initial_state, progress=None, checkpoint=None, resume=None):
    model = TimeResponseRouter(normalization,config,arm=arm,seed=seed).to(device)
    missing,unexpected = model.load_state_dict(initial_state,strict=False)
    if missing != ['time_projection.weight'] or unexpected:
        raise ValueError('Expected an unchanged endpoint model plus zero time projection')
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    tensors = []
    for index,group in enumerate(groups):
        values = dict(group)
        if arm == 'shuffled_velocity_router':
            values['velocity'] = shuffle_velocity(group['velocity'],seed,index,config['shuffle_seed_offset'])
        tensors.append({k:torch.as_tensor(values[k],dtype=torch.float32,device=device)
                        for k in ('z','velocity','target','global_rate')})
    generator = np.random.default_rng(seed);history = [];start = 0
    if resume is not None:
        model.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer'])
        generator.bit_generator.state = resume['numpy_rng'];torch.set_rng_state(resume['torch_rng'].cpu())
        if str(device).split(':')[0] == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in resume['cuda_rng']])
        start = resume['step'];history = list(resume['history'])
    began = time.monotonic()
    for step in range(start,config['steps']):
        gi = step%len(tensors);group = tensors[gi]
        si = torch.as_tensor(generator.integers(len(group['z']),size=config['source_batch']),device=device)
        ti = torch.as_tensor(generator.integers(len(group['target']),size=config['target_batch']),device=device)
        candidates,q,penalty = model(group['z'][si],group['velocity'][si],group['global_rate'],
                                    config['calibration_hours'],dynamics=dynamics)
        energy = bridge_energy(candidates,q,group['target'][ti],config)/math.sqrt(candidates.shape[-1])
        entropy = -(q*q.clamp_min(1e-12).log()).sum(1).mean()
        usage = q.mean(0);routing = entropy+(usage*usage.clamp_min(1e-12).log()).sum()
        loss = energy+config['residual_regularization']*penalty+config['routing_regularization']*routing
        if not torch.isfinite(loss):
            raise ArithmeticError('Nonfinite time-model objective')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step == start or (step+1)%config['log_interval']==0 or step+1==config['steps']:
            row = dict(step=step+1,group_index=gi,loss=float(loss.detach()),energy=float(energy.detach()),
                gradient_norm=float(gradient),route_entropy=float(entropy.detach()),
                mode_usage=usage.detach().cpu().tolist(),elapsed_seconds=time.monotonic()-began)
            history.append(row)
            if progress:progress(row)
        if checkpoint and step+1 in config['checkpoint_steps']:
            checkpoint(step+1,model,optimizer,generator,history)
    model.eval()
    return model,dict(history=history,parameters=sum(p.numel() for p in model.parameters()),
        dynamics=dynamics,mean_frozen=True,routing_fixed_along_trajectory=True,
        integration='differentiable_RK4' if dynamics=='ode' else 'no_integration',
        velocity_consumers=[] if arm=='static_router' else ['source_router'])
