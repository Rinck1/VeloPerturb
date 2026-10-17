"""Velocity-conditioned response mixtures with an exactly preserved population mean."""
from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn


def local_pooled_velocity(transform,s,u,z,baseline,neighbors):
    """Within-source S-neighborhoods; pool counts BEFORE the log derivative."""
    if neighbors < 2 or s.shape != u.shape or len(z) != s.shape[0]:
        raise ValueError('Aligned source S/U and neighborhood required')
    k = min(neighbors,len(z))
    indices = NearestNeighbors(n_neighbors=k,n_jobs=4).fit(z).kneighbors(z,return_distance=False)
    row = np.repeat(np.arange(len(z)),k)
    adjacency = sparse.csr_matrix((np.ones(len(row)),(row,indices.reshape(-1))),shape=(len(z),len(z)))
    depth = np.asarray(s.sum(1)).ravel()
    exposure = np.asarray(adjacency@depth).reshape(-1,1)
    if (exposure <= 0).any():raise ValueError('Empty local S library')
    factor = transform.target_sum/exposure
    sn = (adjacency@s[:,transform.genes]).toarray()*factor
    un = (adjacency@u[:,transform.genes]).toarray()*factor
    values = transform.projected_rate(sn,un-transform.gamma*sn)-np.asarray(baseline)
    if not np.isfinite(values).all():raise ValueError('Nonfinite local kinetic features')
    return values.astype('float32')


def energy_from_weights(x,w,target):
    """Energy V-statistic for a normalized weighted empirical source."""
    w = w/w.sum()
    cross = torch.cdist(x,target).mean(1)@w
    within_x = w@(torch.cdist(x,x)@w)
    within_y = torch.cdist(target,target).mean()
    return 2*cross-within_x-within_y


def weighted_energy(candidates,probabilities,target):
    """Exact energy V-statistic of the full weighted candidate mixture."""
    return energy_from_weights(candidates.reshape(-1,candidates.shape[-1]),
        probabilities.reshape(-1)/candidates.shape[0],target)


class LocalResponseRouter(nn.Module):
    def __init__(self,z_mean,z_scale,v_mean,v_scale,config,*,arm,seed):
        super().__init__()
        if arm not in ('k1','static_router','velocity_router'):raise ValueError('Unknown local model')
        torch.manual_seed(seed)
        self.arm = arm
        self.modes = 1 if arm == 'k1' else config['experts']
        self.calibration_hours = config['calibration_hours']
        self.limit = config['maximum_state_correction']
        for name,value in (('z_mean',z_mean),('z_scale',z_scale),('v_mean',v_mean),('v_scale',v_scale)):
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32).detach().clone())
        d = len(z_mean);width = config['hidden'];rank = config['expert_rank']
        self.backbone = nn.Sequential(nn.Linear(d,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(width,rank),nn.SiLU(),nn.Linear(rank,d))
            for _ in range(self.modes)])
        for expert in self.experts:
            nn.init.normal_(expert[-1].weight,std=.001);nn.init.zeros_(expert[-1].bias)
        self.router = None if self.modes == 1 else nn.Sequential(nn.Linear(2*d,config['router_hidden']),
            nn.SiLU(),nn.Linear(config['router_hidden'],self.modes))

    def forward(self,z,velocity,global_rate,duration):
        state = (z-self.z_mean)/self.z_scale
        if self.router is None:
            q = torch.ones((len(z),1),dtype=z.dtype,device=z.device)
        else:
            kinetic = ((velocity-self.v_mean)/self.v_scale).clamp(-5,5) if self.arm == 'velocity_router' else torch.zeros_like(velocity)
            q = self.router(torch.cat((state,kinetic),dim=1)).softmax(-1)
        h = self.backbone(state)
        raw = torch.stack([expert(h) for expert in self.experts],dim=1).tanh()*self.limit
        rates = raw*self.z_scale/self.calibration_hours
        # Preserve global prediction for the ACTUAL mixture, not an averaged expert endpoint.
        mean_rate = (q[:,:,None]*rates).sum(1).mean(0)
        centered = rates-mean_rate[None,None,:]
        positions = z[:,None,:]+duration*(global_rate[None,None,:]+centered)
        return positions,q,raw


def fit_local(groups,normalization,config,*,arm,seed,device,progress=None):
    model = LocalResponseRouter(**normalization,config=config,arm=arm,seed=seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    tensors = [{key:torch.as_tensor(group[key],dtype=torch.float32,device=device)
                for key in ('z','velocity','target','global_rate')} for group in groups]
    generator = np.random.default_rng(seed)
    history = []
    for step in range(config['steps']):
        group = tensors[step%len(tensors)]
        source = torch.as_tensor(generator.integers(len(group['z']),size=config['source_batch']),device=device)
        target = torch.as_tensor(generator.integers(len(group['target']),size=config['target_batch']),device=device)
        candidates,q,raw = model(group['z'][source],group['velocity'][source],group['global_rate'],config['calibration_hours'])
        energy = weighted_energy(candidates,q,group['target'][target])/math.sqrt(candidates.shape[-1])
        entropy = -(q*q.clamp_min(1e-12).log()).sum(1).mean()
        usage = q.mean(0)
        routing = entropy+(usage*usage.clamp_min(1e-12).log()).sum()
        loss = energy+config['residual_regularization']*raw.square().mean()+config['routing_regularization']*routing
        if not torch.isfinite(loss):raise ArithmeticError('Nonfinite local-router loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step == 0 or (step+1)%200 == 0 or step+1 == config['steps']:
            row = dict(step=step+1,loss=float(loss.detach()),energy=float(energy.detach()),
                route_entropy=float(entropy.detach()),mode_usage=usage.detach().cpu().tolist())
            history.append(row)
            if progress:progress(row)
    model.eval()
    return model,dict(history=history,parameters=sum(p.numel() for p in model.parameters()),
        arm=arm,modes=model.modes,preserved_mean=True,velocity_consumers=['router'] if arm == 'velocity_router' else [])


def predict_local(model,z,velocity,global_rate,duration):
    device = model.z_mean.device
    with torch.no_grad():
        x,v,g = [torch.as_tensor(value,dtype=torch.float32,device=device) for value in (z,velocity,global_rate)]
        positions,q,_ = model(x,v,g,duration)
    return positions.cpu().numpy(),q.cpu().numpy()
