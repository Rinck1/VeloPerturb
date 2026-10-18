"""Budget-matched W14 inputs, unchanged b1 endpoint objective and mean identity."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn

from w6_local_router import LocalResponseRouter
from w10_recipe_model import bridge_energy


ARMS=('static','real','direct_U','nonlinear_SU','shuffled_v')


def arm_feature(group,arm):
    if arm=='static':return np.zeros_like(group['z'])
    return group[{'real':'velocity','shuffled_v':'velocity','direct_U':'direct_U','nonlinear_SU':'pooled_SU'}[arm]]


def permute_source(values,seed,shuffle_seed,group_index):
    rng=np.random.default_rng(np.random.SeedSequence([100000,seed,shuffle_seed,group_index]))
    return values[rng.permutation(len(values))].copy()


def make_model(normalization,config,arm,seed):
    if arm not in ARMS:raise ValueError('Unregistered W14 arm')
    key={'direct_U':'direct_U','nonlinear_SU':'pooled_SU'}.get(arm,'velocity')
    norm=dict(z_mean=normalization['z_mean'],z_scale=normalization['z_scale'],
        v_mean=np.zeros_like(normalization['z_mean']) if arm=='static' else normalization[key+'_mean'],
        v_scale=np.ones_like(normalization['z_scale']) if arm=='static' else normalization[key+'_scale'])
    model=LocalResponseRouter(**norm,config=config,arm='static_router' if arm=='static' else 'velocity_router',seed=seed)
    if arm=='nonlinear_SU':
        # The backbone/expert initialization is identical to the other arms.
        with torch.random.fork_rng():
            torch.manual_seed(seed+10000)
            model.router=nn.Sequential(nn.Linear(3*len(norm['z_mean']),config['nonlinear_router_hidden']),nn.SiLU(),
                nn.Linear(config['nonlinear_router_hidden'],config['experts']))
    return model


def fit(groups,normalization,config,*,arm,seed,shuffle_seed=0,device='cpu',checkpoint=None,progress=None):
    model=make_model(normalization,config,arm,seed).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    tensors=[]
    for index,group in enumerate(groups):
        feature=arm_feature(group,arm)
        if arm=='shuffled_v':feature=permute_source(feature,seed,shuffle_seed,index)
        tensors.append(dict(z=torch.as_tensor(group['z'],dtype=torch.float32,device=device),
            feature=torch.as_tensor(feature,dtype=torch.float32,device=device),
            target=torch.as_tensor(group['target'],dtype=torch.float32,device=device),
            global_rate=torch.as_tensor(group['global_rate'],dtype=torch.float32,device=device)))
    rng=np.random.default_rng(seed);history=[];began=time.monotonic()
    for step in range(config['steps']):
        gi=step%len(tensors);g=tensors[gi]
        si=torch.as_tensor(rng.integers(len(g['z']),size=config['source_batch']),device=device)
        ti=torch.as_tensor(rng.integers(len(g['target']),size=config['target_batch']),device=device)
        x,q,raw=model(g['z'][si],g['feature'][si],g['global_rate'],config['calibration_hours'])
        energy=bridge_energy(x,q,g['target'][ti],config)/math.sqrt(x.shape[-1])
        entropy=-(q*q.clamp_min(1e-12).log()).sum(1).mean();usage=q.mean(0)
        regularizer=entropy+(usage*usage.clamp_min(1e-12).log()).sum()
        loss=energy+config['residual_regularization']*raw.square().mean()+config['routing_regularization']*regularizer
        if not torch.isfinite(loss):raise ArithmeticError('Nonfinite W14 objective')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step==0 or (step+1)%config['log_interval']==0 or step+1==config['steps']:
            row=dict(step=step+1,group_index=gi,loss=float(loss.detach()),energy=float(energy.detach()),
                gradient_norm=float(gradient),route_entropy=float(entropy.detach()),mode_usage=usage.detach().cpu().tolist(),
                elapsed_seconds=time.monotonic()-began)
            history.append(row)
            if progress:progress(row)
        if checkpoint and step+1 in config['checkpoint_steps']:
            checkpoint(step+1,model,optimizer,rng,history)
    model.eval()
    return model,dict(history=history,parameters=sum(p.numel() for p in model.parameters()),arm=arm,
        seed=seed,shuffle_seed=shuffle_seed,elapsed_seconds=time.monotonic()-began)


def source_predictions(model,group,arm,seed,shuffle_seed,group_index,duration):
    device=model.z_mean.device
    feature=arm_feature(group,arm)
    if arm=='shuffled_v':feature=permute_source(feature,seed,shuffle_seed,group_index)
    z,v,g=[torch.as_tensor(a,dtype=torch.float32,device=device) for a in (group['z'],feature,group['global_rate'])]
    with torch.no_grad():
        positions,q,raw=model(z,v,g,duration)
        rates=raw*model.z_scale/model.calibration_hours
        average_q=q.mean(0).expand_as(q)
        centered=rates-(average_q[:,:,None]*rates).sum(1).mean(0)[None,None,:]
        averaged=z[:,None,:]+duration*(g[None,None,:]+centered)
    x,q,a,aq=[t.cpu().numpy().astype(float) for t in (positions,q,averaged,average_q)]
    anchor=group['z'].astype(float).mean(0)+duration*group['global_rate'].astype(float)
    q/=q.sum(1,keepdims=True);aq/=aq.sum(1,keepdims=True)
    x+=anchor-(x*q[:,:,None]).sum(1).mean(0)
    a+=anchor-(a*aq[:,:,None]).sum(1).mean(0)
    np.testing.assert_allclose((x*q[:,:,None]).sum(1).mean(0),anchor,atol=1e-5,rtol=0)
    return dict(candidates=x,q=q,cohort_candidates=a,cohort_q=aq,anchor=anchor,
        full_prediction_mean=(x*q[:,:,None]).sum(1).mean(0))
