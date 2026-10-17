"""Task-trained no-JVP kinetic calibration, initialized at analytic forecasting."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from w4_arms import fit_ridge_training_cv
from w4_kinetic_head import KineticCalibrator


class JointPopulationForecaster(nn.Module):
    def __init__(self, base, velocity, target, *, controls, arm, config, seed):
        super().__init__()
        if arm not in ('static_task','analytic_task','velocity_joint'):
            raise ValueError('Unknown task model')
        torch.manual_seed(seed)
        self.arm = arm
        self.base_dimensions = base.shape[1]
        self.head = None
        design = base if arm == 'static_task' else np.concatenate((base,velocity),axis=1)
        scaler,ridge,self.selection = fit_ridge_training_cv(design,target,controls=controls)
        self.alpha = self.selection['alpha']
        self.training_groups = len(base)
        self.register_buffer('input_mean',torch.tensor(scaler.mean_,dtype=torch.float32))
        self.register_buffer('input_scale',torch.tensor(scaler.scale_,dtype=torch.float32))
        self.readout = nn.Linear(design.shape[1],target.shape[1])
        with torch.no_grad():
            self.readout.weight.copy_(torch.tensor(ridge.coef_,dtype=torch.float32))
            self.readout.bias.copy_(torch.tensor(ridge.intercept_,dtype=torch.float32))
        if arm == 'velocity_joint':
            scale = np.maximum(np.sqrt(np.mean(velocity**2,axis=0)),1e-6)
            self.head = KineticCalibrator(scale,hidden=config['hidden'],
                maximum_relative_correction=config['maximum_relative_correction'])

    def forward(self,base,velocity):
        feature = base
        if self.arm != 'static_task':
            kinetic = self.head(velocity) if self.head is not None else velocity
            feature = torch.cat((base,kinetic),dim=1)
        return self.readout((feature-self.input_mean)/self.input_scale)


def fit_task_model(base,velocity,target,bank,config,*,controls,arm,seed,device):
    model = JointPopulationForecaster(base,velocity,target,controls=controls,arm=arm,config=config,seed=seed).to(device)
    groups = [dict(params=model.readout.parameters(),lr=config['learning_rate_readout'])]
    if model.head is not None:groups.append(dict(params=model.head.parameters(),lr=config['learning_rate_head']))
    optimizer = torch.optim.Adam(groups)
    x,v,y = [torch.as_tensor(bank[key],dtype=torch.float32,device=device) for key in ('base','velocity','delta')]
    generator = np.random.default_rng(seed)
    history = []
    max_task_gradient = 0.
    model.train()
    for step in range(config['steps']):
        index = torch.as_tensor(generator.integers(len(x),size=config['batch_size']),device=device)
        prediction = model(x[index],v[index])
        mse = (prediction-y[index]).square().mean()
        # Keep the original ridge strength per biological training group, not per reused half-sample.
        ridge = model.alpha/(model.training_groups*target.shape[1])*model.readout.weight.square().sum()
        anchor = mse.new_zeros(())
        if model.head is not None:
            anchor = ((model.head(v[index])-v[index])/model.head.scale).square().mean()
            if step == 0 or (step+1) % 100 == 0:
                gradients = torch.autograd.grad(mse,tuple(model.head.parameters()),retain_graph=True)
                norm = float(torch.sqrt(sum(g.square().sum() for g in gradients)).detach())
                max_task_gradient = max(max_task_gradient,norm)
        loss = mse+ridge+config['head_anchor_weight']*anchor
        if not torch.isfinite(loss):raise ArithmeticError('Nonfinite task objective')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step == 0 or (step+1) % 100 == 0 or step+1 == config['steps']:
            history.append(dict(step=step+1,task_mse=float(mse.detach()),ridge=float(ridge.detach()),
                anchor=float(anchor.detach()),loss=float(loss.detach())))
    model.eval()
    return model,dict(history=history,parameters=sum(p.numel() for p in model.parameters()),
        ridge_selection=model.selection,head_task_gradient_max=max_task_gradient,
        head_task_supervision=arm == 'velocity_joint',head_condition_input=False)


def predict_delta(model,base,velocity):
    with torch.no_grad():
        device = model.input_mean.device
        values = [torch.as_tensor(x,dtype=torch.float32,device=device) for x in (base,velocity)]
        return model(*values).cpu().numpy().astype(float)
