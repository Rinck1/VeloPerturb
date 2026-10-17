"""Training-only numerical probe and exact finite-batch energy-bias identity."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from w4_artifacts import Artifacts
from w6_local_router import LocalResponseRouter
from w7_router_scaling import training_path, write_csv
from w9_large_training import read_training
from w9_population_router import independent_source_energy
from w10_recipe_model import bridge_energy


def finite_batch_identity():
    x = torch.tensor([[[-1.],[1.]],[[.2],[2.]],[[1.],[3.]]],dtype=torch.float64)
    q = torch.tensor([[.3,.7],[.4,.6],[.8,.2]],dtype=torch.float64)
    y = torch.tensor([[-.7],[2.1]],dtype=torch.float64)
    cfg = dict(estimator='V',centering='full_source',distance_kernel='direct')
    exact = bridge_energy(x,q,y,cfg);n,k,d = x.shape
    distance = torch.cdist(x.reshape(-1,d),x.reshape(-1,d)).reshape(n,k,n,k)
    pair = (distance*q[:,:,None,None]*q[None,None,:,:]).sum((1,3))
    across,conditional = pair.mean(),pair.diagonal().mean();target = torch.cdist(y,y).mean()
    v,u = [],[]
    for si in itertools.product(range(n),repeat=2):
        for ti in itertools.product(range(len(y)),repeat=2):
            v.append(bridge_energy(x[list(si)],q[list(si)],y[list(ti)],cfg))
            u.append(independent_source_energy(x[list(si)],q[list(si)],y[list(ti)]))
    empirical_bias = torch.stack(v).mean()-exact
    predicted_bias = (across-conditional)/2+target/2
    return dict(exact_energy=float(exact),expected_V=float(torch.stack(v).mean()),
        expected_U=float(torch.stack(u).mean()),observed_V_bias=float(empirical_bias),
        predicted_V_bias=float(predicted_bias),identity_error=float(abs(empirical_bias-predicted_bias)),
        U_expectation_error=float(abs(torch.stack(u).mean()-exact)),
        qualification='Fixed full-population candidate distribution; not an identity for random batch-centered cross distances')


def objective_gradient(model,group,si,ti,*,dtype,kernel):
    model = copy.deepcopy(model).to(dtype=dtype)
    z,v,g,y = [torch.as_tensor(group[k],dtype=dtype) for k in ('z','velocity','global_rate','target')]
    x,q,_ = model(z[si],v[si],g,24.)
    cfg = dict(estimator='V',centering='batch',distance_kernel=kernel)
    energy = bridge_energy(x,q,y[ti],cfg)/math.sqrt(z.shape[-1])
    gradients = torch.autograd.grad(energy,tuple(model.parameters()))
    return float(energy.detach()),torch.cat([g.detach().reshape(-1).double() for g in gradients])


def run(root,output):
    root = Path(root);manifest = json.loads((root/'study_manifest.json').read_text())
    job = next(j for j in manifest['jobs'] if j['setting']=='b0_w8_exact' and j['arm']=='velocity_router' and j['seed']==0)
    _,groups,norm,_ = read_training(root,manifest,job)
    cp = training_path(root,job)/'model_step12000.pt'
    run = Artifacts(output,seed=0,inputs=[root/'study_manifest.json',cp,Path(__file__)])
    initial = LocalResponseRouter(**norm,config=job['config'],arm='velocity_router',seed=0)
    trained = copy.deepcopy(initial);trained.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['state_dict'])
    rows = []
    for name,model in (('initial',initial),('trained_12000',trained)):
        for index in (0,len(groups)-1):
            group = groups[index];rng = np.random.default_rng(0)
            si = rng.integers(len(group['z']),size=64);ti = rng.integers(len(group['target']),size=64)
            exact,reference = objective_gradient(model,group,si,ti,dtype=torch.float64,kernel='direct')
            for kernel in ('heuristic','direct'):
                value,gradient = objective_gradient(model,group,si,ti,dtype=torch.float32,kernel=kernel)
                rows.append(dict(state=name,training_group=index,kernel=kernel,
                    energy=value,reference_fp64_energy=exact,energy_absolute_error=abs(value-exact),
                    gradient_relative_error=float((gradient-reference).norm()/reference.norm()),
                    gradient_cosine=float(torch.nn.functional.cosine_similarity(gradient,reference,dim=0))))
    identity = finite_batch_identity()
    if identity['identity_error']>1e-12 or identity['U_expectation_error']>1e-12:
        raise ArithmeticError('Finite-batch identity failed')
    write_csv(run.directory/'precision.csv',rows)
    run.json('summary.json',dict(status='TRAINING_OBJECTIVE_PROBE_COMPLETE',finite_batch_identity=identity,
        precision=rows,optimizer_or_predictions_changed=False,heldout_targets_read=False))
    run.json('status.json',dict(status='complete'));print(json.dumps(dict(identity=identity,precision=rows)),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args = parser.parse_args();run(args.root,args.output)
