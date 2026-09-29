"""Four disjoint workers; fixed primary grid with resumable native GFG runs."""
import argparse
import json
from pathlib import Path
import time

from veloroute.artifacts import load_config, save_json, sha256
from veloroute.gpu_policy import enforce_gpu_policy

parser=argparse.ArgumentParser()
parser.add_argument('--worker',type=int,choices=range(4),required=True)
parser.add_argument('--config',default='configs/veloroute_kang_20260914.yaml')
parser.add_argument('--wait-hours',type=float,default=72)
parser.add_argument('--resume-interrupted',action='store_true')
args=parser.parse_args()
c=load_config(args.config);root=Path(c['root']);enforce_gpu_policy('cuda')
donors=c['donors'][args.worker::4]
deadline=time.monotonic()+args.wait_hours*3600
while not (root/'folds/ready.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Kang data pipeline readiness timeout')
    print(json.dumps(dict(status='waiting_for_SU_folds',worker=args.worker,donors=donors)),flush=True)
    time.sleep(30)

# Import implementation only once the data are ready. A waiting worker must not
# retain stale function objects while engineering tests finish in the workspace.
from veloroute.gfg_experiments import train_gfg
from veloroute.gfg_resume import resume_gfg
from veloroute.kang_evaluation import define_modes
from veloroute.kang_experiments import (ROUTER_ARMS,base_config,completed,prepare_geometry,
                                       predict_router,predict_simple,train_router)


def train_or_resume(base,fold,directory,kind,**kwargs):
    candidates=[directory]+sorted(directory.parent.glob(directory.name+'_resumed_*'))
    done=[p for p in candidates if completed(p)]
    if done:return done[-1]
    if directory.exists():
        if not args.resume_interrupted:raise ValueError(f'Interrupted stage needs explicit resume: {directory}')
        checkpoints=[p for d in candidates for p in d.glob('checkpoints/step_*/manifest.json')]
        if not checkpoints:raise ValueError(f'No recoverable checkpoint: {directory}')
        latest=max(checkpoints,key=lambda p:int(p.parent.name.split('_')[1])).parent
        out=directory.with_name(directory.name+f'_resumed_{len(candidates)}')
        if kind=='geometry':resume_gfg(latest,out,device='cuda')
        else:train_router(c,**kwargs,output=out,resume=latest)
    else:
        out=directory
        if kind=='geometry':train_gfg(base,fold,out,arm='static',seed=kwargs['seed'],pilot=False)
        else:train_router(c,**kwargs,output=out)
    return out


for donor in donors:
    fold=root/'folds'/donor
    modes=root/'modes'/donor
    if not completed(modes):print(define_modes(c,donor,modes),flush=True)
    for seed in c['experiments']['seeds']:
        run_root=root/'experiments'/donor/f'seed{seed}'
        base=base_config(c,donor,seed)
        training=train_or_resume(base,fold,run_root/'shared_geometry_train','geometry',seed=seed)
        geometry=run_root/'shared_geometry'
        if not completed(geometry):prepare_geometry(c,donor,seed,training/'model.pt',geometry)
        for arm in ROUTER_ARMS:
            directory=run_root/arm
            training=train_or_resume(base,fold,directory/'train','router',donor=donor,seed=seed,arm=arm,geometry=geometry)
            prediction=directory/'predict'
            if not completed(prediction):predict_router(c,donor,seed,arm,geometry,training/'model.pt',prediction)
            save_json(directory/'selection.json',dict(arm=arm,seed=seed,donor=donor,training=str(training),prediction=str(prediction),
                model_sha256=sha256(training/'model.pt'),prediction_sha256=sha256(prediction/'predictions.npz'),
                selection_rule='fixed_budget_no_effect_selection'))
            print(json.dumps(dict(status='primary_arm_complete',donor=donor,seed=seed,arm=arm)),flush=True)
        k1_base=base_config(c,donor,seed,experts=1)
        k1_training=train_or_resume(k1_base,fold,run_root/'K1_geometry_train','geometry',seed=seed)
        k1_geometry=run_root/'K1_geometry'
        if not completed(k1_geometry):prepare_geometry(c,donor,seed,k1_training/'model.pt',k1_geometry)
        for arm in ['static_K1']+c['experiments']['simple_baselines']:
            directory=run_root/arm;prediction=directory/'predict'
            if not completed(prediction):predict_simple(c,donor,seed,arm,prediction,geometry=k1_geometry if arm=='static_K1' else None)
            save_json(directory/'selection.json',dict(arm=arm,seed=seed,donor=donor,prediction=str(prediction),
                prediction_sha256=sha256(prediction/'predictions.npz'),selection_rule='registered_baseline'))
    print(json.dumps(dict(status='donor_primary_complete',donor=donor)),flush=True)
save_json(root/f'primary_worker{args.worker}_complete.json',dict(status='PRIMARY_WORKER_COMPLETE',donors=donors,worker=args.worker))
