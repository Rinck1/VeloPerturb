"""Run the registered full mainline first; run remaining ablations afterwards."""
import argparse
import json
from pathlib import Path
import time

from veloroute.artifacts import load_config,save_json,sha256
from veloroute.gpu_policy import enforce_gpu_policy
from veloroute.kang_schedule import full_dependencies,full_variants

parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,choices=range(4),required=True)
parser.add_argument('--phase',choices=['all','mainline','followups'],default='all')
parser.add_argument('--resume-interrupted',action='store_true')
parser.add_argument('--output-root',default=None)
parser.add_argument('--variants',default=None,help='comma-separated registered variant names')
parser.add_argument('--seeds',default=None,help='comma-separated integer seeds')
args=parser.parse_args()
extended=load_config('configs/veloroute_kang_extended_20260914.yaml');c=load_config(extended['base_protocol'])
root=Path(c['root']);output_root=Path(args.output_root) if args.output_root else root/'full_experiments'
if args.output_root and not output_root.is_absolute(): raise ValueError('--output-root must be absolute')
enforce_gpu_policy('cuda');deadline=time.monotonic()+7*86400
variants=full_variants(extended,args.phase)
if args.variants:
    # Explicit core-arm reruns may combine the registered mainline with
    # registered follow-up controls while still using the ready-fold dependency.
    all_registered=full_variants(extended,'all')
    requested=set(args.variants.split(','));known={v['name'] for v in all_registered}
    if not requested <= known: raise ValueError(f'Unknown variants: {sorted(requested-known)}')
    variants=[v for v in all_registered if v['name'] in requested]
if not variants: raise ValueError('No variants selected')
seeds=[int(x) for x in args.seeds.split(',')] if args.seeds else c['experiments']['seeds']
if not seeds or len(set(seeds)) != len(seeds): raise ValueError('Invalid or duplicate seeds')
dependencies=full_dependencies(root,args.worker,args.phase)
while not all(p.exists() for p in dependencies):
    if time.monotonic()>deadline:raise TimeoutError('Kang full-grid dependency timeout')
    print(json.dumps(dict(status='waiting_for_SU_folds' if args.phase=='mainline' else 'waiting_for_predecessors',
        worker=args.worker,phase=args.phase,dependencies=[str(p) for p in dependencies])),flush=True);time.sleep(30)

from veloroute.gfg_experiments import train_gfg
from veloroute.gfg_resume import resume_gfg
from veloroute.kang_experiments import completed
from veloroute.kang_full import predict_full,variant_config

for donor in c['donors'][args.worker::4]:
    fold=root/'folds'/donor
    for seed in seeds:
        for variant in variants:
            directory=output_root/donor/f'seed{seed}'/variant['name'];training=directory/'train'
            candidates=[training]+sorted(directory.glob('train_resumed_*'));done=[p for p in candidates if completed(p)]
            if done:training=done[-1]
            elif training.exists():
                if not args.resume_interrupted:raise ValueError('Interrupted full run needs explicit resume')
                cps=[p for d in candidates for p in d.glob('checkpoints/step_*/manifest.json')]
                if not cps:raise ValueError('No full checkpoint available')
                cp=max(cps,key=lambda p:int(p.parent.name.split('_')[1])).parent
                training=directory/f'train_resumed_{len(candidates)}';resume_gfg(cp,training,device='cuda')
            else:train_gfg(variant_config(c,donor,seed,variant),fold,training,arm=variant['arm'],seed=seed,pilot=False)
            prediction=directory/'predict'
            if not completed(prediction):predict_full(c,donor,seed,variant,training/'model.pt',prediction)
            save_json(directory/'selection.json',dict(donor=donor,seed=seed,arm=variant['name'],training=str(training),prediction=str(prediction),
                model_sha256=sha256(training/'model.pt'),prediction_sha256=sha256(prediction/'predictions.npz'),selection_rule='fixed_budget'))
            print(json.dumps(dict(status='full_variant_complete',donor=donor,seed=seed,variant=variant['name'])),flush=True)
if args.phase=='mainline':
    save_json(output_root/f'mainline_worker{args.worker}_complete.json',dict(status='MAINLINE_WORKER_COMPLETE',worker=args.worker,variants=[v['name'] for v in variants],seeds=seeds))
else:
    save_json(output_root/f'full_worker{args.worker}_complete.json',dict(status='FULL_WORKER_COMPLETE',worker=args.worker,variants=[v['name'] for v in variants],seeds=seeds))
