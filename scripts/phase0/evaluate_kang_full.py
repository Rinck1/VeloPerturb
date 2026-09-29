import json
from pathlib import Path
import time
import argparse

from veloroute.artifacts import load_config

parser=argparse.ArgumentParser();parser.add_argument('--prediction-root',default=None);parser.add_argument('--output',default=None)
parser.add_argument('--variants',default=None);args=parser.parse_args()
extended=load_config('configs/veloroute_kang_extended_20260914.yaml');c=load_config(extended['base_protocol'])
root=Path(c['root']);prediction_root=Path(args.prediction_root) if args.prediction_root else root/'full_experiments';deadline=time.monotonic()+14*86400
while not all((prediction_root/f'mainline_worker{i}_complete.json').exists() for i in range(4)):
    if time.monotonic()>deadline:raise TimeoutError('Full ablation grid incomplete')
    print(json.dumps(dict(status='waiting_for_full_ablation_predictions')),flush=True);time.sleep(30)
from veloroute.kang_evaluation import evaluate_grid
arms=args.variants.split(',') if args.variants else [v['name'] for v in extended['full_variants']]
output=Path(args.output) if args.output else root/'evaluation_full'
print(json.dumps(evaluate_grid(c,output,experiment_group='full_experiments',experiment_root=prediction_root,
    registered_arms=arms,real_arm='full_GFG_EM_gate',
    primary_controls=('full_static_matched','full_GFG_shuffled_U'))),flush=True)
