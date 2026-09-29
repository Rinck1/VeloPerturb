"""Wait for every fixed donor/seed/arm prediction before releasing outer metrics."""
import argparse
import json
from pathlib import Path
import time

from veloroute.artifacts import load_config

parser=argparse.ArgumentParser()
parser.add_argument('--config',default='configs/veloroute_kang_20260914.yaml')
parser.add_argument('--wait-hours',type=float,default=168)
args=parser.parse_args()
c=load_config(args.config);root=Path(c['root']);deadline=time.monotonic()+args.wait_hours*3600
while not all((root/f'primary_worker{i}_complete.json').exists() for i in range(4)):
    if time.monotonic()>deadline:raise TimeoutError('Full primary prediction grid did not complete')
    print(json.dumps(dict(status='waiting_for_all_primary_predictions_no_outer_outcomes_opened')),flush=True)
    time.sleep(30)
from veloroute.kang_evaluation import evaluate_grid
print(json.dumps(evaluate_grid(c,root/'evaluation_primary')),flush=True)
