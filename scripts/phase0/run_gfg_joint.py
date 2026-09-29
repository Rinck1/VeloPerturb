"""Explicit entry points for GFG inputs, fixed pilot arms, and full training."""
import argparse
import json

from veloroute.artifacts import load_config
from veloroute.gfg_experiments import prepare_gfg_inputs, predict_gfg, train_gfg

parser = argparse.ArgumentParser()
parser.add_argument('phase', choices=('prepare', 'pilot', 'full', 'predict'))
parser.add_argument('--config', default='configs/veloroute_gfg_joint_20260914.yaml')
parser.add_argument('--inputs', default='outputs/veloroute_gfg_inputs_20260914')
parser.add_argument('--arm', default='gfg_joint')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--checkpoint')
parser.add_argument('--output', required=True)
args = parser.parse_args()
config = load_config(args.config)
if args.phase == 'prepare':
    result = prepare_gfg_inputs(config, args.output)
elif args.phase == 'predict':
    result = predict_gfg(args.checkpoint, config, args.inputs, args.output, seed=args.seed)
else:
    result = train_gfg(config, args.inputs, args.output, arm=args.arm, seed=args.seed, pilot=args.phase == 'pilot')
print(json.dumps(result))
