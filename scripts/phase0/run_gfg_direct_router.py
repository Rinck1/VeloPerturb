import argparse
import json
from pathlib import Path

from veloroute.artifacts import load_config, save_json, sha256
from veloroute.gfg_direct_router import prepare_geometry, predict_direct, train_direct

parser = argparse.ArgumentParser()
parser.add_argument('--seed', required=True, type=int)
parser.add_argument('--resume-interrupted', action='store_true')
args = parser.parse_args()
base = load_config('configs/veloroute_gfg_joint_20260914.yaml')
config = load_config('configs/veloroute_gfg_direct_router_20260914.yaml')
root = Path('outputs/veloroute_gfg_direct_router_20260914')/f'seed{args.seed}'
geometry = root/'geometry'
if not geometry.exists():
    prepare_geometry(base, args.seed, geometry)
elif json.loads((geometry/'provenance.json').read_text())['status'] != 'complete':
    raise ValueError('Interrupted geometry needs a new output root, not an overwrite')
for arm in config['arms']:
    directory = root/arm
    training = directory/'train'
    candidates = [training]+sorted(directory.glob('train_resumed_*'))
    completed = [p for p in candidates if (p/'provenance.json').exists()
        and json.loads((p/'provenance.json').read_text())['status'] == 'complete']
    resume = None
    if completed:
        training = completed[-1]
    else:
        if training.exists():
            if not args.resume_interrupted:
                raise ValueError('Explicit resume required for interrupted direct training')
            points = [p for d in candidates for p in d.glob('checkpoints/step_*/manifest.json')]
            if not points:
                raise ValueError('Interrupted direct-router run has no checkpoint')
            resume = max(points, key=lambda p: int(p.parent.name.split('_')[1])).parent
            training = directory/f'train_resumed_{len(candidates)}'
        result = train_direct(base, config, args.seed, arm, geometry, training, resume=resume)
        print(json.dumps({k: result[k] for k in ('status', 'arm', 'seed', 'steps')}), flush=True)
    prediction = directory/'predict'
    candidates = [prediction]+sorted(directory.glob('predict_resumed_*'))
    completed = [p for p in candidates if (p/'provenance.json').exists()
        and json.loads((p/'provenance.json').read_text())['status'] == 'complete']
    if completed:
        prediction = completed[-1]
    else:
        if prediction.exists():
            if not args.resume_interrupted:
                raise ValueError('Explicit prediction resume required')
            prediction = directory/f'predict_resumed_{len(candidates)}'
        predict_direct(base, args.seed, arm, geometry, training/'model.pt', prediction)
    save_json(directory/'selection.json', dict(training=str(training), prediction=str(prediction), arm=arm,
        seed=args.seed, model_sha256=sha256(training/'model.pt'), prediction_sha256=sha256(prediction/'predictions.npz')))
    print(json.dumps(dict(status='DIRECT_ROUTER_ARM_FROZEN', arm=arm, seed=args.seed)), flush=True)
