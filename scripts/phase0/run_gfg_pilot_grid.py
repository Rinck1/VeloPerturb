"""Run assigned fixed pilot arms; only source-side predictions are generated."""
import argparse
import json
from pathlib import Path

from veloroute.artifacts import load_config, save_json, sha256
from veloroute.gfg_experiments import predict_gfg, train_gfg
from veloroute.gfg_resume import resume_gfg

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', nargs='+', type=int, required=True)
parser.add_argument('--arms', nargs='+', required=True)
parser.add_argument('--resume-interrupted', action='store_true')
args = parser.parse_args()
config = load_config('configs/veloroute_gfg_joint_20260914.yaml')
root = Path('outputs/veloroute_gfg_pilot_20260914')
inputs = 'outputs/veloroute_gfg_inputs_20260914'
for seed in args.seeds:
    for arm in args.arms:
        directory = root/f'{arm}_seed{seed}'
        training = directory/'train'
        if training.exists():
            candidates = [training]+sorted(directory.glob('train_resumed_*'))
            completed = [p for p in candidates if (p/'provenance.json').exists()
                and json.loads((p/'provenance.json').read_text())['status'] == 'complete']
            if completed:
                training = completed[-1]
            elif args.resume_interrupted:
                checkpoints = [p for c in candidates for p in c.glob('checkpoints/step_*/manifest.json')]
                if not checkpoints:
                    raise ValueError(f'No saved C-stage checkpoint for interrupted run: {directory}')
                checkpoint = max(checkpoints, key=lambda p: int(p.parent.name.split('_')[1])).parent
                training = directory/f'train_resumed_{len(candidates)}'
                result = resume_gfg(checkpoint, training, device=config['device'])
                print(json.dumps({k: result[k] for k in ('status', 'arm', 'seed', 'completed_steps')}), flush=True)
            else:
                raise ValueError(f'Existing unfinished/failed run: {directory}; explicit resume required')
        else:
            result = train_gfg(config, inputs, training, arm=arm, seed=seed, pilot=True)
            print(json.dumps({k: result[k] for k in ('status', 'arm', 'seed', 'completed_steps')}), flush=True)
        prediction = directory/'predict'
        candidates = [prediction]+sorted(directory.glob('predict_resumed_*'))
        completed = [p for p in candidates if (p/'provenance.json').exists()
            and json.loads((p/'provenance.json').read_text())['status'] == 'complete']
        if completed:
            prediction = completed[-1]
        else:
            if prediction.exists():
                if not args.resume_interrupted:
                    raise ValueError('Interrupted prediction requires explicit resume')
                prediction = directory/f'predict_resumed_{len(candidates)}'
            print(json.dumps(predict_gfg(training/'model.pt', config, inputs, prediction, seed=seed)), flush=True)
        save_json(directory/'selection.json', dict(arm=arm, seed=seed, training=str(training),
            prediction=str(prediction), model_sha256=sha256(training/'model.pt'),
            prediction_sha256=sha256(prediction/'predictions.npz'),
            selection_rule='completed_fixed_budget_only_no_effect_based_selection'))
