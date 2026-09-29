"""Hash-checked C-stage continuation of actual GFG joint training."""
from pathlib import Path
import json

import numpy as np
import torch

from .artifacts import Run, object_hash, save_csv, save_json, sha256
from .contracts import FitScope
from .full_experiments import condition_token_sets
from .full_model import load_full_checkpoint, save_full_checkpoint
from .full_training import FullTrainConfig, FullTrainer
from .gfg_experiments import read_gene_input, training_gene_us
from .latent import load_pack
from .gpu_policy import enforce_gpu_policy


def resume_gfg(directory, output, *, device='cuda'):
    enforce_gpu_policy(device)
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    for name in ('model.pt', 'training_state.pt'):
        if sha256(directory/name) != manifest['files'][name]:
            raise ValueError('GFG resume checkpoint file was modified')
    for path, digest in manifest['inputs'].items():
        if sha256(path) != digest:
            raise ValueError('GFG resume training input was modified')
    model, metadata = load_full_checkpoint(directory/'model.pt', map_location=device)
    if metadata.get('architecture') != 'GFG_joint_VeloRoute' or model.config.dynamics_backend != 'gfg':
        raise ValueError('Not an actual GFG experiment checkpoint')
    effective = metadata['configuration']
    if object_hash(effective) != manifest['config_hash']:
        raise ValueError('GFG resume configuration mismatch')
    fold = Path(effective['fold'])
    gene_paths = [Path(p) for p in manifest['inputs'] if Path(p).name == 'train_gene_source.npz']
    if len(gene_paths) != 1:
        raise ValueError('Cannot resolve unique GFG training gene input')
    values, source, sm = read_gene_input(gene_paths[0], fold/'train_source.npz')
    target, tm = load_pack(fold/'train_target.npz', expected_side='target')
    if sm['role'] != 'train' or tm['role'] != 'train' or sm['fit_ids_hash'] != metadata['fit_ids_hash']:
        raise ValueError('GFG resume must remain on the original training fold')
    values = training_gene_us(values, source, metadata['arm'], metadata['seed'])
    tokens, mask, _ = condition_token_sets(effective['conditions'], source['conditions'])
    z, us, y, proteins = [torch.tensor(v, dtype=torch.float32, device=device)
                         for v in (source['z'], values, target['z'], tokens)]
    padding = torch.tensor(mask, dtype=torch.bool, device=device)
    torch.set_num_threads(effective['cpu_threads'])
    trainer = FullTrainer(model, FullTrainConfig(**effective['training']))
    trainer.load_state_dict(torch.load(directory/'training_state.pt', map_location=device, weights_only=True))
    if not trainer.stage_a_done or not trainer.stage_b_done:
        raise ValueError('Only completed A/B boundary checkpoints can be resumed')
    start = trainer.completed_c_steps
    with Run(output, stage='GFG_joint_training_resume', kind='engineering', config=effective,
        inputs=[directory/name for name in ('model.pt', 'training_state.pt', 'manifest.json')],
        seed=metadata['seed']) as run:
        def progress(row):
            if row['step'] % 10 == 0:
                save_json(run.directory/'status.json', dict(status='running', resumed_c_start=start, **row))
        trainer.progress_callback = progress

        def checkpoint(current):
            path = run.directory/'checkpoints'/f'step_{current.completed_c_steps:07d}'
            path.mkdir(parents=True, exist_ok=False)
            save_full_checkpoint(path/'model.pt', model, metadata=metadata)
            with (path/'training_state.pt').open('xb') as stream:
                torch.save(current.state_dict(), stream)
            save_json(path/'manifest.json', dict(config_hash=manifest['config_hash'], inputs=manifest['inputs'],
                files={name: sha256(path/name) for name in ('model.pt', 'training_state.pt')}))

        trace = trainer.fit(z, us, torch.zeros_like(z), y, proteins, padding,
            source_conditions=source['conditions'].tolist(), target_conditions=target['conditions'].tolist(),
            source_ids=source['cell_ids'].tolist(), target_ids=target['cell_ids'].tolist(),
            scope=FitScope(frozenset(list(source['cell_ids'])+list(target['cell_ids']))), checkpoint_callback=checkpoint)
        save_full_checkpoint(run.directory/'model.pt', model, metadata=metadata)
        save_csv(run.directory/'training_trace.csv', trace)
        gradients = [row['gfg_task_gradient_norm'] for row in trace if row['stage'] == 'C+D']
        result = dict(status='GFG_TRAINING_COMPLETE', **metadata, resumed_c_start=start,
            completed_steps=trainer.completed_c_steps, teacher_updates=trainer.teacher_updates,
            corruption_batches=trainer.corruptions, gfg_task_gradient_max=max(gradients, default=0),
            gfg_task_gradient_median=float(np.median(gradients)))
        save_json(run.directory/'summary.json', result)
        save_json(run.directory/'status.json', {'status': 'complete'})
        (run.directory/'RESULTS.md').write_text('# GFG resumed training\n\n'+json.dumps(result, indent=2))
    return result
