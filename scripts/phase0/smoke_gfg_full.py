"""All-component engineering acceptance on training cells, not an effect experiment."""
import argparse
import json
from dataclasses import replace

import torch

from veloroute.artifacts import Run, load_config, save_csv, save_json
from veloroute.contracts import FitScope
from veloroute.full_experiments import condition_token_sets
from veloroute.full_model import FullConfig, FullVeloRoute, load_full_checkpoint, save_full_checkpoint
from veloroute.full_training import FullTrainConfig, FullTrainer
from veloroute.gfg_experiments import read_gene_input
from veloroute.latent import FrozenSplicingTransform, load_pack
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
config = load_config('configs/veloroute_gfg_joint_20260914.yaml')
fold = Path(config['fold'])
gene_path = Path('outputs/veloroute_gfg_inputs_20260914/train_gene_source.npz')
values, source, sm = read_gene_input(gene_path, fold/'train_source.npz')
target, tm = load_pack(fold/'train_target.npz', expected_side='target')
device = config['device']
torch.set_num_threads(4)
torch.manual_seed(81)
model = FullVeloRoute(FullConfig(**config['model'])).to(device)
model.dynamics.core.load_pretrained(config['gfg_checkpoint'])
transform = FrozenSplicingTransform.load(fold/'transform.npz')
gene_us = torch.tensor(values, device=device)
model.dynamics.prepare(gene_us, torch.tensor(transform.components, dtype=torch.float32, device=device),
    cell_ids=source['cell_ids'].tolist(), scope=FitScope(frozenset(source['cell_ids'])))
conditions = sorted(set(source['conditions']))[:2]  # Lexical choice, never future effects.
si = [i for c in conditions for i in list((source['conditions'] == c).nonzero()[0])[:16]]
ti = [i for c in conditions for i in list((target['conditions'] == c).nonzero()[0])[:16]]
tokens, mask, _ = condition_token_sets(config['conditions'], source['conditions'][si])
z, us, y, protein = [torch.tensor(v, dtype=torch.float32, device=device)
                    for v in (source['z'][si], values[si], target['z'][ti], tokens)]
padding = torch.tensor(mask, dtype=torch.bool, device=device)
train_config = replace(FullTrainConfig(**config['training']), stage_a_steps=2, stage_b_steps=2,
    stage_c_steps=16, e_interval=1, teacher_lag=3, activation_interval=2,
    reference_interval=8, usage_interval=8, checkpoint_interval=8, seed=81)
with Run(args.output, stage='GFG_full_width_all_component_engineering', kind='engineering',
         config={'model': config['model'], 'short_steps': [2, 2, 16], 'effect_claim': False},
         inputs=[gene_path, fold/'train_target.npz', config['gfg_checkpoint']], seed=81) as run:
    trainer = FullTrainer(model, train_config)
    kwargs = dict(source_conditions=source['conditions'][si].tolist(), target_conditions=target['conditions'][ti].tolist(),
        source_ids=source['cell_ids'][si].tolist(), target_ids=target['cell_ids'][ti].tolist(),
        scope=FitScope(frozenset(list(source['cell_ids'][si])+list(target['cell_ids'][ti]))))
    trace = trainer.fit(z, us, torch.zeros_like(z), y, protein, padding, **kwargs)
    path = run.directory/'model.pt'
    save_full_checkpoint(path, model, metadata={'kind': 'engineering', 'effect_claim': False})
    restored, _ = load_full_checkpoint(path, map_location=device)
    model.eval()
    restored.eval()
    modes = torch.zeros(8, device=device, dtype=torch.long)
    inputs = [v[:8] for v in (z, us, protein, padding)]
    a = model.predict(*inputs, modes=modes, stochastic=False)
    b = restored.predict(*inputs, modes=modes, stochastic=False)
    torch.testing.assert_close(a.endpoint, b.endpoint)
    stochastic = restored.predict(*inputs, generator=torch.Generator(device=device).manual_seed(3), stochastic=True)
    assert torch.isfinite(stochastic.endpoint).all()
    assert trainer.corruptions == 4 and trainer.teacher_updates > 0
    assert int(model.active_modes.sum()) == 8
    assert max(r['gfg_task_gradient_norm'] for r in trace) > 0
    save_csv(run.directory/'training_trace.csv', trace)
    summary = dict(status='GFG_FULL_COMPONENT_SMOKE_PASS', checkpoint_roundtrip=True,
        teacher_updates=trainer.teacher_updates, corruption_batches=trainer.corruptions,
        active_experts=int(model.active_modes.sum()), stochastic_prediction_finite=True,
        deployment_parameters=sum(p.numel() for p in model.parameters()),
        task_gradient_max=max(r['gfg_task_gradient_norm'] for r in trace), effect_claim=False)
    save_json(run.directory/'summary.json', summary)
    (run.directory/'RESULTS.md').write_text('# GFG full-width engineering smoke\n\n'+json.dumps(summary, indent=2))
print(json.dumps(summary))
