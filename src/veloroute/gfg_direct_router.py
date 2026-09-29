"""One bounded follow-up: jointly train GFG/router from unpaired distributions.

Every arm shares the same seed-matched static candidate fields. No OT/EM mode
labels supervise the router. This does not identify real individual cell fates.
"""
from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import torch

from .artifacts import Run, object_hash, save_csv, save_json, sha256
from .full_experiments import condition_token_sets
from .full_model import load_full_checkpoint, save_full_checkpoint
from .gfg_experiments import read_gene_input, training_gene_us
from .latent import FrozenSplicingTransform, load_pack
from .mixture_energy import candidate_endpoints, mixture_energy
from .gpu_policy import enforce_gpu_policy


def prepare_geometry(base, seed, output):
    enforce_gpu_policy(base['device'])
    selection = json.loads(Path(f"outputs/veloroute_gfg_pilot_20260914/static_seed{seed}/selection.json").read_text())
    checkpoint = Path(selection['training'])/'model.pt'
    if sha256(checkpoint) != selection['model_sha256']:
        raise ValueError('Static geometry checkpoint was modified')
    model, metadata = load_full_checkpoint(checkpoint, map_location=base['device'])
    model.eval()
    torch.set_num_threads(base['cpu_threads'])
    fold = Path(base['fold'])
    with Run(output, stage='GFG_direct_router_frozen_geometry', kind='engineering',
        config=dict(seed=seed, selection='same_seed_static_no_effect_search', all_fields_frozen=True),
        inputs=[checkpoint, fold/'train_source.npz', fold/'validation_source.npz', base['conditions']], seed=seed) as run:
        for role in ('train', 'validation'):
            source, sm = load_pack(fold/f'{role}_source.npz', expected_side='source')
            tokens, mask, _ = condition_token_sets(base['conditions'], source['conditions'])
            states, conditions, candidates = [], [], []
            with torch.no_grad():
                for start in range(0, len(source['z']), 256):
                    ix = slice(start, start+256)
                    z = model.encode_state(torch.tensor(source['z'][ix], device=base['device']))
                    e = model.condition_encoder(torch.tensor(tokens[ix], device=base['device']),
                        torch.tensor(mask[ix], device=base['device']))
                    states.append(z.cpu().numpy())
                    conditions.append(e.cpu().numpy())
                    candidates.append(candidate_endpoints(model, z, e).cpu().numpy())
            with (run.directory/f'{role}.npz').open('xb') as stream:
                np.savez_compressed(stream, state=np.concatenate(states), embedding=np.concatenate(conditions),
                    candidates=np.concatenate(candidates), cell_ids=source['cell_ids'], conditions=source['conditions'])
        save_json(run.directory/'summary.json', dict(status='FROZEN_CANDIDATES_READY',
            checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), seed=seed,
            transform_hash=metadata['transform_hash'], validation_target_read=False))
        (run.directory/'RESULTS.md').write_text('# Fixed static candidate geometry\n\nNo development target was loaded.\n')


def load_geometry(directory, role, source):
    directory = Path(directory)
    provenance = json.loads((directory/'provenance.json').read_text())
    if provenance['status'] != 'complete' or sha256(directory/f'{role}.npz') != provenance['outputs'][f'{role}.npz']:
        raise ValueError('Frozen geometry incomplete or modified')
    with np.load(directory/f'{role}.npz', allow_pickle=False) as file:
        data = {key: file[key] for key in file.files}
    for name in ('cell_ids', 'conditions'):
        if not np.array_equal(data[name], source[name]):
            raise ValueError('Frozen geometry/source alignment mismatch')
    return data


def train_direct(base, config, seed, arm, geometry, output, *, resume=None):
    enforce_gpu_policy(base['device'])
    device, fold = base['device'], Path(base['fold'])
    torch.set_num_threads(base['cpu_threads'])
    gene_path = Path('outputs/veloroute_gfg_inputs_20260914/train_gene_source.npz')
    values, source, sm = read_gene_input(gene_path, fold/'train_source.npz')
    target, tm = load_pack(fold/'train_target.npz', expected_side='target')
    if sm['role'] != 'train' or tm['role'] != 'train' or set(source['conditions']) != set(target['conditions']):
        raise ValueError('Direct supervision requires the matched training fold')
    data = load_geometry(geometry, 'train', source)
    gm = json.loads((Path(geometry)/'summary.json').read_text())
    if sha256(gm['checkpoint']) != gm['checkpoint_sha256'] or arm not in config['arms']:
        raise ValueError('Invalid common initialization or arm')
    model, _ = load_full_checkpoint(Path(resume)/'model.pt' if resume else gm['checkpoint'], map_location=device)
    model.config = replace(model.config, use_dynamics=arm != 'static', joint_dynamics=True)
    model.requires_grad_(False)
    model.router.requires_grad_(True)
    model.dynamics.requires_grad_(True)
    for encoder in (model.dynamics.core.manifold_encoder, model.dynamics.core.velocity_encoder):
        encoder.attention_layer.requires_grad_(False)
    model.eval()  # Deterministic GFG map; eval does not disable task gradients.
    parameters = list(model.router.parameters())+[p for p in model.dynamics.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([dict(params=model.router.parameters(), lr=config['router_lr']),
        dict(params=[p for p in model.dynamics.parameters() if p.requires_grad], lr=config['gfg_lr'])])
    values = training_gene_us(values, source, arm, seed)
    convert = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
    z, state, us, e, candidates, y = map(convert,
        (source['z'], data['state'], values, data['embedding'], data['candidates'], target['z']))
    labels = sorted(set(source['conditions']))
    source_groups = {c: np.flatnonzero(source['conditions'] == c) for c in labels}
    target_groups = {c: np.flatnonzero(target['conditions'] == c) for c in labels}
    rng, trace, start_step = np.random.default_rng(seed), [], 0
    effective = dict(**config, seed=seed, arm=arm, initial_checkpoint_sha256=gm['checkpoint_sha256'])
    inputs = [gene_path, fold/'train_source.npz', fold/'train_target.npz', Path(geometry)/'train.npz', gm['checkpoint']]
    binding = {str(p): sha256(p) for p in inputs}
    metadata = dict(architecture='GFG_direct_distribution_router', kind=sm['kind'], task='ER-short', arm=arm, seed=seed,
        transform_hash=sm['transform_hash'], condition_hash=sha256(base['conditions']), GFG_joint=True,
        supervision='unpaired_distribution_not_EM', configuration=effective, confirmation='SEALED')
    if resume:
        resume = Path(resume)
        manifest = json.loads((resume/'manifest.json').read_text())
        if manifest['config_hash'] != object_hash(effective) or manifest['inputs'] != binding:
            raise ValueError('Direct-router resume input/config mismatch')
        for name in ('model.pt', 'optimizer.pt'):
            if sha256(resume/name) != manifest['files'][name]:
                raise ValueError('Direct-router checkpoint modified')
        saved = torch.load(resume/'optimizer.pt', map_location=device, weights_only=True)
        optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state, trace, start_step = saved['rng'], saved['trace'], saved['step']
        inputs += [resume/'model.pt', resume/'optimizer.pt', resume/'manifest.json']
    with Run(output, stage='GFG_direct_distribution_router_training', kind='engineering',
             config=effective, inputs=inputs, seed=seed) as run:
        for step in range(start_step, config['steps']):
            label = labels[step % len(labels)]
            si = torch.tensor(rng.choice(source_groups[label], config['source_batch'], replace=True), device=device)
            ti = torch.tensor(rng.choice(target_groups[label], config['target_batch'], replace=True), device=device)
            r, _, _ = model.dynamics(z[si], us[si])
            native = model.dynamics.last_native_loss
            if arm == 'static':
                r = torch.zeros_like(r)
            q = model.router.logits(state[si], r, e[si]).softmax(-1)
            task = mixture_energy(candidates[si], q, y[ti])
            loss = task+config['native_weight']*native
            task_gradient = None
            if step % 50 == 0 and arm != 'static':
                gradient = torch.autograd.grad(task, model.dynamics.core.velocity_encoder.net[-1].weight, retain_graph=True)[0]
                task_gradient = float(gradient.detach().norm())
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite direct-router loss; do not emit a successful result')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            row = dict(step=step, task_loss=float(task.detach()), native_loss=float(native.detach()),
                entropy=float((-(q*q.clamp_min(1e-12).log()).sum(-1)).mean().detach()), task_gradient=task_gradient)
            trace.append(row)
            if step % 10 == 0:
                save_json(run.directory/'status.json', dict(status='running', arm=arm, seed=seed, **row))
            if (step+1) % 200 == 0:
                checkpoint = run.directory/'checkpoints'/f'step_{step+1:07d}'
                checkpoint.mkdir(parents=True)
                save_full_checkpoint(checkpoint/'model.pt', model, metadata=metadata)
                with (checkpoint/'optimizer.pt').open('xb') as stream:
                    torch.save(dict(optimizer=optimizer.state_dict(), rng=rng.bit_generator.state,
                        step=step+1, trace=trace), stream)
                save_json(checkpoint/'manifest.json', dict(inputs=binding, config_hash=object_hash(effective),
                    files={name: sha256(checkpoint/name) for name in ('model.pt', 'optimizer.pt')}))
        save_full_checkpoint(run.directory/'model.pt', model, metadata=metadata)
        save_csv(run.directory/'training_trace.csv', trace)
        result = dict(status='GFG_DIRECT_ROUTER_TRAINED', arm=arm, seed=seed, steps=len(trace),
            task_gradient_max=max((r['task_gradient'] for r in trace if r['task_gradient'] is not None), default=0),
            frozen_fields=True, individual_fate_truth_used=False, configuration=effective)
        save_json(run.directory/'summary.json', result)
        save_json(run.directory/'status.json', dict(status='complete'))
        (run.directory/'RESULTS.md').write_text('# Direct unpaired-distribution router training\n\n'+json.dumps(result, indent=2))
    return result


def predict_direct(base, seed, arm, geometry, checkpoint, output):
    enforce_gpu_policy(base['device'])
    fold, device = Path(base['fold']), base['device']
    gene_path = Path('outputs/veloroute_gfg_inputs_20260914/validation_gene_source.npz')
    values, source, sm = read_gene_input(gene_path, fold/'validation_source.npz')
    data = load_geometry(geometry, 'validation', source)
    values = training_gene_us(values, source, arm, seed)
    model, metadata = load_full_checkpoint(checkpoint, map_location=device)
    if metadata['arm'] != arm or metadata['seed'] != seed or metadata['transform_hash'] != sm['transform_hash']:
        raise ValueError('Direct prediction identity/transform mismatch')
    model.eval()
    transform = FrozenSplicingTransform.load(fold/'transform.npz')
    with Run(output, stage='GFG_direct_router_source_only_prediction', kind='engineering',
        config=dict(arm=arm, seed=seed, future_target_read=False),
        inputs=[checkpoint, Path(geometry)/'validation.npz', gene_path], seed=seed) as run:
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(values), 8):
                ix = slice(start, start+8)
                z, us, state, e = [torch.tensor(v[ix], dtype=torch.float32, device=device)
                    for v in (source['z'], values, data['state'], data['embedding'])]
                r, _, _ = model.dynamics(z, us)
                if arm == 'static':
                    r = torch.zeros_like(r)
                probabilities.append(model.router.logits(state, r, e).softmax(-1).cpu().numpy())
        q = np.concatenate(probabilities)
        modes = (np.random.default_rng(seed).random(len(q)) >= q[:, 0]).astype('int64')
        endpoint = data['candidates'][np.arange(len(q)), modes]
        with (run.directory/'predictions.npz').open('xb') as stream:
            np.savez_compressed(stream, z=endpoint, candidates=data['candidates'], q=q, modes=modes,
                conditions=source['conditions'], source_ids=source['cell_ids'],
                gene_logspliced=transform.decode(endpoint).astype('float32'),
                gene_ids=np.asarray(transform.gene_ids)[transform.selected])
        save_json(run.directory/'prediction_manifest.json', dict(prediction_sha256=sha256(run.directory/'predictions.npz'),
            transform_hash=sm['transform_hash'], role='validation', kind=sm['kind'], seed=seed,
            task='ER-short', velocity_arm=arm, future_target_read=False, gene_space_decoded=True,
            exact_mixture_candidates_saved=True))
        (run.directory/'RESULTS.md').write_text('# Frozen source-only direct-router prediction\n\nNo development target was read.\n')
