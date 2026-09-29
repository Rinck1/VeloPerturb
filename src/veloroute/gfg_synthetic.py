"""Unpaired group-composition tests with real GFG JVP, no cell fate supervision.

Within-identical-S permutation preserves the source distribution; that arm is
expected to tie on marginal prediction. An interaction system separately tests
whether source S/velocity correspondence controls branch proportions.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from .artifacts import Run, load_config, save_csv, save_json
from .contracts import FitScope
from .full_model import FullConfig
from .gfg import GFGDynamics
from .metrics import energy_distance
from .model import ModelConfig, SourceRouter


def groups(system, prevalences, cells, seed, genes=12):
    rng = np.random.default_rng(seed)
    us, state, future, modes = [], [], [], []
    pattern = np.tile([1., -1.], genes//2)
    for p in prevalences:
        # Balanced source states and velocity marginals; correlation sets prevalence.
        h = np.tile([-1., 1.], cells//2)
        fate = rng.random(cells) < p
        w = np.where(fate, 1., -1.)
        if system == 'state_velocity_interaction':
            w = w*h
        if system == 'position_branch':
            h = np.where(fate, 1., -1.)
            w = rng.choice([-1., 1.], cells)
        if system == 'same_S_velocity_branch':
            h[:] = 0
        s = 5.+.5*h[:, None]*pattern[None]
        u = 5.+2*w[:, None]*pattern[None]+rng.normal(0, .03, (cells, genes))
        z = np.stack((h*.1, np.zeros(cells)), 1)
        if system == 'no_increment':
            fate = np.zeros(cells, bool)
        # Future samples are independently drawn: no actual source-target pairing.
        future_fate = rng.random(cells) < (0 if system == 'no_increment' else p)
        y = np.stack((np.where(future_fate, 1., -1.)*1.5, np.zeros(cells)), 1)
        y += rng.normal(0, .04, y.shape)
        us.append(np.concatenate((u, s), 1).astype('float32'))
        state.append(z.astype('float32'))
        future.append(y.astype('float32'))
        modes.append(fate.astype('int64'))
    return np.stack(us), np.stack(state), np.stack(future), np.stack(modes)


def permute_group_u(values, seed):
    """Destroy S/U correspondence BEFORE GFG; preserve each group's U multiset."""
    result = values.copy()
    rng = np.random.default_rng(seed)
    genes = values.shape[-1]//2
    for group in range(len(values)):
        result[group, :, :genes] = values[group, rng.permutation(values.shape[1]), :genes]
    return result


def run_gfg_synthetic(config, output, *, shuffle_source='representation', resume_directory=None):
    if shuffle_source not in ('representation', 'input_U'):
        raise ValueError('Unknown intervention position')
    design = config['synthetic']
    device = config['device']
    torch.set_num_threads(config['cpu_threads'])
    systems = ['same_S_velocity_branch', 'state_velocity_interaction', 'position_branch', 'no_increment']
    arms = ['static', 'gfg_frozen', 'gfg_joint', 'gfg_joint_shuffled']
    executed = {**design, 'systems': systems, 'arms': arms, 'shuffle': shuffle_source,
        'native_weight': .1, 'router_lr': .001, 'gfg_lr': .0001,
        'same_S_shuffle_expected': 'marginal_invariance_NOT_a_required_real_advantage',
        'covariate_interaction_shuffle_expected': ('correspondence_degradation' if shuffle_source == 'input_U'
            else 'no_required_degradation_after_GFG_already_encodes_joint_SU'),
        'prototype_supervision': 'training_target_kmeans_only_no_true_cell_modes'}
    resume_directory = Path(resume_directory) if resume_directory else None
    input_paths = [config['gfg_checkpoint']]
    if resume_directory:
        original = load_config(resume_directory/'config.yaml')
        if shuffle_source != 'input_U' or original != executed:
            raise ValueError('Synthetic recovery must preserve the exact input-U experiment protocol')
        input_paths += sorted((resume_directory/'checkpoints').glob('*.pt'))
        input_paths += [resume_directory/'config.yaml', resume_directory/'provenance.json']
    with Run(output, stage='GFG_unpaired_advantage_synthetic', kind='synthetic', config=executed,
             inputs=input_paths, seed=design['seeds']) as run:
        rows, trace = [], []
        reused = []
        for system in systems:
            for seed in design['seeds']:
                torch.manual_seed(seed)
                train_us, train_z, train_y, _ = groups(system, design['train_prevalences'], design['cells_per_group'], seed+100)
                test_us, test_z, test_y, true_modes = groups(system, design['test_prevalences'], design['cells_per_group'], seed+200)
                k = 1 if system == 'no_increment' else 2
                centers = KMeans(k, n_init=10, random_state=seed).fit(train_y.reshape(-1, 2)).cluster_centers_
                centers = centers[np.argsort(centers[:, 0])]
                target_labels = np.argmin(((train_y[:, :, None]-centers[None, None])**2).sum(-1), -1)
                target_pi = torch.tensor(np.stack([(target_labels == index).mean(1) for index in range(k)], 1),
                                         dtype=torch.float32, device=device)
                c = FullConfig(state_dim=2, representation_dim=16, gfg_genes=12)
                dynamics = GFGDynamics(c).to(device)
                dynamics.core.load_pretrained(config['gfg_checkpoint'])
                flat = torch.tensor(train_us.reshape(-1, 24), device=device)
                ids = [f'toy:{i}' for i in range(len(flat))]
                components = torch.stack((torch.tensor(np.tile([1., -1.], 6)), torch.tensor(np.tile([1., 1., -1., -1.], 3)))).float().to(device)/12**.5
                dynamics.prepare(flat, components, cell_ids=ids, scope=FitScope(frozenset(ids)))
                with torch.no_grad():
                    _, velocity, _ = dynamics(torch.tensor(train_z.reshape(-1, 2), device=device), flat)
                    dynamics.velocity_scale.copy_(velocity.square().mean().sqrt().clamp_min(1e-6))
                # Forward-cache tensors are transient and must not be deep-copied.
                dynamics.last_native_loss = None
                router = SourceRouter(ModelConfig(state_dim=2, velocity_dim=16, condition_dim=2,
                    hidden_dim=32, router_hidden_dim=64, time_hidden_dim=8, residual_blocks=1,
                    expert_rank=4, n_experts=k, top_k=k)).to(device)
                initial_dynamics, initial_router = copy.deepcopy(dynamics.state_dict()), copy.deepcopy(router.state_dict())
                for arm in arms:
                    arm_train_us, arm_test_us = train_us, test_us
                    if arm == 'gfg_joint_shuffled' and shuffle_source == 'input_U':
                        arm_train_us = permute_group_u(train_us, seed+891)
                        arm_test_us = permute_group_u(test_us, seed+982)
                    dynamics.load_state_dict(initial_dynamics)
                    router.load_state_dict(initial_router)
                    recovered = False
                    if resume_directory:
                        previous = resume_directory/'checkpoints'/f'{system}_{arm}_seed{seed}.pt'
                        if previous.exists():
                            record = torch.load(previous, map_location=device, weights_only=True)
                            if (record['system'], record['arm'], record['seed'], record['intervention']) != (system, arm, seed, shuffle_source):
                                raise ValueError('Synthetic recovery checkpoint identity mismatch')
                            np.testing.assert_allclose(record['prototypes'], centers)
                            dynamics.load_state_dict(record['GFG'], strict=True)
                            router.load_state_dict(record['router'], strict=True)
                            reused.append(str(previous))
                            recovered = True
                    dynamics.requires_grad_(arm != 'gfg_frozen')
                    for encoder in (dynamics.core.manifold_encoder, dynamics.core.velocity_encoder):
                        encoder.attention_layer.requires_grad_(False)
                    optimizer = torch.optim.AdamW([
                        {'params': router.parameters(), 'lr': .001},
                        {'params': [p for p in dynamics.parameters() if p.requires_grad], 'lr': .0001}])
                    rng = np.random.default_rng(seed)
                    for step in range(0 if recovered else design['steps']):
                        # All groups contribute equally; their identity is not a model input.
                        indices = rng.integers(0, design['cells_per_group'], (4, 24))
                        x = torch.tensor(np.stack([arm_train_us[g, indices[g]] for g in range(4)]).reshape(-1, 24), device=device)
                        z = torch.tensor(np.stack([train_z[g, indices[g]] for g in range(4)]).reshape(-1, 2), device=device)
                        r, velocity, _ = dynamics(z, x)
                        if arm == 'static':
                            r = torch.zeros_like(r)
                        elif arm == 'gfg_joint_shuffled' and shuffle_source == 'representation':
                            permutation = np.concatenate([g*24+rng.permutation(24) for g in range(4)])
                            r = r[torch.tensor(permutation, device=device)]
                        q = router.logits(z, r, torch.zeros(len(z), 2, device=device)).softmax(-1)
                        mean_q = q.reshape(4, 24, k).mean(1)
                        distribution_loss = -(target_pi*mean_q.clamp_min(1e-8).log()).sum(-1).mean()
                        loss = distribution_loss
                        if arm != 'gfg_frozen':
                            loss = loss+.1*dynamics.last_native_loss
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(list(router.parameters())+list(dynamics.parameters()), 1.)
                        optimizer.step()
                        if step % 50 == 0:
                            trace.append(dict(system=system, seed=seed, arm=arm, step=step,
                                group_loss=float(distribution_loss.detach()), native=float(dynamics.last_native_loss.detach())))
                            save_json(run.directory/'status.json', trace[-1])
                    with torch.no_grad():
                        predictions = []
                        for group, p in enumerate(design['test_prevalences']):
                            x, z = torch.tensor(arm_test_us[group], device=device), torch.tensor(test_z[group], device=device)
                            r, _, _ = dynamics(z, x)
                            if arm == 'static':
                                r = torch.zeros_like(r)
                            elif arm == 'gfg_joint_shuffled' and shuffle_source == 'representation':
                                r = r[torch.tensor(rng.permutation(len(z)), device=device)]
                            q = router.logits(z, r, torch.zeros(len(z), 2, device=device)).softmax(-1)
                            pi = q.mean(0).cpu().numpy()
                            ratio_error = abs(float(pi[-1])-(p if k == 2 else 1.))
                            count = 0 if k == 1 else int(round(len(z)*pi[-1]))
                            samples = np.repeat(centers, [len(z)] if k == 1 else [len(z)-count, count], axis=0)
                            row = dict(system=system, seed=seed, arm=arm, true_prevalence=p,
                                predicted_prevalence=float(pi[-1]), ratio_error=ratio_error,
                                endpoint_energy=energy_distance(samples, test_y[group]),
                                routing_accuracy=float((q.argmax(-1).cpu().numpy() == true_modes[group]).mean()))
                            rows.append(row)
                    checkpoints = run.directory/'checkpoints'
                    checkpoints.mkdir(exist_ok=True)
                    with (checkpoints/f'{system}_{arm}_seed{seed}.pt').open('xb') as stream:
                        torch.save(dict(GFG=dynamics.state_dict(), router=router.state_dict(),
                            prototypes=centers.tolist(), kind='synthetic', system=system, seed=seed,
                            arm=arm, intervention=shuffle_source), stream)
                    print(json.dumps(dict(system=system, seed=seed, arm=arm, mean_ratio_error=float(np.mean([r['ratio_error'] for r in rows if r['system']==system and r['seed']==seed and r['arm']==arm])))), flush=True)
        save_csv(run.directory/'metrics.csv', rows)
        save_csv(run.directory/'training_trace.csv', trace)
        means = {system: {arm: float(np.mean([r['ratio_error'] for r in rows if r['system']==system and r['arm']==arm]))
                          for arm in arms} for system in systems}
        checks = dict(interaction_real_vs_static=means['state_velocity_interaction']['static']-means['state_velocity_interaction']['gfg_joint'] >= design['minimum_branch_ratio_error_gain'],
            interaction_real_vs_shuffle=means['state_velocity_interaction']['gfg_joint_shuffled']-means['state_velocity_interaction']['gfg_joint'] >= design['minimum_branch_ratio_error_gain'],
            position_control=means['position_branch']['gfg_joint']-means['position_branch']['static'] <= design['maximum_static_control_degradation'],
            no_increment_control=means['no_increment']['gfg_joint'] == 0)
        result = dict(status='GFG_SYNTHETIC_ADVANTAGE_PASS' if all(checks.values()) else 'GFG_SYNTHETIC_ADVANTAGE_NOT_ESTABLISHED',
                      mean_ratio_error=means, checks=checks, real_data_evidence=False,
                      true_individual_fates_used_for_training=False, expected_same_S_shuffle_invariance=True,
                      recovered_checkpoints=len(reused), trace_scope='newly_trained_arms_only' if reused else 'all_arms')
        save_json(run.directory/'recovered_checkpoints.json', reused)
        save_json(run.directory/'summary.json', result)
        (run.directory/'RESULTS.md').write_text('# GFG unpaired composition tests\n\n'+json.dumps(result, indent=2))
    return result
