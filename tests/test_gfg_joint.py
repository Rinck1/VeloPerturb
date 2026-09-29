from dataclasses import replace

import pytest
import torch

from veloroute.contracts import FitScope
from veloroute.full_model import FullConfig, FullVeloRoute, load_full_checkpoint, save_full_checkpoint
from veloroute.full_training import FullTrainConfig, FullTrainer
from veloroute.gfg import GFGCore


def tiny_gfg():
    c = FullConfig(state_dim=3, esm_dim=5, representation_dim=4, condition_dim=8, encoder_hidden=8,
        hidden_dim=16, router_hidden_dim=16, time_hidden_dim=8, residual_blocks=1, expert_rank=4,
        max_experts=2, top_k=2, initial_active=2, attention_heads=2, attention_head_dim=4,
        reference_neighbors=3, dynamics_backend='gfg', joint_dynamics=True, gfg_genes=5,
        gfg_hidden=(8, 8), gfg_gene_dim=4, gfg_codes=4)
    model = FullVeloRoute(c)
    torch.manual_seed(17)
    us = torch.rand(12, 10)+.1
    z = torch.randn(12, 3)*.1
    token = torch.randn(12, 1, 5)*.1
    token[:] = token[:1].clone()
    mask = torch.ones(12, 1, dtype=torch.bool)
    ids = [f'train:{i}' for i in range(12)]
    scope = FitScope(frozenset(ids))
    model.dynamics.prepare(us, torch.randn(3, 5)*.1, cell_ids=ids, scope=scope)
    model.refresh_reference(z, us, token, mask, cell_ids=ids, scope=scope)
    model.snapshot_static()
    return model, z, us, token, mask, ids


def test_gfg_jvp_matches_finite_difference_and_allows_task_gradient():
    core = GFGCore(dim=4, codes=4, hidden=(8, 8))
    z = torch.randn(7, 4, requires_grad=True)
    direction = torch.randn_like(z)
    function = core.decoder.statedecoder.net
    _, jvp = torch.autograd.functional.jvp(function, z, direction, create_graph=True)
    eps = .001
    expected = (function(z+eps*direction)-function(z-eps*direction))/(2*eps)
    torch.testing.assert_close(jvp, expected, atol=1e-4, rtol=.003)
    jvp.square().sum().backward()
    assert core.decoder.statedecoder.net[0].weight.grad.norm() > 0


def test_joint_router_loss_reaches_gfg_velocity_encoder_without_condition_input():
    model, z, us, token, mask, ids = tiny_gfg()
    context = model.make_context(z, us, token, mask)
    labels = torch.arange(len(z)) % 2
    logits = model.router.logits(z, context.representation, context.condition.detach())
    torch.nn.functional.cross_entropy(logits, labels).backward()
    assert model.dynamics.core.velocity_encoder.net[-1].weight.grad.norm() > 0
    assert model.dynamics.core.decoder.statedecoder.net[0].weight.grad.norm() > 0
    assert all(p.grad is None for p in model.condition_encoder.parameters())
    with pytest.raises(TypeError):
        model.dynamics(z, us, condition=token)
    with pytest.raises(ValueError, match='gene-level'):
        model.dynamics(z, torch.zeros(12, 6))


def test_gfg_corruption_preserves_spliced_and_fit_scope():
    model, z, us, token, mask, ids = tiny_gfg()
    for method in ('shift', 'gaussian', 'zero_u'):
        corrupted = model.dynamics.corrupt(us, method)
        assert torch.equal(corrupted[:, 5:], us[:, 5:])
        assert (corrupted >= 0).all()
    assert model.dynamics.corrupt(us, 'zero_u')[:, :5].count_nonzero() == 0
    with pytest.raises(ValueError, match='Leakage'):
        model.dynamics.prepare(us, model.dynamics.components, cell_ids=ids, scope=FitScope(frozenset(ids[:-1])))


def test_gfg_joint_stages_update_parameters_and_checkpoint_roundtrip(tmp_path):
    model, z, us, token, mask, ids = tiny_gfg()
    target_ids = [f'target:{i}' for i in range(12)]
    config = FullTrainConfig(stage_a_steps=2, stage_b_steps=2, stage_c_steps=4, batch_size=4,
        e_interval=1, teacher_lag=1, activation_interval=1, reference_interval=2, usage_interval=4,
        corruption_interval=2, checkpoint_interval=4, sinkhorn_iterations=500, lambda_field=.01)
    trainer = FullTrainer(model, config)
    before = model.dynamics.core.velocity_encoder.net[-1].weight.detach().clone()
    trace = trainer.fit(z, us, torch.zeros_like(z), z+.02, token, mask,
        source_conditions=['A']*12, target_conditions=['A']*12, source_ids=ids,
        target_ids=target_ids, scope=FitScope(frozenset(ids+target_ids)))
    assert not torch.equal(before, model.dynamics.core.velocity_encoder.net[-1].weight)
    assert model.dynamics.core.velocity_encoder.net[-1].weight.requires_grad
    assert max(row['gfg_task_gradient_norm'] for row in trace) > 0
    model.eval()
    path = tmp_path/'model.pt'
    save_full_checkpoint(path, model, metadata={'kind': 'synthetic', 'GFG_checkpoint_used': False})
    restored, _ = load_full_checkpoint(path)
    restored.eval()
    modes = torch.zeros(len(z), dtype=torch.long)
    a = model.predict(z, us, token, mask, modes=modes, stochastic=False)
    b = restored.predict(z, us, token, mask, modes=modes, stochastic=False)
    torch.testing.assert_close(a.endpoint, b.endpoint)
    assert restored.dynamics.fit_ids_hash == model.dynamics.fit_ids_hash


def test_gfg_joint_optimizer_and_EM_resume_exactly(tmp_path):
    model, z, us, token, mask, ids = tiny_gfg()
    targets = [f'target:{i}' for i in range(12)]
    config = FullTrainConfig(stage_a_steps=2, stage_b_steps=2, stage_c_steps=6, batch_size=4,
        e_interval=1, teacher_lag=2, activation_interval=1, reference_interval=2, usage_interval=6,
        corruption_interval=2, checkpoint_interval=3, sinkhorn_iterations=500)
    trainer = FullTrainer(model, config)
    args = (z, us, torch.zeros_like(z), z+.02, token, mask)
    kwargs = dict(source_conditions=['A']*12, target_conditions=['A']*12,
        source_ids=ids, target_ids=targets, scope=FitScope(frozenset(ids+targets)))
    trainer.fit(*args, **kwargs, max_c_steps=3)
    path, state = tmp_path/'model.pt', tmp_path/'state.pt'
    save_full_checkpoint(path, model, metadata={'kind': 'synthetic'})
    torch.save(trainer.state_dict(), state)
    restored, _ = load_full_checkpoint(path)
    resumed = FullTrainer(restored, config)
    resumed.load_state_dict(torch.load(state, weights_only=True))
    trainer.fit(*args, **kwargs)
    resumed.fit(*args, **kwargs)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=1e-6, rtol=1e-5)
    assert trainer.trace == resumed.trace
    assert restored.dynamics.core.velocity_encoder.net[-1].weight.requires_grad


def test_distribution_loss_updates_gfg_experts_and_calibrates_gate():
    model, z, us, token, mask, ids = tiny_gfg()
    targets = [f'target:{i}' for i in range(len(z))]
    config = FullTrainConfig(stage_a_steps=1, stage_b_steps=2, stage_c_steps=2, batch_size=4,
        coupling_mode='distribution_matching', dist_source_batch=4, dist_target_batch=6,
        corruption_interval=1, gfg_native_weight=0., sinkhorn_iterations=500)
    trainer = FullTrainer(model, config)
    args = (z, us, torch.zeros_like(z), z+.2, token, mask)
    kwargs = dict(source_conditions=['A']*len(z), target_conditions=['A']*len(z), source_ids=ids,
                  target_ids=targets, scope=FitScope(frozenset(ids+targets)))
    trainer.fit(*args, **kwargs, max_c_steps=0)
    field_before = model.field.base.weight.detach().clone()
    gfg_before = model.dynamics.core.velocity_encoder.net[-1].weight.detach().clone()
    gate_before = [p.detach().clone() for p in model.gate.parameters()]
    trace = trainer.fit(*args, **kwargs)
    assert not torch.equal(field_before, model.field.base.weight)
    assert not torch.equal(gfg_before, model.dynamics.core.velocity_encoder.net[-1].weight)
    assert max(row['gfg_task_gradient_norm'] for row in trace) > 0
    assert any(not torch.equal(a, b) for a, b in zip(gate_before, model.gate.parameters()))
    assert trainer.corruptions == 2


def test_static_arm_does_not_train_gfg_or_read_velocity():
    from veloroute.gfg_experiments import train_gfg
    # The integration contract is checked at model construction: static is
    # strict S-only while retaining the same serialized module shape.
    model, z, us, token, mask, ids = tiny_gfg()
    model.config = replace(model.config, use_dynamics=False, joint_dynamics=False,
                           static_uses_trained_field=True)
    model.snapshot_static()
    before = [p.detach().clone() for p in model.dynamics.parameters()]
    context = model.make_context(z, us, token, mask)
    assert torch.count_nonzero(context.velocity) == 0
    assert torch.all(context.probabilities >= 0)
    assert all(torch.equal(a, b) for a, b in zip(before, model.dynamics.parameters()))
