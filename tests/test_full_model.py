from dataclasses import asdict, replace

import pytest
import torch

from veloroute.contracts import FitScope
from veloroute.full_components import (DynamicsEncoder, FrozenReferenceField, ModeController, ReliabilityGate,
                                      SetConditionEncoder, StateEncoder, corrupt_source)
from veloroute.full_model import FullConfig, FullVeloRoute, load_full_checkpoint, save_full_checkpoint
from veloroute.full_training import (DelayedLabels, FullTrainConfig, FullTrainer, field_consistency_loss,
                                     full_responsibilities)


def tiny(**updates):
    config = FullConfig(state_dim=3, esm_dim=5, representation_dim=4, condition_dim=8, encoder_hidden=16,
                        hidden_dim=16, router_hidden_dim=16, time_hidden_dim=8, residual_blocks=1,
                        expert_rank=4, max_experts=3, attention_heads=2, attention_head_dim=4, reference_neighbors=3)
    return FullVeloRoute(replace(config, **updates))


def fixture(n=12):
    torch.manual_seed(4)
    s, u = torch.randn(n, 3)*.1, torch.randn(n, 3)*.1
    v, y = torch.randn(n, 3)*.1, s+torch.randn_like(s)*.05
    token = torch.randn(1, 2, 5).expand(n, -1, -1).clone()
    mask = torch.ones(n, 2, dtype=torch.bool)
    ids = [f'train_source_{i}' for i in range(n)]
    target_ids = [f'train_target_{i}' for i in range(n)]
    return s, u, v, y, token, mask, ids, target_ids


def ready_model(**kwargs):
    model = tiny(**kwargs)
    s, u, v, y, token, mask, ids, target_ids = fixture()
    model.refresh_reference(s, u, token, mask, cell_ids=ids, scope=FitScope(frozenset(ids)))
    model.snapshot_static()
    return model, (s, u, v, y, token, mask, ids, target_ids)


def test_gate_support_cannot_claim_unobserved_training_days():
    with pytest.raises(ValueError, match='actual training interval'):
        FullTrainer(tiny(support_start=2.), FullTrainConfig(time_start=4., time_end=5.))
    trainer = FullTrainer(tiny(support_start=4.), FullTrainConfig(time_start=4., time_end=5.))
    assert trainer.model.config.support_start == 4.


def test_state_encoder_is_identity_initialized_and_reconstructs():
    s, *_ = fixture()
    encoder = StateEncoder(3, 16)
    torch.testing.assert_close(encoder(s), s)
    loss = encoder.reconstruction_loss(s, generator=torch.Generator().manual_seed(2))
    loss.backward()
    assert encoder.network[-1].weight.grad.abs().sum() > 0


def test_dynamics_has_no_condition_argument_and_scalar_uncertainty():
    s, u, v, *_ = fixture()
    encoder = DynamicsEncoder(3, 4, 16)
    r, predicted, rho = encoder(s, u)
    assert r.shape == (12, 4) and predicted.shape == (12, 3) and rho.shape == (12,)
    assert (rho > 0).all()
    with pytest.raises(TypeError):
        encoder(s, u, condition=torch.ones(12, 1))
    encoder.reference_loss(s, u, v).backward()
    assert encoder.velocity_head.weight.grad is not None and encoder.uncertainty_head.weight.grad is not None


def test_set_attention_permutation_padding_and_gradients():
    encoder = SetConditionEncoder(5, 8, 2, 4)
    tokens = torch.randn(4, 3, 5)
    mask = torch.tensor([[1, 1, 0]]*4, dtype=torch.bool)
    reference = encoder(tokens, mask)
    torch.testing.assert_close(reference, encoder(tokens[:, [1, 0, 2]], mask))
    tokens[:, 2] = torch.nan
    torch.testing.assert_close(reference, encoder(tokens, mask))
    reference.square().mean().backward()
    assert encoder.qkv.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        encoder(tokens, torch.zeros_like(mask))


def test_reference_scope_and_no_gradient_to_queries():
    ref = FrozenReferenceField(3, 3)
    s, u, v, y, token, mask, ids, _ = fixture()
    with pytest.raises(ValueError, match='Leakage'):
        ref.fit(s, v, cell_ids=ids, scope=FitScope(frozenset(ids[:-1])))
    ref.fit(s, v, cell_ids=ids, scope=FitScope(frozenset(ids)))
    value, distance = ref(s.requires_grad_())
    assert value.shape == s.shape and distance.shape == (len(s),)
    assert not value.requires_grad and not distance.requires_grad


def test_gate_gradient_isolation_monotonicity_and_fail_closed():
    gate = ReliabilityGate()
    features = torch.rand(10, 6, requires_grad=True)
    gate.calibration_loss(features, torch.tensor([1, 0]*5)).backward()
    assert features.grad is None
    assert gate.network[0].weight.grad is not None
    valid = torch.ones(10, dtype=torch.bool)
    noisier = features.detach().clone(); noisier[:, 0] += 10
    assert (gate(noisier, valid) <= gate(features, valid)).all()
    assert torch.count_nonzero(gate(features, ~valid)) == 0


def test_adaptive_modes_require_complete_condition_review_and_never_prune_last():
    control = ModeController(3, initial=1, patience=2, minimum_usage=.1)
    assert control.activate_next() == 1 and control.activate_next() == 2
    p = torch.tensor([[1., 0., 0.]]*6)
    with pytest.raises(ValueError, match='complete'):
        control.review(p, ['A']*6, ['A', 'B'])
    assert control.review(p, ['A']*3+['B']*3, ['A', 'B']) == []
    assert control.review(p, ['A']*3+['B']*3, ['A', 'B']) == [1, 2]
    assert control.active.tolist() == [True, False, False]
    assert control.activate_next() is None


def test_delayed_labels_do_not_train_on_current_gamma():
    history = DelayedLabels(3)
    value = torch.ones(1, requires_grad=True)
    for step in range(3):
        assert history.push({'step': step, 'labels': value}) is None
    old = history.push({'step': 3, 'labels': value})
    assert old['step'] == 0 and not old['labels'].requires_grad


def test_full_coupling_mode_mask_marginals_and_along_path_cost():
    model, (s, u, v, y, token, mask, ids, _) = ready_model()
    model.mode_controller.activate_next()
    context = model.make_context(s, u, token, mask)
    result = full_responsibilities(model, context, y, FullTrainConfig(sinkhorn_iterations=500))
    assert not result['gamma'].requires_grad and result['gamma'].shape == (12, 12, 3)
    torch.testing.assert_close(result['gamma'].sum((1, 2)), torch.full((12,), 1/12), atol=1e-4, rtol=1e-4)
    assert torch.count_nonzero(result['gamma'][:, :, 2]) == 0
    assert torch.isfinite(result['dynamic_cost']).all()


def test_field_constraint_only_updates_shared_backbone():
    model, (s, u, v, y, token, mask, ids, _) = ready_model()
    context = model.make_context(s, u, token, mask)
    field_consistency_loss(model, s, y, context.condition, FullTrainConfig()).backward()
    assert model.field.base.weight.grad is not None
    for module in (model.router, model.dynamics, model.state_encoder, model.condition_encoder, model.gate, model.intrinsic):
        assert all(p.grad is None for p in module.parameters())
    assert all(p.grad is None for expert in model.field.experts for p in expert.parameters())


def test_full_prediction_locked_modes_noise_attribution_and_future_rejection(tmp_path):
    model, (s, u, v, y, token, mask, ids, _) = ready_model(initial_active=3)
    modes = torch.arange(len(s)) % 3
    a = model.predict(s, u, token, mask, modes=modes, generator=torch.Generator().manual_seed(7),
                      return_trajectory=True, snapshots=(4.3, 4.8), force_gate=1.)
    b = model.predict(s, u, token, mask, modes=modes, generator=torch.Generator().manual_seed(7), force_gate=1.,
                      return_trajectory=True, snapshots=(4.3, 4.8))
    torch.testing.assert_close(a.endpoint, b.endpoint)
    torch.testing.assert_close(a.endpoint-model.encode_state(s), sum(a.contributions.values()), atol=1e-6, rtol=1e-5)
    assert 4.3 in a.times and 4.8 in a.times
    assert torch.count_nonzero(a.contributions['within_mode_noise']) > 0
    with pytest.raises(TypeError):
        model.predict(s, u, token, mask, future_expression=y)
    path = tmp_path/'full.pt'
    save_full_checkpoint(path, model, metadata={'kind': 'synthetic'})
    restored, _ = load_full_checkpoint(path)
    again = restored.predict(s, u, token, mask, modes=modes, generator=torch.Generator().manual_seed(7),
                              return_trajectory=True, snapshots=(4.3, 4.8), force_gate=1.)
    torch.testing.assert_close(a.endpoint, again.endpoint)
    assert not any('teacher' in key for key in restored.state_dict())


def test_gate_zero_exact_frozen_baseline_and_time_attenuation():
    model, (s, u, v, y, token, mask, ids, _) = ready_model()
    static_prediction = model.predict(s, u, token, mask, force_gate=0.)
    assert torch.count_nonzero(static_prediction.gate) == 0
    before = static_prediction.endpoint
    with torch.no_grad():
        for parameter in list(model.field.parameters())+list(model.condition_encoder.parameters())+list(model.intrinsic.parameters()):
            parameter.add_(torch.randn_like(parameter)*.1)
    after = model.predict(s, u*10, token, mask, force_gate=0.).endpoint
    torch.testing.assert_close(before, after)
    context = model.make_context(s, u, token, mask)
    assert (model.gate_value(context, 7.) <= model.gate_value(context, 6.)).all()


@pytest.mark.parametrize('method', ['shift', 'gaussian', 'zero_u'])
def test_corruption_definitions(method):
    s, u, v, *_ = fixture()
    bad_u, bad_v = corrupt_source(s, u, v, method, generator=torch.Generator().manual_seed(1))
    if method == 'zero_u':
        assert bad_v is None and torch.count_nonzero(bad_u) == 0
    else:
        assert bad_v.shape == v.shape and not torch.equal(v, bad_v)


def train_args():
    s, u, v, y, token, mask, ids, targets = fixture()
    scope = FitScope(frozenset(ids+targets))
    return (s, u, v, y, token, mask), dict(source_conditions=['A']*12, target_conditions=['A']*12,
              source_ids=ids, target_ids=targets, scope=scope)


def tiny_training_config():
    return FullTrainConfig(stage_a_steps=2, stage_b_steps=2, stage_c_steps=8, batch_size=6,
        e_interval=1, teacher_lag=3, activation_interval=1, reference_interval=4, usage_interval=8,
        corruption_interval=2, checkpoint_interval=4, lambda_field=.01, sinkhorn_iterations=500)


def test_all_stages_train_and_resume_exactly(tmp_path):
    args, kwargs = train_args()
    torch.manual_seed(123)
    model = tiny()
    trainer = FullTrainer(model, tiny_training_config())
    trainer.fit(*args, **kwargs, max_c_steps=4)
    checkpoint = tmp_path/'model.pt'
    save_full_checkpoint(checkpoint, model, metadata={'kind': 'synthetic'})
    state_file = tmp_path/'trainer.pt'
    torch.save(trainer.state_dict(), state_file)
    restored, _ = load_full_checkpoint(checkpoint)
    resumed = FullTrainer(restored, tiny_training_config())
    resumed.load_state_dict(torch.load(state_file, weights_only=True))
    trainer.fit(*args, **kwargs)
    resumed.fit(*args, **kwargs)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=1e-6, rtol=1e-5)
    assert trainer.teacher_updates > 0 and trainer.corruptions == 4 and trainer.e_round == 8
    assert {row['stage'] for row in trainer.trace} == {'A', 'B', 'C+D'}
    assert bool(model.fallback_ready) and int(model.active_modes.sum()) >= 1
    assert all(p.grad is None for p in model.dynamics.parameters())
    assert any(p.grad is not None for p in model.gate.parameters())
    assert any(p.grad is not None for p in model.intrinsic.parameters())
    assert model.noise_raw.grad is not None
    assert trainer.trace == resumed.trace


def test_training_integrator_matches_inference_and_reaches_expert_weights():
    model, (s, u, v, y, token, mask, ids, _) = ready_model(initial_active=3, use_gate=False)
    modes = torch.arange(len(s)) % 3
    context = model.make_context(s, u, token, mask)
    train_prediction = model.integrate_context(context, modes=modes, stochastic=False)
    inference = model.predict(s, u, token, mask, modes=modes, stochastic=False)
    torch.testing.assert_close(train_prediction.endpoint, inference.endpoint)
    assert train_prediction.endpoint.requires_grad and not inference.endpoint.requires_grad
    (train_prediction.endpoint-y).square().mean().backward()
    assert model.field.base.weight.grad.abs().sum() > 0
    assert all(expert.up.weight.grad.abs().sum() > 0 for expert in model.field.experts)
    assert all(p.grad is None for p in model.gate.parameters())


def test_static_router_uses_trained_experts_and_is_independent_of_velocity():
    model, (s, u, v, y, token, mask, ids, _) = ready_model(
        initial_active=3, use_dynamics=False, static_uses_trained_field=True)
    modes = torch.zeros(len(s), dtype=torch.long)
    before = model.predict(s, u, token, mask, modes=modes, stochastic=False).endpoint
    with torch.no_grad():
        model.field.experts[0].up.weight.fill_(.5)
    after = model.predict(s, u, token, mask, modes=modes, stochastic=False).endpoint
    assert not torch.allclose(before, after)
    shuffled = model.predict(s, u.flip(0), token, mask, modes=modes, stochastic=False)
    torch.testing.assert_close(after, shuffled.endpoint)
    assert torch.all(shuffled.gate == 1)


def test_distribution_training_updates_field_and_resumes_exactly(tmp_path):
    args, kwargs = train_args()
    torch.manual_seed(123)
    model = tiny(use_dynamics=False, static_uses_trained_field=True)
    config = replace(tiny_training_config(), coupling_mode='distribution_matching',
                     dist_source_batch=6, dist_target_batch=8, stage_c_steps=4)
    trainer = FullTrainer(model, config)
    trainer.fit(*args, **kwargs, max_c_steps=0)
    before = model.field.base.weight.detach().clone()
    trainer.fit(*args, **kwargs, max_c_steps=2)
    assert not torch.equal(before, model.field.base.weight)
    assert model.noise_raw.grad.abs().sum() > 0
    path = tmp_path/'dist.pt'
    save_full_checkpoint(path, model, metadata={'kind': 'synthetic'})
    state = tmp_path/'trainer.pt'
    torch.save(trainer.state_dict(), state)
    restored, _ = load_full_checkpoint(path)
    resumed = FullTrainer(restored, config)
    resumed.load_state_dict(torch.load(state, weights_only=True))
    trainer.fit(*args, **kwargs)
    resumed.fit(*args, **kwargs)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=1e-6, rtol=1e-5)
    assert trainer.trace == resumed.trace
    assert sum(row['stage'] == 'C-dist' for row in trainer.trace) == 4
