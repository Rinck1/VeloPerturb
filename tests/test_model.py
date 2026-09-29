from dataclasses import replace

import pytest
import torch

from veloroute.coupling import build_responsibilities, sinkhorn_log, source_mode_cost
from veloroute.model import (ModelConfig, VeloRoute, expert_regression_loss, flow_matching_loss,
                            load_checkpoint, routing_loss, save_checkpoint)


def tiny(**updates):
    cfg = ModelConfig(state_dim=3, velocity_dim=3, condition_dim=2, hidden_dim=16,
                      router_hidden_dim=16, time_hidden_dim=8, residual_blocks=1, expert_rank=4)
    return VeloRoute(replace(cfg, **updates))


def inputs(n=8):
    return torch.randn(n, 3), torch.randn(n, 3), torch.randn(n, 2)


def test_model_shape_and_sparse_router():
    model = tiny(n_experts=4)
    z, v, c = inputs()
    q = model.router(z, v, c)
    assert q.shape == (8, 4)
    assert torch.all((q > 0).sum(1) <= 2)
    torch.testing.assert_close(q.sum(1), torch.ones(8))
    assert model.field(z, 4., c).shape == (8, 4, 3)


def test_k1_and_static_router_ignore_velocity():
    z, v, c = inputs()
    for model in (tiny(n_experts=1, top_k=1), tiny(use_velocity=False)):
        torch.testing.assert_close(model.router(z, v, c), model.router(z, v*100, c))


def test_heads_start_equal_but_can_receive_gradient():
    model = tiny()
    z, v, c = inputs()
    fields = model.field(z, 4., c)
    torch.testing.assert_close(fields[:, 0], fields[:, 1])
    assert all(torch.count_nonzero(h.down.weight) for h in model.field.experts)
    optimizer = torch.optim.SGD(model.field.parameters(), lr=.1)
    responsibilities = torch.nn.functional.one_hot(torch.arange(8) % 2, 2).float()
    loss = expert_regression_loss(model, z, 4., c, torch.randn(8, 3), responsibilities)
    loss.backward()
    assert all(torch.count_nonzero(h.up.weight.grad) for h in model.field.experts)
    optimizer.step()
    assert not torch.allclose(model.field(z, 4., c)[:, 0], model.field(z, 4., c)[:, 1])


def test_fm_does_not_train_router_or_inputs():
    model = tiny()
    z, v, c = [x.requires_grad_() for x in inputs()]
    weights = torch.softmax(torch.randn(8, 2), -1).requires_grad_()
    flow_matching_loss(model, z, z+1, c, weights, t0=4., t1=5.).backward()
    assert all(p.grad is None for p in model.router.parameters())
    assert any(p.grad is not None for p in model.field.parameters())
    assert weights.grad is None and z.grad is None and v.grad is None and c.grad is None


def test_route_loss_does_not_train_fields_or_inputs():
    model = tiny()
    z, v, c = [x.requires_grad_() for x in inputs()]
    weights = torch.softmax(torch.randn(8, 2), -1).requires_grad_()
    routing_loss(model, z, v, c, weights).backward()
    assert all(p.grad is None for p in model.field.parameters())
    assert any(p.grad is not None for p in model.router.parameters())
    assert weights.grad is None and z.grad is None and v.grad is None and c.grad is None


def test_loss_is_not_average_field_regression(monkeypatch):
    model = tiny()
    z, _, c = inputs(2)
    fields = torch.tensor([[[1., 1., 1.], [-1., -1., -1.]]]).expand(2, -1, -1)
    monkeypatch.setattr(model.field, "forward", lambda *args: fields)
    loss = expert_regression_loss(model, z, 4., c, torch.zeros_like(z), torch.full((2, 2), .5))
    assert loss.item() == 1.  # The erroneous loss on the mean field would be zero.


def test_modes_locked_during_rk4(monkeypatch):
    model = tiny()
    z, v, c = inputs(4)
    modes = torch.tensor([0, 1, 1, 0])
    calls, router_calls = [], []
    hook = model.router.register_forward_hook(lambda *args: router_calls.append(1))
    def constant(state, time, condition, selected):
        calls.append(selected.clone())
        return (2*selected-1)[:, None].expand_as(state).float()
    monkeypatch.setattr(model.field, "selected", constant)
    result = model.predict(z, v, c, t0=2., t1=5., max_step=.25, modes=modes, return_trajectory=True)
    hook.remove()
    assert len(router_calls) == 1 and len(calls) == 48 and len(result.trajectory) == 13
    assert all(torch.equal(call, modes) for call in calls)
    torch.testing.assert_close(result.endpoint, z+3*(2*modes-1)[:, None])


def test_prediction_sampling_reproducible():
    model = tiny()
    z, v, c = inputs()
    a = model.predict(z, v, c, generator=torch.Generator().manual_seed(7))
    b = model.predict(z, v, c, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(a.endpoint, b.endpoint)
    assert torch.equal(a.modes, b.modes)


def test_future_not_accepted_at_inference():
    model = tiny()
    z, v, c = inputs()
    with pytest.raises(TypeError):
        model.predict(z, v, c, target_expression=z)


def test_checkpoint_roundtrip_and_no_overwrite(tmp_path):
    model = tiny()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, metadata={"kind": "synthetic"})
    restored, metadata = load_checkpoint(path)
    assert metadata["kind"] == "synthetic"
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, metadata={})


def test_directional_cost_actually_distinguishes_modes():
    velocity = torch.tensor([[1., 0.], [0., 0.]])
    fields = torch.tensor([[[1., 0.], [-1., 0.]], [[1., 0.], [-1., 0.]]])
    torch.testing.assert_close(source_mode_cost(velocity, fields), torch.tensor([[0., 2.], [0., 0.]]))


def test_sinkhorn_rectangular_marginals():
    plan = sinkhorn_log(torch.rand(5, 7), epsilon=.2, iterations=500)
    torch.testing.assert_close(plan.sum(1), torch.full((5,), .2), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(plan.sum(0), torch.full((7,), 1/7), atol=1e-5, rtol=1e-5)


def test_coupling_normalized_and_detached():
    model = tiny()
    z, v, c = inputs()
    c = c[:1].expand_as(c)
    labels = ["TF1"]*8
    result = build_responsibilities(model, z, z+.1, v, c, source_conditions=labels,
                                    target_conditions=labels, lambda_dynamic=.2, iterations=500)
    torch.testing.assert_close(result["source_responsibilities"].sum(1), torch.ones(8))
    assert result["gamma"].shape == (8, 8, 2) and not result["gamma"].requires_grad


def test_coupling_refuses_cross_condition():
    model = tiny()
    z, v, c = inputs()
    with pytest.raises(ValueError, match="cross-condition"):
        build_responsibilities(model, z, z, v, c, source_conditions=["TF1"]*8, target_conditions=["TF2"]*8)


def test_coupling_refuses_inconsistent_condition_embeddings():
    model = tiny()
    z, v, c = inputs()
    with pytest.raises(ValueError, match="consistent condition embedding"):
        build_responsibilities(model, z, z, v, c, source_conditions=["TF1"]*8, target_conditions=["TF1"]*8)


@pytest.mark.parametrize("changes", [{"n_experts": 0}, {"top_k": 3}, {"hidden_dim": -1}])
def test_config_rejects_invalid_dimensions(changes):
    with pytest.raises(ValueError):
        tiny(**changes)


def test_prediction_rejects_bad_time_and_nan():
    model = tiny()
    z, v, c = inputs()
    with pytest.raises(ValueError):
        model.predict(z, v, c, t1=4.)
    v[0, 0] = torch.nan
    with pytest.raises(ValueError):
        model.predict(z, v, c)
