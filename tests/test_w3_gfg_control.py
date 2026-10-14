import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/phase0'))
from w3_gfg_v2_control import seeded_control


def fitted():
    model = seeded_control(8, state_dim=3, dim=4, codes=4, hidden=(8,), seed=7)
    rng = torch.Generator().manual_seed(20)
    data = torch.rand(24, 16, generator=rng) * 10
    ids = [str(i) for i in range(24)]
    model.fit_source(data, cell_ids=ids, allowed_ids=ids)
    return model.eval(), data, ids


def test_velocity_and_innovation_invariant_to_batch_composition_and_order():
    model, data, _ = fitted()
    with torch.no_grad():
        whole = model(data)[1]
        singles = torch.cat([model(row[None])[1] for row in data])
        order = torch.arange(len(data) - 1, -1, -1)
        permuted = model(data[order])[1][order]
        chunks = torch.cat([model(data[i:i+5])[1] for i in range(0, len(data), 5)])
    torch.testing.assert_close(whole, singles)
    torch.testing.assert_close(whole, permuted)
    torch.testing.assert_close(whole, chunks)


def test_transform_scope_refit_and_legacy_prepare_fail_closed():
    model, data, ids = fitted()
    with pytest.raises(ValueError, match='frozen'):
        model.fit_source(data, cell_ids=ids, allowed_ids=ids)
    with pytest.raises(ValueError, match='explicit'):
        model.prepare(data, model.components)
    fresh = seeded_control(8, state_dim=3, dim=4, codes=4, hidden=(8,))
    with pytest.raises(ValueError, match='scope'):
        fresh.fit_source(data, cell_ids=ids, allowed_ids=ids[:-1])
    with pytest.raises(ValueError, match='not fitted'):
        fresh(data)


def test_seed_before_initialization_and_frozen_buffers_roundtrip():
    model, data, _ = fitted()
    copy, _, _ = fitted()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, copy.state_dict()[key])
    restored = seeded_control(8, state_dim=3, dim=4, codes=4, hidden=(8,), seed=999)
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(data)[1], model(data)[1])


def test_reconstruction_can_go_below_training_mean_and_train_is_finite():
    model, data, _ = fitted()
    with torch.no_grad():
        model.core.decoder.net[-1].weight.zero_()
        model.core.decoder.net[-1].bias.fill_(-2.)
    output, velocity = model(data)
    assert torch.all(output['ms'] < model.s_mean)
    assert torch.isfinite(output['native']) and torch.isfinite(velocity).all()
    output['native'].backward()
    assert model.core.vel_head.weight.grad.norm() > 0


def test_linear_control_is_not_mislabeled_as_full_representation_velocity():
    model, data, _ = fitted()
    model(data)[1].sum().backward()
    assert model.core.vel_head.weight.grad is not None
    assert all(p.grad is None for p in model.core.manifold_encoder.parameters())
