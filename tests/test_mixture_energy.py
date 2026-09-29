import numpy as np
import pytest
import torch

from veloroute.metrics import energy_distance
from veloroute.mixture_energy import mixture_energy


def test_one_hot_mixture_equals_existing_distribution_metric():
    torch.manual_seed(4)
    candidates, target = torch.randn(12, 2, 3, dtype=torch.float64), torch.randn(9, 3, dtype=torch.float64)
    modes = torch.arange(12) % 2
    q = torch.nn.functional.one_hot(modes, 2).double()
    actual = mixture_energy(candidates, q, target)
    expected = energy_distance(candidates[torch.arange(12), modes].numpy(), target.numpy())
    assert float(actual) == pytest.approx(expected, abs=1e-10)


def test_only_router_probabilities_receive_distribution_gradients():
    candidates = torch.tensor([[[-1.], [1.]]]*8, requires_grad=True)
    target = torch.ones(16, 1, requires_grad=True)
    logits = torch.zeros(8, 2, requires_grad=True)
    loss = mixture_energy(candidates, logits.softmax(-1), target)
    loss.backward()
    assert candidates.grad is None and target.grad is None
    assert (logits.grad[:, 1] < 0).all()
    assert float(mixture_energy(candidates, torch.tensor([[0., 1.]]*8), target)) == 0


def test_exact_mixture_is_invariant_to_source_order():
    torch.manual_seed(3)
    p, y = torch.randn(12, 2, 3), torch.randn(9, 3)
    q = torch.randn(12, 2).softmax(-1)
    order = torch.randperm(12)
    torch.testing.assert_close(mixture_energy(p, q, y), mixture_energy(p[order], q[order], y))


def test_joint_mixture_preserves_two_peaks_and_differentiates_endpoints():
    points = torch.tensor([[[-1.], [1.]]]*4, requires_grad=True)
    q = torch.full((4, 2), .5, requires_grad=True)
    target = torch.tensor([[-1.], [1.]], requires_grad=True)
    loss = mixture_energy(points, q, target, detach_candidates=False)
    assert float(loss.detach()) == pytest.approx(0., abs=1e-7)
    # A barycentre at zero is not the same distribution as the two true peaks.
    averaged = (points.detach()*q.detach()[..., None]).sum(1)[:, None, :]
    assert float(mixture_energy(averaged, torch.ones(4, 1), target)) > .5
    shifted = mixture_energy(points, q, target.detach()+.3, detach_candidates=False)
    shifted.backward()
    assert points.grad.abs().sum() > 0 and q.grad.abs().sum() > 0
    assert target.grad is None
