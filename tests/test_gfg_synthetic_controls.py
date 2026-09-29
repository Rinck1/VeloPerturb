import numpy as np
import torch

from veloroute.gfg_synthetic import groups, permute_group_u
from veloroute.model import ModelConfig, SourceRouter


def test_identical_S_postencoder_shuffle_preserves_predicted_population():
    torch.manual_seed(3)
    router = SourceRouter(ModelConfig(state_dim=2, velocity_dim=3, condition_dim=2,
        hidden_dim=8, router_hidden_dim=8, time_hidden_dim=4, residual_blocks=1,
        expert_rank=2, n_experts=2, top_k=2))
    z, e, v = torch.zeros(48, 2), torch.zeros(48, 2), torch.randn(48, 3)
    permutation = torch.randperm(48)
    a = router.logits(z, v, e).softmax(-1).mean(0)
    b = router.logits(z, v[permutation], e).softmax(-1).mean(0)
    torch.testing.assert_close(a, b)


def test_preencoder_U_shuffle_preserves_S_and_breaks_joint_interaction():
    us, z, y, modes = groups('state_velocity_interaction', [.1, .3, .7, .9], 2000, 101)
    shuffled = permute_group_u(us, 4)
    np.testing.assert_array_equal(shuffled[:, :, 12:], us[:, :, 12:])
    np.testing.assert_array_equal(np.sort(shuffled[:, :, :12], axis=1), np.sort(us[:, :, :12], axis=1))
    # The toy's branch is the sign of the S*U contrast, before any learnable encoder.
    def proportion(values):
        return (((values[:, :, 0]-5)*(values[:, :, 12]-5)) > 0).mean(1)
    np.testing.assert_allclose(proportion(us), modes.mean(1))
    np.testing.assert_allclose(proportion(shuffled), .5, atol=.04)


def test_source_and_target_fates_are_unpaired_not_training_labels():
    us, z, target, source_fates = groups('same_S_velocity_branch', [.3], 500, 18)
    assert (source_fates[0] != (target[0, :, 0] > 0)).mean() > .3
    np.testing.assert_array_equal(us[0, :, 12:], 5)
