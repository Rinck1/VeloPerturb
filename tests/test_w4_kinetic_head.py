import inspect
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w4_kinetic_head import (KineticCalibrator, apply_calibrator, fit_calibrator,
                             pooled_feature_values, source_partition_pairs)
from w4_streams import SourceTransform, population_features


@pytest.fixture
def source():
    rng = np.random.default_rng(12)
    s, u = rng.poisson(10, (60, 12)), rng.poisson(5, (60, 12))
    ids = np.array([f's{i}' for i in range(60)])
    t = SourceTransform.fit(s, u, cell_ids=ids, allowed_ids=set(ids), gene_names=[f'g{i}' for i in range(12)],
        n_hvg=10, dimensions=5, clusters=4)
    return sparse.csr_matrix(s), sparse.csr_matrix(u), np.repeat(['A','B','C'], 20), ids, t


def test_identity_initialization_zero_preservation_and_no_condition_interface():
    head = KineticCalibrator(np.ones(5))
    v = torch.randn(9, 5)
    torch.testing.assert_close(head(v), v, rtol=0, atol=0)
    assert list(inspect.signature(head.forward).parameters) == ['velocity']
    with pytest.raises(TypeError):
        head(v, condition=torch.ones(9, 2))
    with torch.no_grad():
        head.up.weight.normal_()
    torch.testing.assert_close(head(torch.zeros_like(v)), torch.zeros_like(v), rtol=0, atol=0)


def test_correction_is_bounded_and_task_gradient_can_reach_head():
    head = KineticCalibrator(np.arange(1, 6), maximum_relative_correction=.25)
    with torch.no_grad():
        head.up.weight.normal_(std=100)
    v = torch.randn(20, 5, requires_grad=True)
    change = torch.linalg.vector_norm((head(v)-v)/head.scale, dim=1)
    bound = .25*torch.linalg.vector_norm(v/head.scale, dim=1)
    assert torch.all(change <= bound+1e-6)
    head(v).square().mean().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert head.up.weight.grad is not None


def test_pooled_feature_values_match_registered_full_gene_normalization(source):
    s, u, groups, ids, t = source
    expected = population_features(t, s, u, groups, ['A','B','C'], ids, neighbors=10)
    sums_s, sums_u = [], []
    for label in ['A','B','C']:
        sums_s.append(np.asarray(s[groups == label].sum(0)).ravel())
        sums_u.append(np.asarray(u[groups == label].sum(0)).ravel())
    sums_s, sums_u = np.asarray(sums_s), np.asarray(sums_u)
    actual = pooled_feature_values(t, sums_s[:, t.genes], sums_u[:, t.genes], sums_s.sum(1))
    for feature in ('vel_pool', 'direct_U_pool'):
        np.testing.assert_allclose(actual[feature], expected[feature], atol=1e-12)


def test_source_partition_bank_is_reproducible_and_does_not_mutate_geometry(source):
    s, u, groups, _, t = source
    before = t.fingerprint()
    a, b, identities = source_partition_pairs(t, s, u, groups, ['A','B'], partitions=4, seed=12)
    a2, b2, identities2 = source_partition_pairs(t, s, u, groups, ['A','B'], partitions=4, seed=12)
    assert set(identities) == {'A','B'} and len(identities) == 8
    np.testing.assert_array_equal(identities, identities2)
    for key in a:
        assert a[key].shape == (8, 5)
        np.testing.assert_array_equal(a[key], a2[key]); np.testing.assert_array_equal(b[key], b2[key])
    assert t.fingerprint() == before


def test_minimal_training_changes_weights_preserves_anchor_and_serializes(tmp_path):
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(50, 5)), rng.normal(size=(50, 5))
    config = dict(steps=30, batch_size=16, hidden=4, maximum_relative_correction=.25,
        learning_rate=.01, weight_decay=.0001, anchor_weight=.1)
    head, result = fit_calibrator(a, b, config)
    assert result['final_pair_mse'] < result['initial_pair_mse']
    assert not result['head_future_supervision'] and result['biological_replicates_added'] == 0
    assert result['parameters'] == 40
    assert torch.count_nonzero(head.up.weight) > 0
    path = tmp_path/'head.pt'
    torch.save(head.state_dict(), path)
    restored = KineticCalibrator(head.scale, hidden=4)
    restored.load_state_dict(torch.load(path, weights_only=True))
    np.testing.assert_array_equal(apply_calibrator(head, a), apply_calibrator(restored, a))
    second, _ = fit_calibrator(a, b, config)
    np.testing.assert_array_equal(apply_calibrator(head, a), apply_calibrator(second, a))


def test_reject_invalid_counts_and_head_specification(source):
    s, u, groups, _, t = source
    with pytest.raises(ValueError, match='positive'):
        KineticCalibrator([0, 1])
    with pytest.raises(ValueError, match='Expected'):
        KineticCalibrator([1, 1])(torch.ones(3, 4))
    with pytest.raises(ValueError, match='two source cells'):
        source_partition_pairs(t, s[:1], u[:1], ['A'], ['A'], partitions=1, seed=0)
    invalid = s.copy(); invalid.data[0] = -1
    with pytest.raises(ValueError, match='nonnegative'):
        source_partition_pairs(t, invalid, u, groups, ['A'], partitions=1, seed=0)


def test_future_or_condition_arguments_do_not_exist_in_source_training_interfaces():
    for function in (source_partition_pairs, fit_calibrator):
        arguments = inspect.signature(function).parameters
        assert not {'target', 'future', 'condition', 'embeddings', 'c'} & set(arguments)
