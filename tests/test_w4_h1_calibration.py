import sys
import json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/phase0"))
from w4_h1_calibration import (ALGORITHM_SEEDS, MAIN_ARMS, Geometry, binomial_rate,
    correlated_latents, fit_outer_fold, h1_decision, one_replicate, parser,
    simulate_population, validate_args)


@pytest.fixture
def small_geometry():
    return Geometry(n_tfs=5, cells=5, genes=8, dimensions=3, clusters=2, embedding_dimensions=2)


def fold_inputs(data, geometry, tf="TF000", interval=0):
    train = data["groups"] != tf
    return dict(source_s=data["s"][interval], source_u=data["u"][interval],
        source_groups=data["groups"], source_ids=data["ids"][interval], gene_names=data["gene_names"],
        train_future_s=data["s"][interval+1][train], train_future_groups=data["groups"][train],
        labels=data["labels"], embeddings=data["embeddings"], heldout=tf, geometry=geometry)


def test_destructive_snapshots_are_distinct_integer_count_samples(small_geometry):
    data = simulate_population(31, "positive", .5, small_geometry)
    assert len(data["s"]) == 3 and len(data["u"]) == 2
    for time in range(3):
        counts = data["s"][time]
        assert counts.shape == (35, 8)
        assert np.issubdtype(counts.dtype, np.integer) and counts.min() >= 0
        np.testing.assert_array_equal(counts.sum(1), np.repeat(10000, 35))
    assert set(data["ids"][0]).isdisjoint(data["ids"][1])
    assert set(data["ids"][1]).isdisjoint(data["ids"][2])
    assert not np.array_equal(data["s"][0], data["s"][1])
    assert data["independent_cells_across_time"]


def test_null_future_does_not_depend_on_U_innovation_seed(small_geometry):
    a = simulate_population(32, "null", .9, small_geometry, innovation_seed=10)
    b = simulate_population(32, "null", .9, small_geometry, innovation_seed=20)
    for time in range(3):
        np.testing.assert_array_equal(a["s"][time], b["s"][time])
        np.testing.assert_array_equal(a["profiles"][time], b["profiles"][time])
    assert not np.array_equal(a["u"][0], b["u"][0])
    pa = simulate_population(32, "positive", .9, small_geometry, innovation_seed=10)
    pb = simulate_population(32, "positive", .9, small_geometry, innovation_seed=20)
    np.testing.assert_array_equal(pa["s"][0], pb["s"][0])
    assert not np.array_equal(pa["s"][1], pb["s"][1])


def test_latent_correlation_parameter_has_declared_meaning():
    effects = correlated_latents(np.random.default_rng(33), 5000, 8, .5)
    assert abs(np.corrcoef(effects.reshape(2, -1))[0, 1] - .5) < .02
    identical = correlated_latents(np.random.default_rng(33), 20, 8, 1.)
    np.testing.assert_array_equal(identical[0], identical[1])
    with pytest.raises(ValueError, match="rho"):
        correlated_latents(np.random.default_rng(33), 5, 8, 1.1)


def test_outer_fold_excludes_heldout_source_and_target(small_geometry):
    data = simulate_population(34, "positive", 0., small_geometry)
    arguments = fold_inputs(data, small_geometry)
    predictions, transform, audit = fit_outer_fold(**arguments)
    heldout_ids = set(data["ids"][0][data["groups"] == "TF000"])
    assert heldout_ids.isdisjoint(transform.source_ids)
    assert not audit["heldout_source_in_fit"] and not audit["heldout_future_in_training"]
    assert audit["training_tfs"] == 4 and audit["controls"] == 2
    assert set(predictions) == set(MAIN_ARMS)
    assert all(p.shape == (len(ALGORITHM_SEEDS), 1, 3) for p in predictions.values())
    poisoned = dict(arguments, train_future_s=data["s"][1], train_future_groups=data["groups"])
    with pytest.raises(ValueError, match="held-out future"):
        fit_outer_fold(**poisoned)


def test_heldout_future_changes_do_not_change_fold_prediction(small_geometry):
    data = simulate_population(35, "positive", .5, small_geometry)
    first, transform, _ = fit_outer_fold(**fold_inputs(data, small_geometry))
    data["s"][1][data["groups"] == "TF000"] *= 100
    second, repeated_transform, _ = fit_outer_fold(**fold_inputs(data, small_geometry))
    assert transform.fingerprint() == repeated_transform.fingerprint()
    for arm in MAIN_ARMS:
        np.testing.assert_array_equal(first[arm], second[arm])


def test_cache_is_exactly_equal_to_four_seed_fits(small_geometry):
    data = simulate_population(36, "positive", .5, small_geometry)
    cached, _, cache_audit = fit_outer_fold(**fold_inputs(data, small_geometry))
    brute, _, brute_audit = fit_outer_fold(**fold_inputs(data, small_geometry), cache_identical_predictions=False)
    for arm in MAIN_ARMS:
        np.testing.assert_array_equal(cached[arm], brute[arm])
    assert cache_audit["ridge_fits"] == 3
    assert brute_audit["ridge_fits"] == 12
    assert cache_audit["nonunique_swap_nearest_neighbors"] == 0
    assert not cache_audit["kmeans_consumed_by_primary_arms"]


def test_tied_swap_donors_are_not_silently_collapsed(small_geometry):
    data = simulate_population(37, "positive", .5, small_geometry)
    # Every group receives the same observed source-cell collection, so their
    # mean source states are identical and all different donor groups tie.
    data["s"][0] = np.tile(data["s"][0][:small_geometry.cells], (len(data["labels"]), 1))
    cached, _, audit = fit_outer_fold(**fold_inputs(data, small_geometry))
    brute, _, _ = fit_outer_fold(**fold_inputs(data, small_geometry), cache_identical_predictions=False)
    for arm in MAIN_ARMS:
        np.testing.assert_array_equal(cached[arm], brute[arm])
    assert audit["nonunique_swap_nearest_neighbors"] > 0
    assert audit["ridge_fits"] > 3


def test_full_H1_intersection_requires_both_references_and_intervals():
    real = np.ones((8, 2))
    errors = {"vel_pool": real, "static": real * 2, "vel_pool_Uswap": real * 2}
    result = h1_decision(errors, bootstrap_seed=1, repetitions=1000)
    assert result["detected"]
    assert all(v["independent_TFs"] == 8 for v in result["comparisons"].values())
    errors["vel_pool_Uswap"] = real.copy()
    assert not h1_decision(errors, bootstrap_seed=1, repetitions=1000)["detected"]
    errors["vel_pool_Uswap"] = np.tile([3., .9], (8, 1))
    assert not h1_decision(errors, bootstrap_seed=1, repetitions=1000)["detected"]
    with pytest.raises(ValueError, match="two fixed intervals"):
        h1_decision({k: v[:, :1] for k, v in errors.items()}, bootstrap_seed=1, repetitions=1000)


def test_complete_small_replicate_has_two_loto_intervals(small_geometry):
    row, audits = one_replicate(38, "null", .5, small_geometry, bootstrap=100)
    assert row["outer_source_fits"] == 2 * small_geometry.n_tfs
    assert len(audits) == 10
    assert row["static_independent_TFs"] == small_geometry.n_tfs
    assert row["vel_pool_Uswap_independent_TFs"] == small_geometry.n_tfs
    assert all(not a["heldout_source_in_fit"] for a in audits)
    assert {a["interval"] for a in audits} == {0, 1}


def test_formal_mode_cannot_be_claimed_by_a_small_smoke_or_unbound_TF_count(tmp_path):
    args = parser().parse_args(["--output", str(tmp_path / "out"), "--mode", "formal"])
    with pytest.raises(ValueError, match=">=200"):
        validate_args(args)
    args.replicates = 200
    with pytest.raises(ValueError, match="technical-QC"):
        validate_args(args)
    args.mode = "smoke"
    geometry, rhos = validate_args(args)
    assert geometry.n_tfs == 8 and rhos == (0., .5, .9)
    args.rho_grid = "0,0.9"
    with pytest.raises(ValueError, match="fixed latent-correlation grid"):
        validate_args(args)


def test_formal_mode_accepts_root_QC_schema_and_rejects_unsealed_or_incomplete_QC(tmp_path):
    qc_path = tmp_path / "qc.json"
    qc = dict(status="TECHNICAL_COHORT_QC_COMPLETE", biological_effects_read=False,
              H1_TFs=[f"TF{i}" for i in range(8)])
    qc_path.write_text(json.dumps(qc))
    args = parser().parse_args(["--output", str(tmp_path / "out"), "--mode", "formal",
        "--replicates", "200", "--qc-manifest", str(qc_path)])
    assert validate_args(args)[0].n_tfs == 8
    qc["biological_effects_read"] = True
    qc_path.write_text(json.dumps(qc))
    with pytest.raises(ValueError, match="without biological"):
        validate_args(args)
    qc.update(biological_effects_read=False, status="running")
    qc_path.write_text(json.dumps(qc))
    with pytest.raises(ValueError, match="complete technical"):
        validate_args(args)


def test_binomial_intervals_report_smoke_uncertainty_honestly():
    zero = binomial_rate(0, 2)
    assert zero["rate"] == 0 and zero["ci_high"] > .8
    one = binomial_rate(2, 2)
    assert one["rate"] == 1 and one["ci_low"] < .2
    with pytest.raises(ValueError, match="Invalid"):
        binomial_rate(3, 2)
