"""Minimal CPU tests for the independent L3 population synthetic runner."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "phase0"))
from w15a_cellflow_moe_synthetic import (prepare_plan, verify_plan, generate_scenario,
    training_arrays, blocked_weighted_energy, state_conditions, run_plan)
from w15a_cellflow_moe import weighted_energy


@pytest.fixture
def plan(tmp_path):
    return prepare_plan(tmp_path / "smoke", engineering_smoke=True, steps=1)


def test_sources_identical_and_positive_target_tracks_kinetic_composition(plan):
    data = generate_scenario(plan, "positive", 0)
    for split in ("train", "test"):
        groups = data[split]
        for group in groups:
            np.testing.assert_array_equal(group["source"], groups[0]["source"])
            assert group["truth"]["source_component_high"].sum() == group["truth"]["target_component_high"].sum()
        assert len(set(group["truth"]["target_component_high"].sum() for group in groups)) == 4


def test_null_breaks_group_composition_target_relation(plan):
    data = generate_scenario(plan, "null", 0)
    for split in ("train", "test"):
        groups = data[split]
        assert len(set(group["truth"]["source_component_high"].sum() for group in groups)) == 4
        assert {group["truth"]["target_component_high"].mean() for group in groups} == {.5}


def test_truth_cannot_change_training_arrays_or_group_conditions(plan):
    group = generate_scenario(plan, "positive", 0)["train"][0]
    before = training_arrays(group, "moe_kin", shuffle_seed=1)
    group["truth"] = {"arbitrary_replaced_labels": np.arange(999)}
    after = training_arrays(group, "moe_kin", shuffle_seed=1)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    conditions = state_conditions(len(group["source"]), plan)
    assert set(conditions) == {"perturbation", "bio_time", "bio_delta"}


def test_group_shuffle_preserves_composition_exactly(plan):
    group = generate_scenario(plan, "positive", 0)["train"][0]
    real = training_arrays(group, "moe_kin", shuffle_seed=3)
    shuffled = training_arrays(group, "moe_kin_shuf", shuffle_seed=3)
    permutation = np.random.default_rng(3).permutation(len(group["source"]))
    np.testing.assert_array_equal(shuffled["kinetic_input"], real["kinetic_input"][permutation])
    for key in ("source", "target"):
        np.testing.assert_array_equal(real[key], shuffled[key])


def test_plan_frozen_before_fit_and_full_grid_requires_l2_freeze(tmp_path):
    with pytest.raises(ValueError, match="committed L2 freeze"):
        prepare_plan(tmp_path / "full")
    plan = prepare_plan(tmp_path / "smoke", engineering_smoke=True, steps=1)
    assert plan["mode"] == "engineering_smoke_not_full_gate"
    assert plan["repetition_seeds"] == [0]
    assert plan["gate"]["shuffle_is_composition_preserving_equivalence_check_not_superiority_control"]
    verify_plan(tmp_path / "smoke")
    path = tmp_path / "smoke" / "plan.json"
    value = json.loads(path.read_text()); value["repetition_seeds"] = list(range(20))
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="plan changed"):
        verify_plan(tmp_path / "smoke")


def test_blocked_exact_energy_matches_small_reference():
    rng = np.random.default_rng(0)
    candidates = rng.normal(size=(5, 3, 4))
    q = rng.dirichlet(np.ones(3), 5)
    target = rng.normal(size=(7, 4))
    assert blocked_weighted_energy(candidates, q, target, block=2) == pytest.approx(
        weighted_energy(candidates, q, target), abs=1e-12)


def test_tiny_cpu_smoke_uses_official_ot_but_cannot_pass_full_gate(tmp_path):
    directory = tmp_path / "actual_engineering_smoke"
    prepare_plan(directory, engineering_smoke=True, steps=1)
    result = run_plan(directory)
    assert result["status"] == "L3_POPULATION_ENGINEERING_SMOKE_COMPLETE"
    assert result["completed_fits"] == 6
    assert result["full_positive_null_grid_completed"] is False
    assert result["population_composition_learning_gate_passed"] is None
    assert result["true_pairing_used"] is False
    assert result["shuffle_superiority_required"] is False
    assert (directory / "all_test_cohorts.csv").exists()
