"""Tiny diagnostics tests; neither JAX training nor biological inputs."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/phase0'))
from w4_artifacts import sha256
from w15a_l3_synthetic_mechanism import (ARMS,SCENARIOS,analyze,mechanism_metrics,
                                       verify_completed_grid)


def test_high_router_entropy_does_not_hide_collapsed_expert_geometry():
    x=np.zeros((3,4,2));q=np.full((3,4),.25)
    r=mechanism_metrics(x,q,mode_amplitude=2.,observed_high_probability=.5)
    assert r['population_q_entropy']==pytest.approx(np.log(4))
    assert r['mean_cell_q_entropy']==pytest.approx(np.log(4))
    assert r['entropy_gap']==0
    assert r['weighted_within_source_expert_variance']==0
    assert r['uniform_within_source_expert_variance']==0
    assert r['weighted_expert_pairwise_distance']==0
    assert r['weighted_absolute_axis_distance_from_known_modes']==2
    assert r['missing_side_count']==2 and r['predicted_zero_axis_mass']==1
    assert r['two_sided_center_mean_absolute_error'] is None


def test_correct_known_modes_with_population_weights():
    x=np.tile(np.array([[[-2.],[2.]]]),(3,1,1));q=np.tile([.75,.25],(3,1))
    r=mechanism_metrics(x,q,mode_amplitude=2.,observed_high_probability=.25)
    assert r['predicted_high_mass']==pytest.approx(.25)
    assert r['mode_mass_absolute_error']==0
    assert r['weighted_absolute_axis_distance_from_known_modes']==0
    assert r['two_sided_center_mean_absolute_error']==0
    assert r['weighted_within_source_expert_variance']==pytest.approx(3.)
    assert r['uniform_within_source_expert_variance']==pytest.approx(4.)
    assert r['weighted_expert_pairwise_distance']==pytest.approx(1.5)
    assert r['entropy_gap']==0 and r['missing_side_count']==0


def test_deterministic_routes_are_distinguished_from_geometric_collapse():
    x=np.tile(np.array([[[-2.],[2.]]]),(2,1,1));q=np.eye(2)
    r=mechanism_metrics(x,q,mode_amplitude=2.,observed_high_probability=.5)
    assert r['weighted_within_source_expert_variance']==0
    assert r['uniform_within_source_expert_variance']==4
    assert r['entropy_gap']==pytest.approx(np.log(2))
    assert r['conditional_mean_between_source_variance']==4
    assert r['mode_mass_absolute_error']==0


def test_partial_grid_is_refused_before_any_candidate_read(tmp_path,monkeypatch):
    (tmp_path/'status.json').write_text(json.dumps(dict(status='running',completed_fits=119)))
    def forbidden(*args,**kwargs):raise AssertionError('Partial candidate values were read')
    monkeypatch.setattr(np,'load',forbidden)
    with pytest.raises(ValueError,match='partial candidate values are not read'):
        analyze(tmp_path,tmp_path/'analysis')
    assert not (tmp_path/'analysis').exists()


def test_all120fits_and480files_required_even_when_original_gate_failed(tmp_path):
    plan=dict(mode='frozen_L2_full_20_positive_20_null',repetition_seeds=list(range(20)),
        scenario_order=list(SCENARIOS),arms=list(ARMS),model_configuration=dict(steps=2),
        test_composition_ratios=[.2,.4,.6,.8],cells_per_cohort=1,experts=2,dimensions=1)
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    summary=dict(status='L3_POPULATION_SYNTHETIC_GRID_COMPLETE',completed_fits=120,
        plan_sha256=sha256(tmp_path/'plan.json'),population_composition_learning_gate_passed=False)
    last=None
    for scenario in SCENARIOS:
        for seed in plan['repetition_seeds']:
            for arm in ARMS:
                d=tmp_path/'fits'/scenario/f'repetition{seed:02d}'/arm;d.mkdir(parents=True)
                (d/'model.msgpack').write_bytes(b'frozen_model')
                (d/'training_history.json').write_text(json.dumps([dict(step=2)]))
                (d/'summary.json').write_text(json.dumps(dict(scenario=scenario,repetition=seed,arm=arm,
                    model_sha256=sha256(d/'model.msgpack'),initial_parameter_sha256=f'seed{seed}',mean_energy=1.)))
                for group in range(4):
                    last=d/f'test_{group}_predictions.npz'
                    np.savez_compressed(last,candidates=np.array([[[-2.],[2.]]]),q=np.array([[.5,.5]]))
    records,fits=verify_completed_grid(tmp_path,plan,summary)
    assert len(records)==480 and len(fits)==120
    assert summary['population_composition_learning_gate_passed'] is False
    last.unlink()
    with pytest.raises(FileNotFoundError):verify_completed_grid(tmp_path,plan,summary)
