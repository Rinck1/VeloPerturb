"""CPU-only RENGE L3 fixtures; no REN GE biological arrays are opened."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"/"phase0"))
import w15c_renge_l3 as module
from w4_artifacts import sha256


def source_fixture(dimensions=3):
    rng = np.random.default_rng(812)
    z = rng.normal(0., .1, (12, dimensions)).astype("float32")
    k = rng.normal(0., .2, z.shape).astype("float32")
    source = dict(z=z, cell_kin=k, cell_directU=k*2+.3,
        cell_ids=np.array([f"source{i}" for i in range(len(z))]),
        embedding=np.linspace(-.3, .3, 7).astype("float32"), time_hours=24.,
        label="TF_source_day3", score_indices=np.array([9, 2, 7, 1]))
    normalizer = dict(mean_cell_kin=np.zeros(dimensions), scale_cell_kin=np.ones(dimensions),
        mean_cell_directU=np.ones(dimensions)*.3, scale_cell_directU=np.ones(dimensions)*2)
    return source, normalizer


def tiny_config():
    return dict(width=8, depth=1, condition_dim=4, time_freqs=8, time_width=8, time_depth=1,
        pooling="mean", learning_rate=.001, ot_epsilon=.1, ot_max_iterations=30,
        ema=1., batch_size=4, steps=2, log_interval=1)


def test_registered_six_variants_and_esm_condition_feature_dimension():
    assert len(module.VARIANTS) == 6
    assert [module.variant_spec(f"moe_kin_shuf_perm{i}")["shuffle_seed"] for i in range(3)] == [0, 1, 2]
    condition = module.source_state_conditions(4, 24., np.arange(256, dtype="float32"))
    assert condition["perturbation"].shape == (4, 1, 256)
    assert set(condition) == {"perturbation", "bio_time", "bio_delta"}
    assert condition["bio_time"].shape == (4, 1, 8)
    with pytest.raises(ValueError, match="Unregistered"):
        module.variant_spec("moe_kin_shuf_perm4")
    with pytest.raises(ValueError, match="embedding"):
        module.source_state_conditions(1, 0., np.ones((2, 7)))


def test_local_direct_u_static_and_full_cohort_permutations():
    source, norm = source_fixture()
    real = module.kinetic_input(source, norm, "moe_kin")
    np.testing.assert_allclose(real, source["cell_kin"])
    np.testing.assert_allclose(module.kinetic_input(source, norm, "moe_directU"), real, atol=1e-7)
    assert not np.allclose(real, real[:1])
    assert not module.kinetic_input(source, norm, "moe_static").any()
    permutations = []
    for index in range(3):
        variant = f"moe_kin_shuf_perm{index}"
        value = module.kinetic_input(source, norm, variant)
        np.testing.assert_array_equal(np.sort(value, axis=0), np.sort(real, axis=0))
        np.testing.assert_array_equal(value, module.kinetic_input(source, norm, variant))
        permutations.append(value)
    assert not np.array_equal(permutations[0], permutations[1])


def test_synthetic_gate_runs_before_load_fold(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "load_fold", lambda *args: pytest.fail("Real data loaded before gate"))
    with pytest.raises(ValueError, match="Complete 120-fit"):
        module.train_predict("missing", "TF_day3", "missing", tmp_path/"must_not_exist",
            variant="moe_kin", seed=0, synthetic_gate=tmp_path/"missing_gate")
    assert not (tmp_path/"must_not_exist").exists()


def test_wrong_configuration_hash_cannot_change_frozen_recipe(tmp_path):
    path = tmp_path/"cfg.json"; path.write_text(json.dumps(tiny_config()))
    with pytest.raises(ValueError, match="exact synthetic-tested"):
        module._registered_configuration(path, dict(l2_configuration_sha256="wrong"))
    with pytest.raises(ValueError, match="architecture and budget"):
        module._registered_configuration(path, dict(l2_configuration_sha256=sha256(path)))


def test_renge_config_inherits_recipe_without_scifate_metadata(tmp_path):
    original = Path(__file__).resolve().parents[1]/"configs"/"w15a_cellflow_selected_20261021_v1.json"
    config = json.loads(original.read_text())
    for key in ("data_path", "data_sha256", "training_replicates", "original_cell_pack"):
        config.pop(key)
    config.update(inherited_config_path=str(original), inherited_config_sha256=sha256(original),
                  dataset="RENGE", data_path="separate_renge_manifest.json")
    path = tmp_path/"renge_config.json"; path.write_text(json.dumps(config))
    gate = dict(l2_configuration_sha256=sha256(original))
    assert module._registered_configuration(path, gate)["dataset"] == "RENGE"
    config["condition_dim"] = 64; path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="cannot change the inherited"):
        module._registered_configuration(path, gate)


def test_q_uses_all_source_before_selected_ids_and_only_one_integration(monkeypatch):
    source, norm = source_fixture()
    calls = []
    class Model:
        config = SimpleNamespace(experts=8)
        def route(self):
            pass
        def apply(self, variables, z, k, *, method):
            calls.append(("route", np.asarray(z), np.asarray(k)))
            q = np.full((len(z), 8), .01, dtype="float32")
            q[:, 0] = np.linspace(.2, .7, len(z)); q[:, 1] = 1.-q[:, 0]-.06
            return q
    def integrator(model):
        def solve(params, z, condition, tau):
            calls.append(("integrate", np.asarray(z), condition, tau))
            return np.repeat(np.asarray(z)[:, None], 8, axis=1)
        return solve
    monkeypatch.setattr(module, "make_candidate_integrator", integrator)
    prediction = module.predict_arrays(Model(), {}, source, norm, "moe_kin")
    assert [c[0] for c in calls] == ["route", "integrate"]
    assert len(calls[0][1]) == 12 and len(calls[1][1]) == 4
    np.testing.assert_array_equal(prediction["source_ids"], source["cell_ids"][[9, 2, 7, 1]])
    assert not np.allclose(prediction["q"].mean(0), prediction["q_cohort"][0])
    assert prediction["candidates"].shape == (4, 8, 3)
    assert calls[1][2]["perturbation"].shape == (4, 1, 7)
    assert calls[1][3] == 1.
    with pytest.raises(ValueError, match="future target"):
        module.predict_arrays(Model(), {}, dict(source, target=np.zeros((1, 3))), norm, "moe_kin")


def test_tiny_official_ot_fit_checkpoint_and_independent_tsit5(tmp_path):
    import jax
    from flax import serialization
    assert all(device.platform == "cpu" for device in jax.devices())
    source, norm = source_fixture()
    groups = [dict(source, target=source["z"]+.1,
                   kinetic_input=module.kinetic_input(source, norm, "moe_kin"))]
    model, params, history, initial = module.fit_group_arrays(groups, tiny_config(), 0)
    assert history[-1]["step"] == 2
    assert np.isfinite(history[-1]["loss"])
    assert history[-1]["kinetic_gradient_norm"] > 0
    checkpoint = tmp_path/"model.msgpack"
    checkpoint.write_bytes(serialization.to_bytes(params))
    restored = serialization.from_bytes(initial, checkpoint.read_bytes())
    result = module.predict_arrays(model, restored, source, norm, "moe_kin")
    assert result["candidates"].shape == (4, 8, 3)
    np.testing.assert_array_equal(result["source_ids"], source["cell_ids"][[9, 2, 7, 1]])
    assert np.isfinite(result["candidates"]).all()
    assert not np.allclose(result["candidates"][:, 0], result["candidates"][:, 1])


def test_production_writer_success_branch_is_mapping_and_complete(tmp_path, monkeypatch):
    source, normalizer = source_fixture(50)
    group = dict(source, target=source["z"]+.1)
    group.pop("score_indices")
    fm = dict(fold="TF_day3", TF="TF", source_day=3, target_day=4)
    monkeypatch.setattr(module, "load_fold", lambda *a: (fm, [group], source, normalizer))
    configuration = Path(__file__).resolve().parents[1]/"configs"/"w15a_cellflow_selected_20261021_v1.json"
    gate = tmp_path/"gate"; gate.mkdir()
    (gate/"summary.json").write_text("{}\n")
    monkeypatch.setattr(module, "require_complete_synthetic_gate", lambda *a:
        dict(l2_configuration_sha256=sha256(configuration), completed_fits=120, engineering_gate_passed=True))
    model = SimpleNamespace(config=SimpleNamespace(experts=8))
    history = [dict(step=5000, loss=.7)]
    def fake_fit(groups, config, seed, *, callback):
        assert config["steps"] == 5000 and groups[0]["embedding"].shape == (7,)
        callback(history[0])
        return model, dict(toy=np.ones(2)), history, dict(toy=np.zeros(2))
    monkeypatch.setattr(module, "fit_group_arrays", fake_fit)
    def fake_predict(model, params, src, normalizer, variant):
        assert "target" not in src and params["toy"].sum() == 2
        n = len(src["score_indices"])
        return dict(candidates=np.zeros((n, 8, 50)), q=np.ones((n, 8))/8,
            q_cohort=np.ones((n, 8))/8, q_uniform=np.ones((n, 8))/8,
            source_ids=src["cell_ids"][src["score_indices"]])
    monkeypatch.setattr(module, "predict_arrays", fake_predict)
    result = module.train_predict(tmp_path/"pack", "TF_day3", configuration, tmp_path/"run",
        variant="moe_kin", seed=0, synthetic_gate=gate)
    assert result["all_training_steps"] == 5000
    assert result["embedding_dimensions"] == 7
    assert json.loads((tmp_path/"run"/"status.json").read_text())["status"] == "complete"
    assert json.loads((tmp_path/"run"/"training_history.json").read_text())["history"] == history
    manifest = json.loads((tmp_path/"run"/"prediction_manifest.json").read_text())
    assert manifest["heldout_future_used"] is False
    assert manifest["prediction_sha256"] == sha256(tmp_path/"run"/"predictions.npz")
    assert manifest["model_sha256"] == sha256(tmp_path/"run"/"model.msgpack")
