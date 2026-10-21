"""CPU-only RENGE CellFlow adapter tests; no biological training is executed."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "phase0"))
import w15c_renge_l2 as module
from w4_artifacts import sha256


@pytest.fixture
def fold_fixture(tmp_path):
    rng = np.random.default_rng(23)
    dimension = 3
    def cohort(label, target=False):
        z = rng.normal(0., .15, (9, dimension)).astype("float32")
        value = dict(z=z, cell_kin=rng.normal(size=z.shape), cell_directU=rng.normal(size=z.shape),
            population_kin=rng.normal(size=dimension), population_Ssum=rng.normal(size=dimension),
            population_directU=rng.normal(size=dimension), cell_ids=np.array([f"{label}_{i}" for i in range(9)]),
            embedding=rng.normal(size=5).astype("float32"), time_hours=0., label=label)
        if target:
            value["target"] = (z + .2).astype("float32")
        else:
            value["score_indices"] = np.array([5, 1, 7, 2])
        return value
    groups = [cohort(label, True) for label in ("TF1", "TF2", "TF3", "CTRL")]
    source = cohort("HELDOUT")
    norm = {}
    for kind in ("kin", "Ssum", "directU", "cell_kin", "cell_directU"):
        norm["mean_" + kind] = np.full(dimension, .1)
        norm["scale_" + kind] = np.full(dimension, 2.)
    config = dict(width=8, depth=1, condition_dim=4, time_freqs=8, time_width=8,
        time_depth=1, pooling="mean", learning_rate=.001, ot_epsilon=.1,
        ot_max_iterations=30, ema=1., batch_size=4, steps=2, log_interval=1)
    cfg = tmp_path / "toy_config.json"
    cfg.write_text(json.dumps(config))
    pack = tmp_path / "toy_pack"
    pack.mkdir()
    (pack / "manifest.json").write_text(json.dumps(dict(kind="CPU_TOY")))
    fm = dict(fold="day2_to_day3__HELDOUT", TF="HELDOUT", source_day=2, target_day=3)
    return dict(groups=groups, source=source, norm=norm, config=config, config_path=cfg, pack=pack, fm=fm)


def test_exact_seven_variants_and_three_shuffle_seeds():
    assert len(module.VARIANTS) == 7
    assert {module.variant_spec(v)["shuffle_seed"] for v in module.VARIANTS
        if "shuf" in v} == {0, 1, 2}
    with pytest.raises(ValueError, match="Unregistered"):
        module.variant_spec("cf_kin_shuf_perm4")


def test_all_arms_use_identical_shapes_and_real_esm(fold_fixture):
    f = fold_fixture
    shapes = []
    for variant in module.VARIANTS:
        groups, source, audit = module.prepare_conditions(f["groups"], f["source"], f["norm"], variant)
        shapes.append({key: value.shape for key, value in source.items()})
        np.testing.assert_array_equal(source["perturbation"][:, 0], np.broadcast_to(
            f["source"]["embedding"], (9, 5)))
        assert not np.count_nonzero(source["cell_token"])
        assert source["bio_delta"].shape == (9, 1, 8)
        assert audit["symmetric_heldout_permutation"] is False
        if variant == "cf_static":
            assert not np.count_nonzero(source["population_token"])
    assert all(value == shapes[0] for value in shapes)


def test_direct_u_and_s_summary_retain_own_training_normalization(fold_fixture):
    f = fold_fixture
    for variant, kind in (("cf_directU", "directU"), ("cf_Ssum", "Ssum"), ("cf_kin", "kin")):
        _, condition, _ = module.prepare_conditions(f["groups"], f["source"], f["norm"], variant)
        expected = (f["source"]["population_" + kind] - .1) / 2.
        np.testing.assert_allclose(condition["population_token"][:, 0], np.broadcast_to(expected, (9, 3)), atol=1e-7)


def test_shuffle_deranges_only_training_conditions_and_preserves_token_bank(fold_fixture):
    f = fold_fixture
    bank = np.stack([(g["population_kin"] - .1) / 2 for g in f["groups"]])
    for seed in (0, 1, 2):
        variant = f"cf_kin_shuf_perm{seed}"
        groups, source, audit = module.prepare_conditions(f["groups"], f["source"], f["norm"], variant)
        mapping = module.training_derangement(f["groups"], seed)
        assert np.all(mapping != np.arange(4))
        np.testing.assert_array_equal(mapping, module.training_derangement(f["groups"], seed))
        got = np.stack([g["condition"]["population_token"][0, 0] for g in groups])
        np.testing.assert_allclose(got, bank[mapping], atol=1e-7)
        assert audit["heldout_training_reference_label"] in [g["label"] for g in f["groups"]]
        reference = [g["label"] for g in f["groups"]].index(audit["heldout_training_reference_label"])
        np.testing.assert_allclose(source["population_token"][0, 0], bank[reference], atol=1e-7)
        assert audit["heldout_source_token_used_in_training"] is False


def test_heldout_real_token_never_read_in_shuffle_or_static(fold_fixture):
    class NoTokens(dict):
        def __getitem__(self, key):
            if key.startswith("population_"):
                pytest.fail("Held-out real source token was read by shuffle/static")
            return super().__getitem__(key)
    f = fold_fixture
    source = NoTokens(f["source"])
    for variant in ("cf_static", "cf_kin_shuf_perm0", "cf_kin_shuf_perm1", "cf_kin_shuf_perm2"):
        module.prepare_conditions(f["groups"], source, f["norm"], variant)


def test_fail_closed_configuration_ids_time_and_training_labels(fold_fixture):
    f = fold_fixture
    with pytest.raises(ValueError, match="fixed"):
        module.validate_configuration(f["config"])
    module.validate_configuration(f["config"], engineering_fixture=True)
    with pytest.raises(ValueError, match="exclude"):
        module.prepare_conditions(f["groups"], dict(f["source"], label="TF1"), f["norm"], "cf_static")
    with pytest.raises(ValueError, match="indices"):
        module.score_rows(dict(f["source"], score_indices=np.array([0, 0])))
    with pytest.raises(ValueError, match="time"):
        module.training_derangement([dict(g, time_hours=i) for i, g in enumerate(f["groups"])], 0)


def test_complete_real_writer_official_two_update_fit_serialization_and_ode(fold_fixture, tmp_path, monkeypatch):
    """Actual production success path, not only a stub before artifact writing."""
    f = fold_fixture
    monkeypatch.setattr(module, "load_fold", lambda *a: (f["fm"], f["groups"], f["source"], f["norm"]))
    output = tmp_path / "cpu_toy_run"
    result = module.train_predict(f["pack"], f["fm"]["fold"], f["config_path"], output,
        "cf_kin_shuf_perm1", 0, engineering_fixture=True)
    assert result["heldout_future_used"] is False
    assert result["steps"] == 2 and result["tau_end"] == 1.
    assert result["solver"] == "Tsit5" and result["mean_anchor"] is False
    assert result["model_sha256"] == sha256(output / "model.msgpack")
    assert result["sha256"] == sha256(output / "predictions.npz")
    with np.load(output / "predictions.npz", allow_pickle=False) as pred:
        assert pred["candidates"].shape == (4, 1, 3)
        assert np.isfinite(pred["candidates"]).all()
        np.testing.assert_array_equal(pred["q"], np.ones((4, 1)))
        np.testing.assert_array_equal(pred["source_ids"], f["source"]["cell_ids"][[5, 1, 7, 2]])
    history = json.loads((output / "training_history.json").read_text())["history"]
    assert history[-1]["step"] == 2 and np.isfinite(history[-1]["loss"])
    assert json.loads((output / "status.json").read_text())["status"] == "complete"
    manifest = json.loads((output / "prediction_manifest.json").read_text())
    assert manifest["shuffle"]["heldout_training_reference_label"] in ("TF1", "TF2", "TF3", "CTRL")
