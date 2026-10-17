import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/phase0"))
import w4_replication as replication
from w4_artifacts import sha256
from w4_arms import CORE_ARMS, paired_gain_test


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "fixture"
    lock = replication.create_synthetic_fixture(root)
    return root, lock


def edit_json(path, function):
    value = json.loads(path.read_text())
    function(value)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return value


def formal_fixture(fixture, *, draft=False):
    root, lock_path = fixture
    protocol = root / replication.PROTOCOL
    if draft:
        protocol.write_text("# Test protocol\n\n状态：**草案**\n")
    # The only commits made by this test are inside pytest's disposable repo.
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", replication.PROTOCOL], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-q", "-m", "synthetic protocol verification fixture"], check=True)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    qc_path = root / "qc_manifest.json"
    edit_json(qc_path, lambda value: value.update(minimum_TF_cells_per_day=50))
    value = json.loads(lock_path.read_text())
    value.pop("fixture_root")
    value.update(kind="formal_replication", freeze_commit=commit, protocol_file_sha256=sha256(protocol),
        QC_manifest_sha256=sha256(qc_path), preprocess=replication.PREPROCESS,
        statistics=dict(bootstrap_repetitions=10000, bootstrap_seed=20261017))
    lock_path.write_text(json.dumps(value, indent=2) + "\n")
    return root, lock_path


def test_draft_blocks_before_metadata_counts_and_output_creation(fixture, tmp_path, monkeypatch):
    root, lock = formal_fixture(fixture, draft=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("Draft protocol must not read any biological metadata/counts")
    monkeypatch.setattr(replication, "read_metadata", forbidden)
    monkeypatch.setattr(replication, "read_count_rows", forbidden)
    output = tmp_path / "blocked"
    with pytest.raises(ValueError, match="DRAFT/not frozen"):
        replication.predict_grid(lock, output, repo=root)
    assert not output.exists()


def test_git_and_current_protocol_are_both_bound_and_statistics_fixed(fixture):
    root, lock = formal_fixture(fixture)
    assert replication.verify_lock(lock, repo=root)["lock"]["kind"] == "formal_replication"
    edit_json(lock, lambda value: value["statistics"].update(bootstrap_seed=7))
    with pytest.raises(ValueError, match="budget differs"):
        replication.verify_lock(lock, repo=root)
    edit_json(lock, lambda value: value["statistics"].update(bootstrap_seed=20261017))
    protocol = root / replication.PROTOCOL
    protocol.write_text(protocol.read_text() + "changed after freeze\n")
    with pytest.raises(ValueError, match="Current protocol differs"):
        replication.verify_lock(lock, repo=root)
    edit_json(lock, lambda value: value.update(protocol_file_sha256=sha256(protocol)))
    with pytest.raises(ValueError, match="Frozen git protocol hash"):
        replication.verify_lock(lock, repo=root)


def test_formal_mode_rejects_synthetic_bypass_and_unrecognized_schema(fixture):
    root, lock = fixture
    with pytest.raises(ValueError, match="schema|Formal prediction"):
        replication.verify_lock(lock)
    edit_json(lock, lambda value: value.update(unregistered_option=True))
    with pytest.raises(ValueError, match="schema"):
        replication.verify_lock(lock, synthetic_root=root)


def test_QC_data_identity_is_checked_not_only_its_own_hash(fixture):
    root, lock = fixture
    qc_path = root / "qc_manifest.json"
    edit_json(qc_path, lambda value: value["data_hashes"].update({str(root / "day2.h5ad"): "0"*64}))
    edit_json(lock, lambda value: value.update(QC_manifest_sha256=sha256(qc_path)))
    with pytest.raises(ValueError, match="QC did not audit this exact"):
        replication.verify_lock(lock, synthetic_root=root)


def test_actual_CSR_reads_exclude_heldout_future_and_geometry_ignores_it(fixture, tmp_path, monkeypatch):
    root, lock = fixture
    verified = replication.verify_lock(lock, synthetic_root=root)
    metadata = replication.cohort_metadata(verified)
    target = root / "day3.h5ad"
    accesses = []
    original = h5py.Dataset.__getitem__
    def watch(dataset, key):
        if Path(dataset.file.filename) == target and dataset.name in ("/layers/spliced/data", "/layers/unspliced/data"):
            accesses.append((dataset.name, key))
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", watch)
    arguments = (root / "day2.h5ad", target, root / "embeddings.npz", metadata[2], metadata[3],
        verified["lock"]["H1_TFs"], "SYNTH_TF0", verified["lock"]["preprocess"])
    first = replication.fit_predict_fold(*arguments, tmp_path / "fold1")
    with h5py.File(target, "r") as handle:
        pointers = handle["layers/spliced/indptr"][:]
    allowed = {j for row in first["training_target_rows"] for j in range(pointers[row], pointers[row+1])}
    assert accesses
    assert all(name == "/layers/spliced/data" and set(range(key.start, key.stop)) <= allowed for name, key in accesses)
    # Poison only the withheld future rows in this generated fixture. This
    # direct fold diagnostic intentionally bypasses the global immutable lock.
    heldout_rows = np.flatnonzero(metadata[3]["labels"] == "SYNTH_TF0")
    with h5py.File(target, "r+") as handle:
        node = handle["layers/spliced/data"]
        for row in heldout_rows:
            start, stop = pointers[row], pointers[row+1]
            node[start:stop] = node[start:stop] * np.arange(1, stop-start+1)
    second = replication.fit_predict_fold(*arguments, tmp_path / "fold2")
    with np.load(tmp_path / "fold1/predictions.npz") as a, np.load(tmp_path / "fold2/predictions.npz") as b:
        for arm in CORE_ARMS:
            np.testing.assert_array_equal(a[arm], b[arm])
    for seed in replication.SEEDS:
        a = replication.SourceTransform.load(tmp_path / f"fold1/transform_seed{seed}.npz")
        b = replication.SourceTransform.load(tmp_path / f"fold2/transform_seed{seed}.npz")
        assert a.fingerprint() == b.fingerprint()
    assert first["training_TFs"] == second["training_TFs"]
    assert first["actual_ridge_fits"] <= first["potential_seed_arm_fits"]


@pytest.fixture
def frozen_grid(fixture, tmp_path):
    root, lock = fixture
    predictions = tmp_path / "predictions"
    manifest = replication.predict_grid(lock, predictions, synthetic_root=root)
    return root, lock, predictions, manifest


def test_entire_grid_freezes_then_independent_evaluation_uses_TF_statistics(frozen_grid, tmp_path):
    root, lock, predictions, manifest = frozen_grid
    assert len(manifest["records"]) == 10
    assert (predictions / "freeze_receipt.json").is_file()
    output = tmp_path / "evaluation"
    result = replication.evaluate_grid(lock, predictions, output, synthetic_root=root)
    assert result["n_TFs"] == 5 and result["seeds_averaged_before_scoring"]
    assert not result["formal_biological_claim"]
    with np.load(output / "TF_interval_errors.npz") as errors:
        for ref in ("static", "vel_pool_Uswap"):
            expected = paired_gain_test(errors["vel_pool"], errors[ref], errors["static"], seed=20261017, repetitions=1000)
            assert result["primary_comparisons"][ref] == expected
            assert expected["independent_TFs"] == 5
    # Independently check the statistic against mean(seed predictions), not
    # mean(seed errors), using saved predictions and actual synthetic targets.
    record = manifest["records"][0]
    transform = replication.SourceTransform.load(predictions / record["directory"] / "transform_seed0.npz")
    obs, _ = replication.read_metadata(root / "day3.h5ad")
    rows = np.flatnonzero(obs["condition"].astype(str).to_numpy() == record["TF"])
    target = replication.read_count_rows(root / "day3.h5ad", "spliced", rows)
    truth = transform.state(target).mean(0)
    with np.load(predictions / record["directory"] / "predictions.npz") as saved:
        manual = float(((saved["static"].mean(0)[0] - truth)**2).sum())
    with (output / "metrics.csv").open() as stream:
        matching = [r for r in csv.DictReader(stream) if r["TF"] == record["TF"] and r["source_day"] == "2" and r["arm"] == "static"]
    assert len(matching) == 1 and float(matching[0]["sse"]) == pytest.approx(manual)


def no_count_reads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Cannot read held-out target before verifying every frozen artifact")
    monkeypatch.setattr(replication, "read_count_rows", forbidden)


def test_incomplete_grid_is_rejected_before_any_target_count_read(frozen_grid, tmp_path, monkeypatch):
    root, lock, predictions, _ = frozen_grid
    manifest = predictions / "grid_manifest.json"
    edit_json(manifest, lambda value: value["records"].pop())
    # Rebind the outer hash to test completeness itself, not only a hash mismatch.
    edit_json(predictions / "freeze_receipt.json", lambda value: value.update(grid_manifest_sha256=sha256(manifest)))
    no_count_reads(monkeypatch)
    with pytest.raises(ValueError, match="Entire registered"):
        replication.evaluate_grid(lock, predictions, tmp_path / "blocked", synthetic_root=root)


def test_symmetric_seed_prediction_offsets_do_not_change_evaluated_error(frozen_grid, tmp_path):
    root, lock, predictions, manifest = frozen_grid
    before = replication.evaluate_grid(lock, predictions, tmp_path / "before", synthetic_root=root)
    # Algebraic counterexample: mean squared seed errors increases, but the
    # squared error of the mean prediction stays exactly unchanged. Re-freeze
    # this deliberately constructed SYNTHETIC prediction grid for the test.
    record = manifest["records"][0]
    fold = predictions / record["directory"]
    prediction_path = fold / "predictions.npz"
    with np.load(prediction_path) as p:
        contents = {key: p[key] for key in p.files}
    initial = contents["static"].copy()
    offsets = np.array([2., -2., 3., -3.])[:, None, None]
    contents["static"] = initial + offsets
    np.testing.assert_allclose(contents["static"].mean(0), initial.mean(0), atol=1e-14)
    assert np.square(contents["static"] - initial.mean(0)).mean() > 1
    np.savez_compressed(prediction_path, **contents)
    edit_json(fold / "fold_manifest.json", lambda value: value.update(prediction_sha256=sha256(prediction_path)))
    def refresh_fold(value):
        match = next(r for r in value["records"] if r["directory"] == record["directory"])
        match["fold_manifest_sha256"] = sha256(fold / "fold_manifest.json")
    edit_json(predictions / "grid_manifest.json", refresh_fold)
    edit_json(predictions / "freeze_receipt.json", lambda value:
        value.update(grid_manifest_sha256=sha256(predictions / "grid_manifest.json")))
    after = replication.evaluate_grid(lock, predictions, tmp_path / "after", synthetic_root=root)
    assert after["mean_TF_interval_sse"]["static"] == pytest.approx(before["mean_TF_interval_sse"]["static"], abs=1e-12)
    assert after["primary_comparisons"]["static"]["independent_TFs"] == 5


@pytest.mark.parametrize("what", ["predictions", "transform", "snapshot", "target"])
def test_changed_artifact_rejected_before_target_read(frozen_grid, tmp_path, monkeypatch, what):
    root, lock, predictions, manifest = frozen_grid
    record = manifest["records"][-1]  # corrupt LAST fold; earlier targets must stay unread too.
    if what == "predictions":
        path = predictions / record["directory"] / "predictions.npz"
    elif what == "transform":
        path = predictions / record["directory"] / "transform_seed3.npz"
    elif what == "snapshot":
        path = predictions / "code_snapshot/w4_arms.py"
    else:
        path = root / "day4.h5ad"
    with path.open("ab") as stream:
        stream.write(b"changed after freeze")
    no_count_reads(monkeypatch)
    with pytest.raises(ValueError, match="changed"):
        replication.evaluate_grid(lock, predictions, tmp_path / "blocked", synthetic_root=root)


def test_running_code_change_prevents_complete_grid_freeze(fixture, tmp_path, monkeypatch):
    root, lock = fixture
    dependency = tmp_path / "synthetic_dependency.py"
    dependency.write_text("INITIAL = True\n")
    paths = (*replication.CODE_PATHS, dependency)
    monkeypatch.setattr(replication, "CODE_PATHS", paths)
    monkeypatch.setattr(replication, "IMPORTED_CODE_HASHES", {str(p): sha256(p) for p in paths})
    original = replication.fit_predict_fold
    def mutate_after_first_fold(*args, **kwargs):
        result = original(*args, **kwargs)
        dependency.write_text("INITIAL = False\n")
        return result
    monkeypatch.setattr(replication, "fit_predict_fold", mutate_after_first_fold)
    output = tmp_path / "interrupted"
    with pytest.raises(ValueError, match="code changed"):
        replication.predict_grid(lock, output, synthetic_root=root)
    assert not (output / "freeze_receipt.json").exists()
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


def test_synthetic_mode_cannot_redirect_to_an_unmarked_external_H5AD(fixture, tmp_path):
    root, lock = fixture
    external = tmp_path / "outside.h5ad"
    shutil.copyfile(root / "day2.h5ad", external)
    edit_json(lock, lambda value: value["data_hashes"]["day2"].update(path=str(external)))
    with pytest.raises(ValueError, match="outside its generated"):
        replication.verify_lock(lock, synthetic_root=root)
