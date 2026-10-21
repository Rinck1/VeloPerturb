"""Source-only RENGE transfer packing mathematics and full writer checks."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "phase0"))
from w4_artifacts import sha256
from w4_streams import SourceTransform
import w15c_renge_data as module


@pytest.fixture
def transform():
    return SourceTransform(genes=np.array([0, 2]), mean=np.array([.2, .4]),
        components=np.array([[1., .3], [-.2, .8]]), gamma=np.array([.5, .7]),
        centers=np.zeros((2, 2)), reference_z=np.zeros((4, 2)),
        source_ids=np.array(["training0", "training1", "training2", "training3"]),
        gene_names=np.array(["g0", "not_hvg_1", "g2", "not_hvg_3"]),
        target_sum=10000., seed=0)


def test_pool_raw_counts_before_log_and_share_all_gene_spliced_depth(transform):
    s = sparse.csr_matrix([[2., 200., 8., 100.], [4., 10., 1., 20.]])
    u = sparse.csr_matrix([[3., 2., 1., 4.], [1., 1., 5., 1.]])
    actual = module.pooled_tokens(transform, s, u)
    factor = 10000. / float(s.sum())
    sn = np.asarray(s.sum(0)).ravel()[transform.genes] * factor
    un = np.asarray(u.sum(0)).ravel()[transform.genes] * factor
    expected = dict(kin=((un - transform.gamma * sn) / (1 + sn)) @ transform.components.T,
        Ssum=np.log1p(sn) @ transform.components.T,
        directU=np.log1p(un) @ transform.components.T)
    for kind in expected:
        np.testing.assert_allclose(actual[kind], expected[kind], atol=1e-12)
    wrong_u = np.asarray(u.sum(0)).ravel()[transform.genes] * (10000. / float(u.sum()))
    assert not np.allclose(actual["directU"], np.log1p(wrong_u) @ transform.components.T)
    per_cell_log = np.log1p(s[:, transform.genes].toarray()
        * (10000. / np.asarray(s.sum(1))) ) @ transform.components.T
    assert not np.allclose(actual["Ssum"], per_cell_log.mean(0))


def test_normalization_only_depends_on_declared_training_source_arrays():
    groups = []
    for index, count in enumerate((2, 3)):
        group = {"population_" + kind: np.array([float(index), 7.])
            for kind in ("kin", "Ssum", "directU")}
        group.update({kind: np.column_stack((np.arange(count) + index, np.full(count, 9.)))
            for kind in ("cell_kin", "cell_directU")})
        group["target"] = np.full((count, 2), 1e9)
        groups.append(group)
    actual = module.normalization(groups)
    np.testing.assert_allclose(actual["mean_kin"], [.5, 7.])
    np.testing.assert_allclose(actual["mean_cell_kin"], [1.4, 9.])
    assert actual["scale_kin"][1] == actual["scale_cell_directU"][1] == 1e-5
    for group in groups:
        group["target"] *= -1e10
    after = module.normalization(groups)
    for key in actual:
        np.testing.assert_array_equal(actual[key], after[key])


def test_convert_excludes_mean_anchor_and_heldout_target():
    class GuardedParent(dict):
        def __getitem__(self, key):
            if key in ("target", "global_rate", "base", "pooled_SU"):
                pytest.fail("Held-out target or old mean-anchor feature read during conversion")
            return super().__getitem__(key)
    parent = dict(z=np.arange(6).reshape(3, 2).astype("float32"),
        cell_ids=np.array(["a", "b", "c"]), velocity=np.ones((3, 2)) * 3,
        direct_U=np.ones((3, 2)) * 5, score_indices=np.array([2, 0]),
        global_rate=np.full(2, 1e8), base=np.full(4, 1e8), target=np.full((3, 2), 1e8))
    tokens = {kind: np.array([5., 6.]) for kind in ("kin", "Ssum", "directU")}
    reference = {kind: np.array([1., 2.]) for kind in ("kin", "Ssum", "directU")}
    result = module.convert(GuardedParent(parent), tokens, reference, np.array([.1, .2]),
        "HELDOUT", 3, training=False)
    assert result["time_hours"] == 24.
    np.testing.assert_array_equal(result["cell_kin"], [[2., 1.]] * 3)
    np.testing.assert_array_equal(result["cell_directU"], [[4., 3.]] * 3)
    assert not {"target", "global_rate", "base", "pooled_SU"}.intersection(result)
    training = module.convert(parent, tokens, reference, np.ones(2), "TRAIN", 2, training=True)
    np.testing.assert_array_equal(training["target"], parent["target"])
    assert "score_indices" not in training and "global_rate" not in training


def test_prepare_32_toy_folds_never_reads_future_counts_and_uses_training_norm(
        transform, tmp_path, monkeypatch):
    """Actual pack writer and checked loader, with raw-read boundaries instrumented."""
    root = tmp_path / "root"
    lock_dir = root / "outputs" / "w4_protocol_lock_v1"
    lock_dir.mkdir(parents=True)
    s = sparse.csr_matrix([[2., 9., 3., 2.], [4., 8., 1., 3.],
        [5., 4., 4., 5.], [6., 2., 3., 3.], [20., 1., 2., 3.], [10., 1., 4., 2.]])
    u = sparse.csr_matrix([[1., 2., 2., 1.], [3., 1., 2., 2.],
        [5., 1., 6., 1.], [4., 2., 5., 1.], [1000., 50., 300., 10.], [900., 10., 400., 10.]])
    ids = np.array(["ctrl1", "ctrl0", "train1", "train0", "held1", "held0"])
    obs = pd.DataFrame(dict(condition=["CTRL", "CTRL", "TRAIN", "TRAIN", "HELDOUT", "HELDOUT"]), index=ids)
    locked = {}
    for key in ("day2", "day3", "embeddings"):
        path = tmp_path / (key + ".bin")
        path.write_bytes(key.encode())
        locked[key] = dict(path=str(path), sha256=sha256(path))
    (lock_dir / "protocol_lock.json").write_text(json.dumps(dict(data_hashes=locked)))
    parent = tmp_path / "parent"
    parent.mkdir()
    transform_path = tmp_path / "frozen_transform.npz"
    transform.save(transform_path)
    records = []
    expected_source_ids = ids[[5, 4]]
    for index in range(32):
        name = f"toy_fold_{index}"
        directory = parent / "folds" / name
        directory.mkdir(parents=True)
        group_records = []
        for label, rows in (("CTRL", [1, 0]), ("TRAIN", [3, 2])):
            path = directory / f"train_{label}.npz"
            np.savez_compressed(path, z=transform.state(s[rows]).astype("float32"),
                velocity=np.full((2, 2), index + 1.), direct_U=np.full((2, 2), index + 2.),
                cell_ids=ids[rows], target=np.full((3, 2), index + 3.), global_rate=np.full(2, 1e9))
            group_records.append(dict(label=label, path=str(path), sha256=sha256(path)))
        source = directory / "source.npz"
        np.savez_compressed(source, z=transform.state(s[[5, 4]]).astype("float32"),
            velocity=np.full((2, 2), 1e8), direct_U=np.full((2, 2), 1e8),
            cell_ids=expected_source_ids, score_indices=np.array([1, 0]), global_rate=np.full(2, 1e9))
        record = dict(name=name, source_day=2, target_day=3, TF="HELDOUT",
            transform_path=str(transform_path), transform_sha256=sha256(transform_path),
            training_groups=group_records, source_path=str(source), source_sha256=sha256(source))
        (directory / "manifest.json").write_text(json.dumps(record))
        records.append(record)
    (parent / "study_manifest.json").write_text(json.dumps(dict(folds=records)))
    protocol = tmp_path / "protocol.md"
    protocol.write_text("CPU toy protocol")
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(parent_root=str(parent), seeds=[0, 1, 2], protocol_path=str(protocol))))
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "read_metadata", lambda path: (obs, transform.gene_names))
    reads = []
    def guarded_read(path, layer, rows):
        assert str(path) == locked["day2"]["path"], "Future target counts read during preparation"
        assert layer in ("spliced", "unspliced")
        reads.append((layer, np.asarray(rows).copy()))
        return {"spliced": s, "unspliced": u}[layer][rows]
    monkeypatch.setattr(module, "read_count_rows", guarded_read)
    monkeypatch.setattr(module, "condition_embeddings", lambda path, labels: np.arange(len(labels) * 3).reshape(len(labels), 3))
    output = tmp_path / "new_pack"
    module.prepare(output, config)
    assert len(reads) == 64
    fm, groups, source, normalizer = module.load_fold(output, "toy_fold_0")
    np.testing.assert_array_equal(source["cell_ids"], expected_source_ids)
    np.testing.assert_array_equal(source["score_indices"], [1, 0])
    assert not {"target", "global_rate", "base"}.intersection(source)
    assert source["time_hours"] == 0. and source["label"] == "HELDOUT"
    assert fm["heldout_future_read"] is False and fm["mean_anchor"] is False
    expected_norm = module.normalization(groups)
    for key in normalizer:
        np.testing.assert_array_equal(normalizer[key], expected_norm[key])
    assert np.max(normalizer["mean_cell_kin"]) < 1e6, "Held-out enormous features entered normalizer"
    assert not np.allclose(normalizer["mean_cell_kin"], module.normalization(groups + [source])["mean_cell_kin"])
    reference = module.pooled_tokens(transform, s[:2], u[:2])
    expected_token = module.pooled_tokens(transform, s[[5, 4]], u[[5, 4]])
    np.testing.assert_allclose(source["population_kin"], expected_token["kin"] - reference["kin"], atol=1e-6)
    norm_path = Path(fm["normalization_path"])
    norm_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Artifact changed"):
        module.load_fold(output, "toy_fold_0")
