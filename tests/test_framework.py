import json
from pathlib import Path

import numpy as np
import pytest

from veloroute.artifacts import Run, load_config, object_hash
from veloroute.contracts import FitScope, FrozenPCA, SourceBatch, stratified_permutation
from veloroute.metrics import energy_distance, paired_condition_bootstrap
from veloroute.protocol import assert_real_training_allowed, freeze_protocol, protocol_errors
from veloroute.raw import audit_fastq_pair, plan_raw, run_command
from veloroute.splits import attach_split, make_condition_split


def source_mapping():
    return {"spliced": np.ones((3, 2)), "unspliced": np.ones((3, 2)),
            "cell_ids": ("day4:a", "day4:b", "day4:c"), "gene_ids": ("G1", "G2"),
            "conditions": ("TF1",)*3, "source_day": 4, "target_day": 5, "task": "ER-short"}


def test_source_contract_valid():
    assert SourceBatch.from_mapping(source_mapping()).source_day == 4


@pytest.mark.parametrize("field", ["target_expression", "target_velocity", "target_mode", "target_statistics"])
def test_source_contract_rejects_future_fields(field):
    with pytest.raises(ValueError, match="Source-only"):
        SourceBatch.from_mapping({**source_mapping(), field: np.zeros((3, 2))})


def test_source_rejects_uncalled_conditions_and_duplicate_ids():
    for update in ({"conditions": ("",)*3}, {"cell_ids": ("a", "a", "b")}):
        with pytest.raises(ValueError):
            SourceBatch.from_mapping({**source_mapping(), **update})


def test_train_only_pca_and_gene_order():
    x = np.random.default_rng(0).normal(size=(5, 3))
    scope = FitScope(frozenset("abcde"))
    pca = FrozenPCA.fit(x, cell_ids=list("abcde"), gene_ids=list("XYZ"), scope=scope,
                        n_components=3, input_space="declared_log_expression")
    z = pca.transform(x, gene_ids=list("XYZ"), input_space="declared_log_expression")
    np.testing.assert_allclose(pca.inverse_transform(z), x, atol=1e-12)
    with pytest.raises(ValueError, match="Leakage"):
        FrozenPCA.fit(x, cell_ids=list("abcdQ"), gene_ids=list("XYZ"), scope=scope,
                      n_components=2, input_space="declared_log_expression")
    with pytest.raises(ValueError):
        pca.transform(x, gene_ids=list("ZYX"), input_space="declared_log_expression")
    with pytest.raises(ValueError):
        pca.transform(x, gene_ids=list("XYZ"), input_space="raw_counts")


def test_permutation_stays_in_time_condition_strata():
    x = np.arange(24.).reshape(12, 2)
    strata = [(4, "TF1")]*6 + [(5, "TF1")]*6
    a, indices = stratified_permutation(x, strata, 4)
    b, _ = stratified_permutation(x, strata, 4)
    np.testing.assert_array_equal(a, b)
    assert set(indices[:6]) == set(range(6)) and set(indices[6:]) == set(range(6, 12))
    with pytest.raises(ValueError):
        stratified_permutation(x, [(None,)]*12, 4)


def guides():
    return [{"guide_name": f"TF{i}_{j}", "target_from_published_name": f"TF{i}", "control_class": "TF"}
            for i in range(23) for j in (1, 2)] + [
                {"guide_name": "CTRL_1", "target_from_published_name": "CTRL", "control_class": "CTRL"},
                {"guide_name": "AAVS1_1", "target_from_published_name": "AAVS1", "control_class": "AAVS1"}]


def test_condition_split_is_order_independent_and_groups_guides():
    split = make_condition_split(guides(), seed=7)
    assert split == make_condition_split(list(reversed(guides())), seed=7)
    for target in {r["condition"] for r in split}:
        assert len({r["role"] for r in split if r["condition"] == target}) == 1
    assert len({r["condition"] for r in split if r["role"] == "confirmation"}) == 5
    assert {r["condition"] for r in split if r["role"] == "control"} == {"CTRL", "AAVS1"}


def test_split_refuses_largest_guide_pseudo_labels():
    split = make_condition_split(guides(), seed=7)
    with pytest.raises(ValueError, match="validated"):
        attach_split([{"cell_id": "day4:a", "assigned_condition": "TF1", "label_status": "not_called"}], split)


def test_energy_distance_definition_and_blocks():
    assert energy_distance([[0.]], [[2.]]) == 4.
    x = np.array([[0., 1.], [1., 3.], [2., 0.]])
    assert energy_distance(x, x, block_size=1) == 0.
    assert energy_distance(x, x+1, block_size=1) == pytest.approx(energy_distance(x, x+1, block_size=256))


def metric_grid():
    return [{"condition": c, "seed": seed, "arm": arm, "energy_distance": value}
            for c in ("A", "B", "C", "D") for seed in (0, 1, 2)
            for arm, value in (("real", 1.), ("static", 2.))]


def bootstrap(rows):
    return paired_condition_bootstrap(rows, real_arm="real", control_arm="static", seeds=[0, 1, 2],
                                       conditions=["A", "B", "C", "D"], n_bootstrap=1000, seed=9)


def test_bootstrap_treats_conditions_not_seeds_as_units():
    result = bootstrap(metric_grid())
    assert result["gain"] == result["ci_low"] == result["ci_high"] == 1.
    assert result["n_conditions"] == 4 and result["n_seeds"] == 3


def test_bootstrap_fails_on_missing_duplicate_or_unregistered_rows():
    rows = metric_grid()
    for changed in (rows[:-1], rows+[rows[0]], rows+[{**rows[0], "condition": "NEW"}]):
        with pytest.raises(ValueError):
            bootstrap(changed)


def test_draft_cannot_be_frozen():
    config = load_config("configs/veloroute_g1.draft.yaml")
    assert protocol_errors(config)
    with pytest.raises(ValueError, match="cannot be frozen"):
        freeze_protocol(config)


def test_synthetic_pass_cannot_authorize_real_training(tmp_path):
    path = tmp_path/"evidence.json"
    path.write_text("{}")
    from veloroute.protocol import REQUIRED_CHECKS
    config = {"task": "ER-short", "metric": "energy_distance_v_statistic", "seeds": [0, 1, 2],
              "minimum_useful_gain": .1, "noninferiority_margin": .1, "effect_margin_rationale": "test fixture",
              "readiness": dict.fromkeys(REQUIRED_CHECKS, True),
              **dict.fromkeys(("split_manifest", "data_provenance", "velocity_provenance", "probe_config"), str(path))}
    frozen = freeze_protocol(config)
    with pytest.raises(ValueError, match="G1-GO"):
        assert_real_training_allowed(frozen, {"status": "GO", "kind": "synthetic", "stage": "G1",
                                              "protocol_hash": frozen["freeze_hash"]})
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        assert_real_training_allowed(frozen, {})


def test_run_logs_failures_and_refuses_overwrite(tmp_path):
    output = tmp_path/"run"
    with pytest.raises(RuntimeError):
        with Run(output, stage="test", kind="engineering", config={}) as run:
            raise RuntimeError("test error")
    assert json.loads((output/"provenance.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        Run(output, stage="test", kind="engineering", config={})


def fastq_fixture(path, names, sequences):
    path.write_text("".join(f"@{name}\n{seq}\n+\n{'I'*len(seq)}\n" for name, seq in zip(names, sequences)))


def test_fastq_layout_offset_and_pair_validation(tmp_path):
    r1, r2 = tmp_path/"a.fastq", tmp_path/"b.fastq"
    barcode = "ACGTACGTACGTACGT"
    fastq_fixture(r1, ["spot.1", "spot.2"], ["N"+barcode+"A"*9]*2)
    fastq_fixture(r2, ["spot.1", "spot.2"], ["A"*91]*2)
    result = audit_fastq_pair(r1, r2, whitelist=[barcode+"-1"], max_records=0)
    assert result["records_examined"] == 2 and result["integrity_scope"] == "full_files"
    assert result["published_barcode_match_fraction_by_zero_based_offset"] == {"0": 0., "1": 1.}
    assert not result["chemistry_approved"]
    fastq_fixture(r2, ["other", "spot.2"], ["A"*91]*2)
    with pytest.raises(ValueError, match="mismatched"):
        audit_fastq_pair(r1, r2)


def test_fastq_rejects_missing_mate(tmp_path):
    r1, r2 = tmp_path/"a.fastq", tmp_path/"b.fastq"
    fastq_fixture(r1, ["spot.1", "spot.2"], ["A"*26]*2)
    fastq_fixture(r2, ["spot.1"], ["A"*91])
    with pytest.raises(ValueError):
        audit_fastq_pair(r1, r2, max_records=0)


def test_raw_plan_keeps_guide_reads_out_of_USA(tmp_path):
    manifest = [{"day": 4, "run_accession": f"SRR{i}", "library_type": kind}
                for i, kind in enumerate(("mRNA", "mRNA", "gRNA", "gRNA"))]
    tasks = plan_raw(manifest, 4, tmp_path, tmp_path/"raw")
    assert all("--include-technical" in t["extract"] for t in tasks)
    assert all(t["usa_quantification"] == "not_GEX_never_send_to_USA" for t in tasks if t["library_type"] == "gRNA")
    with pytest.raises(ValueError):
        plan_raw(manifest[:-1], 4, tmp_path, tmp_path/"raw")


def test_failed_external_job_is_recorded(tmp_path):
    import sys
    with pytest.raises(RuntimeError, match="exit code 3"):
        run_command([sys.executable, "-c", "raise SystemExit(3)"], tmp_path/"job", timeout_seconds=10)
    assert json.loads((tmp_path/"job/status.json").read_text())["status"] == "failed"


def test_toy_hidden_state_has_no_static_position_leakage():
    from veloroute.synthetic import generate, true_path
    import torch
    batch = generate("bimodal_velocity", 128, 1)
    assert torch.count_nonzero(batch["z0"]) == 0
    assert set(batch["modes"].tolist()) == {0, 1}
    path = generate("single_endpoint_path_velocity", 128, 1)
    endpoint, _ = true_path("single_endpoint_path_velocity", path, 1.)
    midpoint, _ = true_path("single_endpoint_path_velocity", path, .5)
    assert endpoint[:, 0].abs().max() < 1e-6
    assert midpoint[:, 0].abs().min() > .9
