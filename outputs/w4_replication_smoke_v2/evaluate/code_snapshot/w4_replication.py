"""Locked W4 H1 replication: separate all-grid prediction and evaluation.

Formal use requires a frozen, git-addressed protocol and technical-QC lock.
The synthetic-smoke command generates its own tiny, marked H5AD fixtures; it
does not accept existing biological-data paths or weaken the formal CLI gate.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import h5py
import numpy as np
from sklearn.cluster import KMeans

from w4_artifacts import Artifacts, sha256
from w4_arms import CORE_ARMS, average_predictions_then_errors, design, paired_gain_test, predict_arms
from w4_shorttime import CONTROL_LABELS, condition_embeddings, qc_mask, read_count_rows, read_metadata
from w4_streams import SourceTransform, add_swap_feature, group_mean, population_features


SCHEMA = "w4_h1_replication_lock_v1"
PROTOCOL = "PLANS/w4_shorttime_prereg.md"
INTERVALS = ((2, 3), (3, 4))
SEEDS = (0, 1, 2, 3)
PREPROCESS = dict(n_hvg=2000, dimensions=50, clusters=8, neighbors=30)
CODE_PATHS = tuple(Path(__file__).with_name(name).resolve() for name in (
    "w4_replication.py", "w4_shorttime.py", "w4_streams.py", "w4_arms.py", "w4_artifacts.py"))
IMPORTED_CODE_HASHES = {str(path): sha256(path) for path in CODE_PATHS}


def encoded_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def resolved(path, repo):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def verify_lock(path, *, repo=None, synthetic_root=None):
    """Verify protocol/QC/data identity BEFORE any metadata or count decoding."""
    repo = Path(repo or Path(__file__).resolve().parents[2]).resolve()
    path = Path(path).resolve()
    lock = json.loads(path.read_text())
    required = {"schema", "kind", "freeze_commit", "protocol_path", "protocol_file_sha256",
        "qc_manifest_path", "QC_manifest_sha256", "data_hashes", "H1_TFs", "intervals", "seeds", "arms",
        "preprocess", "statistics", "training_TF_scope"}
    allowed = required | {"git_commit", "created_utc", "command", "seed", "runtime_seconds"}
    if synthetic_root is not None:
        allowed.add("fixture_root")
    if not required <= set(lock) or set(lock) - allowed:
        raise ValueError("Protocol lock schema has missing or unrecognized fields")
    if lock["schema"] != SCHEMA or lock["training_TF_scope"] != "H1_TFs_only":
        raise ValueError("Unregistered replication schema or training scope")
    if lock["intervals"] != [list(x) for x in INTERVALS] or lock["seeds"] != list(SEEDS) or lock["arms"] != list(CORE_ARMS):
        raise ValueError("Intervals, algorithm seeds and six arms must match the fixed H1 grid")
    if lock["protocol_path"] != PROTOCOL:
        raise ValueError("Unexpected protocol path")
    preprocess = lock["preprocess"]
    stats = lock["statistics"]
    if set(preprocess) != set(PREPROCESS) or set(stats) != {"bootstrap_repetitions", "bootstrap_seed"}:
        raise ValueError("Unregistered preprocessing/statistics fields")
    if type(stats["bootstrap_seed"]) is not int or stats["bootstrap_seed"] < 0:
        raise ValueError("Invalid bootstrap seed")
    synthetic = synthetic_root is not None
    if synthetic:
        fixture_root = Path(synthetic_root).resolve()
        if lock["kind"] != "synthetic_fixture" or resolved(lock["fixture_root"], repo) != fixture_root:
            raise ValueError("Synthetic mode only accepts its explicitly generated fixture root")
        protocol_path = fixture_root / PROTOCOL
        frozen_content = protocol_path.read_bytes()
        if stats["bootstrap_repetitions"] < 100:
            raise ValueError("Synthetic bootstrap budget too small")
    else:
        if lock["kind"] != "formal_replication" or not re.fullmatch(r"[0-9a-f]{40}", lock["freeze_commit"]):
            raise ValueError("Formal prediction requires a full frozen commit; synthetic/draft locks are forbidden")
        if preprocess != PREPROCESS or stats != dict(bootstrap_repetitions=10000, bootstrap_seed=20261017):
            raise ValueError("Formal H1 geometry or bootstrap budget differs from the registered protocol")
        protocol_path = repo / PROTOCOL
        frozen_content = subprocess.check_output(["git", "-C", str(repo), "show",
            f"{lock['freeze_commit']}:{PROTOCOL}"])
    if not re.search(r"^状态：\*\*已冻结\*\*", frozen_content.decode(), flags=re.MULTILINE):
        raise ValueError("Protocol is DRAFT/not frozen; replication expression must remain sealed")
    if hashlib.sha256(frozen_content).hexdigest() != lock["protocol_file_sha256"]:
        raise ValueError("Frozen git protocol hash mismatch")
    if sha256(protocol_path) != lock["protocol_file_sha256"]:
        raise ValueError("Current protocol differs from the locked frozen version")
    qc_path = Path(lock["qc_manifest_path"])
    if not qc_path.is_absolute() or sha256(qc_path) != lock["QC_manifest_sha256"]:
        raise ValueError("QC manifest identity/hash mismatch")
    qc = json.loads(qc_path.read_text())
    tfs = lock["H1_TFs"]
    if (qc.get("status") != "TECHNICAL_COHORT_QC_COMPLETE" or qc.get("biological_effects_read") is not False
            or qc.get("H1_TFs") != tfs or tfs != sorted(set(tfs)) or len(tfs) < 5
            or set(tfs) & {*CONTROL_LABELS, "unassigned"}):
        raise ValueError("QC common-TF grid is incomplete, changed or not technical-only")
    if not synthetic and qc.get("minimum_TF_cells_per_day") != 50:
        raise ValueError("Formal technical eligibility threshold must be 50 cells")
    if set(lock["data_hashes"]) != {"day2", "day3", "day4", "embeddings"}:
        raise ValueError("H1 requires exactly day2/day3/day4 and the frozen embeddings")
    paths = {}
    for name, entry in lock["data_hashes"].items():
        if set(entry) != {"path", "sha256"} or not Path(entry["path"]).is_absolute():
            raise ValueError("Data lock requires absolute paths and SHA256 fields")
        data_path = Path(entry["path"]).resolve()
        if sha256(data_path) != entry["sha256"]:
            raise ValueError(f"Locked data changed: {name}")
        if synthetic and not data_path.is_relative_to(fixture_root):
            raise ValueError("Synthetic smoke cannot access data outside its generated fixture root")
        if name.startswith("day"):
            matches = [digest for p, digest in qc.get("data_hashes", {}).items() if resolved(p, repo) == data_path]
            if matches != [entry["sha256"]]:
                raise ValueError(f"QC did not audit this exact day file: {name}")
            if synthetic:
                # Only a metadata marker is read here; never a count layer.
                with h5py.File(data_path, "r") as f:
                    marker = f.get("uns/w4_synthetic_fixture")
                    if marker is None or marker.asstr()[()] != str(fixture_root):
                        raise ValueError("Missing generated synthetic H5AD fixture marker")
        paths[name] = data_path
    return dict(lock=lock, lock_path=path, lock_sha256=sha256(path), qc=qc, qc_path=qc_path,
        protocol_path=protocol_path, paths=paths, synthetic=synthetic, repo=repo)


def assert_code_unchanged(expected):
    if {str(p): sha256(p) for p in CODE_PATHS} != expected:
        raise ValueError("Replication code changed during the run; refusing to freeze or score")


def input_fingerprint(verified):
    files = [verified["lock_path"], verified["qc_path"], verified["protocol_path"], *verified["paths"].values()]
    return {str(p): (p.stat().st_dev, p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in files}


def assert_inputs_unchanged(verified, initial):
    if input_fingerprint(verified) != initial:
        raise ValueError("Locked inputs changed during execution")


def cohort_metadata(verified):
    metadata = {}
    reference_genes = None
    for day in (2, 3, 4):
        obs, genes = read_metadata(verified["paths"][f"day{day}"])
        if set(obs["day"].astype(int)) != {day} or len(set(obs.index)) != len(obs):
            raise ValueError("Day identity or source cell keys changed")
        if reference_genes is not None and not np.array_equal(genes, reference_genes):
            raise ValueError("Count files do not use the same ordered genes")
        reference_genes = genes
        labels = obs["condition"].astype(str).to_numpy()
        mask = qc_mask(obs)
        counts = {label: int(np.sum(mask & (labels == label))) for label in set(labels)}
        metadata[day] = dict(obs=obs, genes=genes, labels=labels, mask=mask, counts=counts)
    threshold = verified["qc"]["minimum_TF_cells_per_day"]
    eligible = [set(label for label, count in metadata[day]["counts"].items()
        if count >= threshold and label not in (*CONTROL_LABELS, "unassigned")) for day in (2, 3, 4)]
    if sorted(set.intersection(*eligible)) != verified["lock"]["H1_TFs"]:
        raise ValueError("Current technical metadata does not reproduce the locked H1 common-TF set")
    return metadata


def fold_identity(source_day, target_day, index):
    return f"day{source_day}_to_day{target_day}/fold_{index:03d}"


def fit_predict_fold(source_path, target_path, embeddings_path, source_meta, target_meta,
                     tfs, heldout, geometry, directory):
    """Source transforms exclude outer TF; target row loader accepts training TFs only."""
    directory.mkdir(parents=True, exist_ok=False)
    controls = [label for label in CONTROL_LABELS if source_meta["counts"].get(label, 0) >= 1
                and target_meta["counts"].get(label, 0) >= 1]
    train_labels = sorted((set(tfs) - {heldout}) | set(controls))
    labels_s, labels_t = source_meta["labels"], target_meta["labels"]
    si = np.flatnonzero(source_meta["mask"] & np.isin(labels_s, train_labels))
    vi = np.flatnonzero(source_meta["mask"] & (labels_s == heldout))
    ti = np.flatnonzero(target_meta["mask"] & np.isin(labels_t, train_labels))
    if heldout in labels_t[ti] or set(labels_t[ti]) != set(train_labels):
        raise ValueError("Held-out target entered training or a training cohort is missing")
    s, u = read_count_rows(source_path, "spliced", si), read_count_rows(source_path, "unspliced", si)
    sv, uv = read_count_rows(source_path, "spliced", vi), read_count_rows(source_path, "unspliced", vi)
    future_train = read_count_rows(target_path, "spliced", ti)
    source_day = int(source_meta["obs"]["day"].iloc[0])
    ids = np.array([f"day{source_day}:{identity}" for identity in source_meta["obs"].index[si]])
    query_ids = np.array([f"day{source_day}:{identity}" for identity in source_meta["obs"].index[vi]])
    base = SourceTransform.fit(s, u, cell_ids=ids, allowed_ids=set(ids), gene_names=source_meta["genes"],
        n_hvg=geometry["n_hvg"], dimensions=geometry["dimensions"], clusters=geometry["clusters"], seed=SEEDS[0])
    train = population_features(base, s, u, labels_s[si], train_labels, ids, neighbors=geometry["neighbors"])
    test = population_features(base, sv, uv, labels_s[vi], [heldout], query_ids, neighbors=geometry["neighbors"])
    train_targets = group_mean(base.state(future_train), labels_t[ti], train_labels)
    et = condition_embeddings(embeddings_path, train_labels)
    ev = condition_embeddings(embeddings_path, [heldout])
    predictions = {arm: [] for arm in CORE_ARMS}
    selections, swaps, transform_hashes, cached_fits = {}, {}, {}, {}
    actual_fits = 0
    for seed in SEEDS:
        if seed == SEEDS[0]:
            transform = base
        else:
            # Deterministic HVG/PCA/gamma is shared, while each seed's genuinely
            # different kmeans centers are saved. None of the six arms uses them.
            centers = KMeans(n_clusters=geometry["clusters"], n_init=10, random_state=seed).fit(base.reference_z).cluster_centers_
            transform = replace(base, centers=centers, seed=seed)
        transform_path = directory / f"transform_seed{seed}.npz"
        transform.save(transform_path)
        transform_hashes[str(seed)] = sha256(transform_path)
        tr, tr_swap = add_swap_feature(transform, train, train, seed)
        te, te_swap = add_swap_feature(transform, test, train, seed)
        selections[str(seed)] = {}
        swaps[str(seed)] = dict(train=tr_swap, heldout_source=te_swap)
        for arm in CORE_ARMS:
            a, b = design(tr, et, arm), design(te, ev, arm)
            signature = (arm, hashlib.sha256(a.tobytes() + b.tobytes()).hexdigest())
            reused = signature in cached_fits
            if not reused:
                pred, selection = predict_arms(tr, te, train_targets, et, ev,
                    arms=(arm,), controls=np.isin(train_labels, CONTROL_LABELS))
                cached_fits[signature] = pred[arm], selection[arm]
                actual_fits += 1
            pred, selection = cached_fits[signature]
            predictions[arm].append(pred.copy())
            selections[str(seed)][arm] = dict(selection, reused_identical_design=bool(reused))
    prediction_path = directory / "predictions.npz"
    np.savez_compressed(prediction_path, labels=np.array([heldout]), seeds=np.array(SEEDS),
        **{arm: np.stack(values) for arm, values in predictions.items()})
    return dict(heldout_TF=heldout, training_TFs=sorted(set(tfs)-{heldout}), controls=controls,
        training_source_rows=si.tolist(), heldout_source_rows=vi.tolist(), training_target_rows=ti.tolist(),
        training_source_ids=ids.tolist(), heldout_source_ids=query_ids.tolist(),
        heldout_future_expression_used=False, source_fit_uses_future=False,
        training_targets_used_for_response_ridge=True, geometry_seed_reuse="deterministic PCA/gamma; separate kmeans per seed",
        actual_ridge_fits=actual_fits, potential_seed_arm_fits=len(SEEDS)*len(CORE_ARMS),
        prediction_sha256=sha256(prediction_path), transform_sha256=transform_hashes,
        selections=selections, U_swaps=swaps)


def predict_grid(protocol_lock, output, *, repo=None, synthetic_root=None):
    verified = verify_lock(protocol_lock, repo=repo, synthetic_root=synthetic_root)
    assert_code_unchanged(IMPORTED_CODE_HASHES)
    initial = input_fingerprint(verified)
    inputs = [*CODE_PATHS, verified["lock_path"], verified["qc_path"], verified["protocol_path"], *verified["paths"].values()]
    run = Artifacts(output, seed=list(SEEDS), inputs=inputs)
    try:
        metadata = cohort_metadata(verified)
        records = []
        for source_day, target_day in INTERVALS:
            for index, tf in enumerate(verified["lock"]["H1_TFs"]):
                assert_code_unchanged(IMPORTED_CODE_HASHES)
                assert_inputs_unchanged(verified, initial)
                relative = fold_identity(source_day, target_day, index)
                directory = run.directory / relative
                directory.parent.mkdir(parents=True, exist_ok=True)
                result = fit_predict_fold(verified["paths"][f"day{source_day}"], verified["paths"][f"day{target_day}"],
                    verified["paths"]["embeddings"], metadata[source_day], metadata[target_day],
                    verified["lock"]["H1_TFs"], tf, verified["lock"]["preprocess"], directory)
                run.json(f"{relative}/fold_manifest.json", dict(status="FOLD_PREDICTIONS_FROZEN",
                    source_day=source_day, target_day=target_day, **result))
                records.append(dict(source_day=source_day, target_day=target_day, TF=tf, directory=relative,
                    fold_manifest_sha256=sha256(directory / "fold_manifest.json")))
                run.json("status.json", dict(status="predicting", completed_folds=len(records)))
                print(json.dumps(dict(status="fold_frozen", interval=[source_day, target_day], TF=tf)), flush=True)
        assert_code_unchanged(IMPORTED_CODE_HASHES)
        assert_inputs_unchanged(verified, initial)
        verify_lock(protocol_lock, repo=repo, synthetic_root=synthetic_root)
        manifest = dict(status="H1_GRID_PREDICTIONS_FROZEN", kind=verified["lock"]["kind"],
            protocol_lock_sha256=verified["lock_sha256"], protocol_freeze_commit=verified["lock"]["freeze_commit"],
            H1_TFs=verified["lock"]["H1_TFs"], intervals=[list(x) for x in INTERVALS],
            arms=list(CORE_ARMS), algorithm_seeds=list(SEEDS), records=records,
            code_hashes=IMPORTED_CODE_HASHES, complete_grid_size=len(records),
            per_fold_heldout_future_expression_used=False, cross_fold_training_future_labels_are_legal=True)
        run.json("grid_manifest.json", manifest)
        run.json("freeze_receipt.json", dict(status="COMPLETE_GRID_FROZEN",
            grid_manifest_sha256=sha256(run.directory / "grid_manifest.json"),
            protocol_lock_sha256=verified["lock_sha256"], complete_grid_size=len(records)))
        run.json("status.json", dict(status="complete", completed_folds=len(records)))
        return manifest
    except BaseException as error:
        run.json("status.json", dict(status="failed", error=f"{type(error).__name__}: {error}"))
        raise


def verify_frozen_grid(directory, verified):
    """Check the WHOLE grid and all artifacts before reading any target rows."""
    directory = Path(directory).resolve()
    receipt = json.loads((directory / "freeze_receipt.json").read_text())
    manifest = json.loads((directory / "grid_manifest.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    if (receipt["status"] != "COMPLETE_GRID_FROZEN" or status["status"] != "complete"
            or sha256(directory / "grid_manifest.json") != receipt["grid_manifest_sha256"]
            or receipt["protocol_lock_sha256"] != verified["lock_sha256"]
            or manifest["protocol_lock_sha256"] != verified["lock_sha256"]):
        raise ValueError("Prediction grid is incomplete, modified or bound to a different protocol lock")
    if (manifest["status"] != "H1_GRID_PREDICTIONS_FROZEN" or manifest["H1_TFs"] != verified["lock"]["H1_TFs"]
            or manifest["intervals"] != [list(x) for x in INTERVALS] or manifest["arms"] != list(CORE_ARMS)
            or manifest["algorithm_seeds"] != list(SEEDS) or manifest["per_fold_heldout_future_expression_used"]):
        raise ValueError("Frozen grid contract mismatch")
    expected = {(a, b, tf): fold_identity(a, b, i) for a, b in INTERVALS
                for i, tf in enumerate(verified["lock"]["H1_TFs"])}
    records = manifest["records"]
    if len(records) != len(expected) or manifest["complete_grid_size"] != len(expected) or receipt["complete_grid_size"] != len(expected):
        raise ValueError("Entire registered TF-by-interval grid is required")
    seen, bundles = set(), []
    assert_code_unchanged(manifest["code_hashes"])
    for path, digest in manifest["code_hashes"].items():
        if sha256(directory / "code_snapshot" / Path(path).name) != digest:
            raise ValueError("Code snapshot changed after prediction")
    for record in records:
        identity = (record["source_day"], record["target_day"], record["TF"])
        if identity in seen or identity not in expected or record["directory"] != expected[identity]:
            raise ValueError("Missing, duplicate or unexpected frozen-grid identity")
        seen.add(identity)
        fold = directory / record["directory"]
        if sha256(fold / "fold_manifest.json") != record["fold_manifest_sha256"]:
            raise ValueError("Fold manifest changed after grid freeze")
        fm = json.loads((fold / "fold_manifest.json").read_text())
        if ((fm["source_day"], fm["target_day"], fm["heldout_TF"]) != identity
                or fm["heldout_future_expression_used"] or fm["source_fit_uses_future"]
                or fm["training_TFs"] != sorted(set(verified["lock"]["H1_TFs"]) - {identity[2]})):
            raise ValueError("Per-fold holdout/training contract changed")
        if sha256(fold / "predictions.npz") != fm["prediction_sha256"]:
            raise ValueError("Predictions changed after freezing")
        transforms = []
        for seed in SEEDS:
            path = fold / f"transform_seed{seed}.npz"
            if sha256(path) != fm["transform_sha256"][str(seed)]:
                raise ValueError("Source transform changed after freezing")
            transform = SourceTransform.load(path)
            if (transform.seed != seed or set(transform.source_ids) != set(fm["training_source_ids"])
                    or set(transform.source_ids) & set(fm["heldout_source_ids"])):
                raise ValueError("Source fit IDs/seed violate the outer holdout")
            if transforms:
                for key in ("genes", "mean", "components", "gamma", "gene_names"):
                    np.testing.assert_array_equal(getattr(transform, key), getattr(transforms[0], key))
            transforms.append(transform)
        with np.load(fold / "predictions.npz", allow_pickle=False) as p:
            if set(p.files) != {"labels", "seeds", *CORE_ARMS}:
                raise ValueError("Prediction arms do not match the frozen grid")
            np.testing.assert_array_equal(p["labels"], [identity[2]])
            np.testing.assert_array_equal(p["seeds"], SEEDS)
            shape = (len(SEEDS), 1, verified["lock"]["preprocess"]["dimensions"])
            if any(p[arm].shape != shape or not np.isfinite(p[arm]).all() for arm in CORE_ARMS):
                raise ValueError("Invalid seed/TF prediction shape or nonfinite prediction")
        bundles.append((record, fm, transforms[0]))
    return manifest, bundles


def evaluate_grid(protocol_lock, predictions, output, *, repo=None, synthetic_root=None):
    verified = verify_lock(protocol_lock, repo=repo, synthetic_root=synthetic_root)
    manifest, bundles = verify_frozen_grid(predictions, verified)
    initial = input_fingerprint(verified)
    directory = Path(predictions).resolve()
    inputs = [*CODE_PATHS, verified["lock_path"], verified["qc_path"], verified["protocol_path"],
        directory / "grid_manifest.json", directory / "freeze_receipt.json", *verified["paths"].values()]
    run = Artifacts(output, seed=list(SEEDS), inputs=inputs)
    try:
        metadata = cohort_metadata(verified)
        tfs = verified["lock"]["H1_TFs"]
        errors = {arm: np.empty((len(tfs), len(INTERVALS))) for arm in CORE_ARMS}
        rows = []
        for record, _, transform in bundles:
            assert_code_unchanged(manifest["code_hashes"])
            assert_inputs_unchanged(verified, initial)
            day, tf = record["target_day"], record["TF"]
            meta = metadata[day]
            np.testing.assert_array_equal(meta["genes"], transform.gene_names)
            index = np.flatnonzero(meta["mask"] & (meta["labels"] == tf))
            # All frozen files have been checked before the first heldout read.
            target = read_count_rows(verified["paths"][f"day{day}"], "spliced", index)
            truth = transform.state(target).mean(0, keepdims=True)
            interval = INTERVALS.index((record["source_day"], day))
            with np.load(directory / record["directory"] / "predictions.npz", allow_pickle=False) as p:
                for arm in CORE_ARMS:
                    error = float(average_predictions_then_errors(p[arm], truth)[0])
                    errors[arm][tfs.index(tf), interval] = error
                    rows.append(dict(TF=tf, source_day=record["source_day"], target_day=day, arm=arm,
                        sse=error, mse=error/truth.shape[1], target_cells=len(index)))
        # Changes during evaluation invalidate the run rather than becoming a
        # result tied to stale manifests or changed source/target files.
        verify_frozen_grid(predictions, verified)
        verify_lock(protocol_lock, repo=repo, synthetic_root=synthetic_root)
        stats = verified["lock"]["statistics"]
        comparisons = {ref: paired_gain_test(errors["vel_pool"], errors[ref], errors["static"],
            seed=stats["bootstrap_seed"], repetitions=stats["bootstrap_repetitions"])
            for ref in ("static", "vel_pool_Uswap")}
        summary = dict(status="H1_REPLICATION_EVALUATED" if not verified["synthetic"] else "SYNTHETIC_H1_PIPELINE_EVALUATED",
            kind=verified["lock"]["kind"], H1_TFs=tfs, n_TFs=len(tfs), intervals=[list(x) for x in INTERVALS],
            arms=list(CORE_ARMS), algorithm_seeds=list(SEEDS), seeds_averaged_before_scoring=True,
            independent_inference_unit="TF; paired intervals averaged within TF; seeds/cells are not independent units",
            primary_comparisons=comparisons, H1_supported=all(v["supported"] for v in comparisons.values()),
            mean_TF_interval_sse={arm: float(error.mean()) for arm, error in errors.items()},
            mean_TF_relative_gain_over_static={arm: float(((errors["static"]-error)/np.maximum(errors["static"], 1e-12)).mean())
                                              for arm, error in errors.items()},
            frozen_grid_sha256=sha256(directory / "grid_manifest.json"),
            protocol_lock_sha256=verified["lock_sha256"], formal_biological_claim=not verified["synthetic"])
        with (run.directory / "metrics.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        np.savez_compressed(run.directory / "TF_interval_errors.npz", TFs=np.array(tfs), intervals=np.array(INTERVALS), **errors)
        run.json("summary.json", summary)
        run.json("status.json", dict(status="complete"))
        (run.directory / "RESULTS.md").write_text("# W4 locked H1 evaluation\n\n" + json.dumps(summary, indent=2) + "\n")
        return summary
    except BaseException as error:
        run.json("status.json", dict(status="failed", error=f"{type(error).__name__}: {error}"))
        raise


def create_synthetic_fixture(root, *, seed=20261017):
    """Generate every input locally; never import or copy a biological dataset."""
    import anndata as ad
    import pandas as pd
    from scipy import sparse

    started = time.monotonic()
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    tfs = [f"SYNTH_TF{i}" for i in range(5)]
    labels = tfs + list(CONTROL_LABELS)
    cells, genes = 6, 8
    baseline = rng.uniform(20, 50, (len(labels), genes))
    innovation = rng.normal(0, 2, baseline.shape)
    paths = {}
    for day in (2, 3, 4):
        means = baseline + (day-2)*innovation
        s = rng.poisson(np.repeat(means, cells, axis=0)).astype(float)
        u = rng.poisson(.7*s + 5 + np.repeat(innovation, cells, axis=0)).astype(float)
        obs = pd.DataFrame(dict(condition=np.repeat(labels, cells), day=day,
            role=np.repeat(["train"]*len(tfs) + ["control"]*len(CONTROL_LABELS), cells),
            technical_qc_pass=True, guide_call_pass=True), index=[f"synth_day{day}_cell{i}" for i in range(len(s))])
        a = ad.AnnData(sparse.csr_matrix(s), obs=obs, var=pd.DataFrame(index=[f"SYNTH_GENE{i}" for i in range(genes)]))
        a.layers["spliced"], a.layers["unspliced"] = sparse.csr_matrix(s), sparse.csr_matrix(u)
        a.uns["w4_synthetic_fixture"] = str(root)
        paths[f"day{day}"] = root / f"day{day}.h5ad"
        a.write_h5ad(paths[f"day{day}"])
    paths["embeddings"] = root / "embeddings.npz"
    np.savez_compressed(paths["embeddings"], conditions=np.array(tfs), embeddings=rng.normal(size=(5, 4)),
        metadata_json=json.dumps(dict(frozen=True, source="generated synthetic fixture, not ESM2")))
    protocol = root / PROTOCOL; protocol.parent.mkdir()
    protocol.write_text("# Synthetic-only protocol fixture\n\n状态：**已冻结**\nNot a biological preregistration.\n")
    qc_path = root / "qc_manifest.json"
    qc = dict(status="TECHNICAL_COHORT_QC_COMPLETE", biological_effects_read=False, H1_TFs=tfs,
        minimum_TF_cells_per_day=cells, data_hashes={str(p): sha256(p) for k, p in paths.items() if k.startswith("day")},
        seed=seed, git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        command=sys.argv, runtime_seconds=time.monotonic()-started)
    qc_path.write_text(json.dumps(qc, indent=2) + "\n")
    lock = dict(schema=SCHEMA, kind="synthetic_fixture", freeze_commit="SYNTHETIC_ONLY_NO_GIT_FREEZE",
        protocol_path=PROTOCOL, protocol_file_sha256=sha256(protocol), qc_manifest_path=str(qc_path),
        QC_manifest_sha256=sha256(qc_path), data_hashes={k: dict(path=str(p), sha256=sha256(p)) for k, p in paths.items()},
        H1_TFs=tfs, intervals=[list(x) for x in INTERVALS], seeds=list(SEEDS), arms=list(CORE_ARMS),
        preprocess=dict(n_hvg=8, dimensions=3, clusters=2, neighbors=3),
        statistics=dict(bootstrap_repetitions=1000, bootstrap_seed=seed), training_TF_scope="H1_TFs_only", fixture_root=str(root),
        git_commit=qc["git_commit"], seed=seed, command=sys.argv, runtime_seconds=time.monotonic()-started)
    lock_path = root / "protocol_lock.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    return lock_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    predict = sub.add_parser("predict"); evaluate = sub.add_parser("evaluate")
    for command in (predict, evaluate):
        command.add_argument("--protocol-lock", required=True)
        command.add_argument("--output", required=True)
    evaluate.add_argument("--predictions", required=True)
    smoke = sub.add_parser("synthetic-smoke"); smoke.add_argument("--output", required=True)
    args = p.parse_args()
    if args.stage == "predict":
        result = predict_grid(args.protocol_lock, args.output)
    elif args.stage == "evaluate":
        result = evaluate_grid(args.protocol_lock, args.predictions, args.output)
    else:
        root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=False)
        fixture = root / "fixture"
        lock = create_synthetic_fixture(fixture)
        predict_grid(lock, root / "predict", synthetic_root=fixture)
        result = evaluate_grid(lock, root / "predict", root / "evaluate", synthetic_root=fixture)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
