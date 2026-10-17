"""Synthetic calibration of W4's complete paired-TF, two-interval H1 design.

Unlike w4_synthetic's independent test-population fixture, each repetition here
uses outer leave-one-TF-out source fitting and inner leave-one-training-TF
ridge selection. The full H1 intersection decision is applied after averaging
the four algorithmic seeds and then the two intervals within TF.

This is assumption-conditional simulation, not an empirical RENGE power
estimate. ``formal`` requires >=200 repetitions, 10000 bootstraps and a
technical-QC manifest containing H1_TFs and biological_effects_read=false
(the explicit legacy alias h1_eligible_tfs is also accepted).
Simulation gene dimensions, noise and embeddings remain explicit assumptions.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import binomtest

from w4_artifacts import Artifacts, sha256
from w4_arms import average_predictions_then_errors, paired_gain_test, predict_arms
from w4_streams import SourceTransform, add_swap_feature, group_mean, pooled_counts


MAIN_ARMS = ("static", "vel_pool", "vel_pool_Uswap")
ALGORITHM_SEEDS = (0, 1, 2, 3)
RHO_GRID = (0., .5, .9)
CONTROLS = ("AAVS1", "CTRL")


@dataclass(frozen=True)
class Geometry:
    n_tfs: int = 8
    cells: int = 12
    genes: int = 12
    dimensions: int = 6
    clusters: int = 4
    embedding_dimensions: int = 8
    library: int = 10000
    innovation_amplitude: float = .4
    response_strength: float = .5
    cell_log_sd: float = .02

    def validate(self):
        if self.n_tfs < 5 or self.cells < 2:
            raise ValueError("At least 5 TFs and 2 independently sampled cells per group are required")
        if not 1 <= self.dimensions < self.genes or self.embedding_dimensions < 1:
            raise ValueError("Invalid gene, PCA or embedding dimensions")
        if self.clusters < 1 or self.clusters > (self.n_tfs + len(CONTROLS) - 1) * self.cells:
            raise ValueError("Invalid source cluster count")
        if self.library < self.genes or not 0 < self.innovation_amplitude < 1:
            raise ValueError("Invalid library or innovation amplitude")
        if not 0 <= self.response_strength * self.innovation_amplitude < 1 or self.cell_log_sd < 0:
            raise ValueError("Invalid response or cell noise")


def array_hash(*arrays) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def correlated_latents(rng, n_tfs: int, genes: int, rho: float):
    if not 0 <= rho <= 1:
        raise ValueError("rho must be between 0 and 1")
    first = rng.normal(size=(n_tfs, genes))
    second = rho * first + np.sqrt(1 - rho**2) * rng.normal(size=first.shape)
    return np.stack((first, second))


def simulate_population(seed: int, kind: str, rho: float, geometry: Geometry, *, innovation_seed=None):
    """Three independent destructive snapshots of evolving TF populations.

    The null's future population drift is independent of U innovations. U is
    sampled conditional on OBSERVED current S and independent innovations, so
    it is not a second measurement of a hidden current S state. Across adjacent
    intervals, TF-level latent innovations (and separately null drifts) have
    correlation rho before tanh and composition normalization.
    """
    geometry.validate()
    if kind not in ("positive", "null"):
        raise ValueError("Expected positive or null scenario")
    if seed < 0 or (innovation_seed is not None and innovation_seed < 0):
        raise ValueError("Seeds must be nonnegative")
    g = geometry
    labels = np.array([f"TF{i:03d}" for i in range(g.n_tfs)] + list(CONTROLS))
    groups = np.repeat(labels, g.cells)
    n_groups = len(labels)
    state_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    u_rng = np.random.default_rng(np.random.SeedSequence([seed if innovation_seed is None else innovation_seed, 2]))
    drift_rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    latent_u = correlated_latents(u_rng, g.n_tfs, g.genes, rho)
    latent_drift = correlated_latents(drift_rng, g.n_tfs, g.genes, rho)
    u_effect = np.concatenate((np.tanh(latent_u), np.zeros((2, len(CONTROLS), g.genes))), axis=1)
    drift = np.concatenate((np.tanh(latent_drift), np.zeros((2, len(CONTROLS), g.genes))), axis=1)
    initial = np.exp(state_rng.normal(0, .15, (n_groups, g.genes)))
    initial /= initial.sum(1, keepdims=True)
    profiles = [initial]
    for interval in range(2):
        effect = u_effect[interval] if kind == "positive" else drift[interval]
        future = profiles[-1] * (1 + g.response_strength * g.innovation_amplitude * effect)
        future /= future.sum(1, keepdims=True)
        profiles.append(future)
    s, u, ids = [], [], []
    for time_index, profile in enumerate(profiles):
        cell_rng = np.random.default_rng(np.random.SeedSequence([seed, 10 + time_index]))
        per_cell = np.repeat(profile, g.cells, axis=0)
        per_cell *= np.exp(cell_rng.normal(0, g.cell_log_sd, per_cell.shape))
        per_cell /= per_cell.sum(1, keepdims=True)
        # Integer counts and fresh cell sampling at every snapshot. No source
        # cell is moved forward to manufacture a paired future cell.
        counts = np.stack([cell_rng.multinomial(g.library, p) for p in per_cell])
        s.append(counts)
        ids.append(np.array([f"t{time_index}_{label}_cell{cell}" for label in labels for cell in range(g.cells)]))
        if time_index < 2:
            count_rng = np.random.default_rng(np.random.SeedSequence([seed, 20 + time_index]))
            u_rate = .8 * counts * (1 + g.innovation_amplitude * np.repeat(u_effect[time_index], g.cells, axis=0))
            u.append(count_rng.poisson(u_rate))
    embeddings = state_rng.normal(size=(n_groups, g.embedding_dimensions))
    embeddings[-len(CONTROLS):] = 0.
    return dict(s=s, u=u, ids=ids, labels=labels, groups=groups,
        gene_names=np.array([f"gene{i:04d}" for i in range(g.genes)]),
        embeddings=embeddings, latent_u=latent_u, profiles=np.stack(profiles),
        input_sha256=array_hash(*s, *u, embeddings),
        shared_snapshot_semantics="time1 same observed snapshot is interval0 target and interval1 source",
        independent_cells_across_time=True)


def core_features(transform, s, u, groups, labels):
    """Only features consumed by the three primary arms; no kNN/pseudo-S work."""
    sp, up = pooled_counts(s, groups, labels), pooled_counts(u, groups, labels)
    sn, un = transform.normalized(sp, up)
    return dict(labels=np.asarray(labels), state=group_mean(transform.state(s), groups, labels),
        vel_pool=transform.projected_rate(sn, un - transform.gamma * sn), pooled_s=sp, pooled_u=up)


def fit_outer_fold(source_s, source_u, source_groups, source_ids, gene_names,
                   train_future_s, train_future_groups, labels, embeddings, heldout, geometry,
                   *, cache_identical_predictions=True):
    """No held-out future counts are accepted by this fit/predict interface."""
    labels, source_groups, source_ids = map(np.asarray, (labels, source_groups, source_ids))
    train_future_groups = np.asarray(train_future_groups)
    if heldout in CONTROLS or heldout not in labels or heldout in train_future_groups:
        raise ValueError("Illegal held-out TF or held-out future counts entered training")
    selected = labels != heldout
    train_labels, test_labels = labels[selected], labels[~selected]
    if set(train_future_groups) != set(train_labels):
        raise ValueError("Training target groups do not match the outer training labels")
    train_rows = source_groups != heldout
    test_rows = ~train_rows
    transform = SourceTransform.fit(source_s[train_rows], source_u[train_rows],
        cell_ids=source_ids[train_rows], allowed_ids=set(source_ids[train_rows]), gene_names=gene_names,
        n_hvg=geometry.genes, dimensions=geometry.dimensions, clusters=geometry.clusters, seed=0)
    train = core_features(transform, source_s[train_rows], source_u[train_rows], source_groups[train_rows], train_labels)
    test = core_features(transform, source_s[test_rows], source_u[test_rows], source_groups[test_rows], test_labels)
    # These future labels are exclusively the OUTER TRAINING TFs + controls.
    target_train = group_mean(transform.state(train_future_s), train_future_groups, train_labels)
    predictions = {arm: [] for arm in MAIN_ARMS}
    cache, selections, audits = {}, [], []
    ridge_fits = 0
    for seed in ALGORITHM_SEEDS:
        train_swapped, train_swap_audit = add_swap_feature(transform, train, train, seed)
        test_swapped, test_swap_audit = add_swap_feature(transform, test, train, seed)
        audits += [dict(algorithm_seed=seed, side="train", **a) for a in train_swap_audit]
        audits += [dict(algorithm_seed=seed, side="test", **a) for a in test_swap_audit]
        for arm in MAIN_ARMS:
            # State, ESM-like features, targets and source PCA/gamma are fixed
            # across seeds. k-means centers are not consumed by these arms.
            # Only swapped-U ties can change the arm-specific feature block.
            signature = (arm, array_hash(train_swapped[arm], test_swapped[arm]) if arm != "static" else "static")
            cached = cache_identical_predictions and signature in cache
            if not cached:
                pred, selection = predict_arms(train_swapped, test_swapped, target_train,
                    embeddings[selected], embeddings[~selected], arms=(arm,), controls=np.isin(train_labels, CONTROLS))
                cache[signature] = (pred[arm], selection[arm])
                ridge_fits += 1
            prediction, selection = cache[signature]
            predictions[arm].append(prediction.copy())
            selections.append(dict(seed=seed, arm=arm, cached=bool(cached), **selection))
    stacked = {arm: np.stack(value) for arm, value in predictions.items()}
    audit = dict(heldout=str(heldout), source_fit_sha256=transform.fingerprint(),
        fit_source_ids_sha256=array_hash(transform.source_ids),
        excluded_source_ids_sha256=array_hash(source_ids[test_rows]),
        heldout_source_in_fit=bool(set(source_ids[test_rows]) & set(transform.source_ids)),
        heldout_future_in_training=False, training_tfs=int((~np.isin(train_labels, CONTROLS)).sum()),
        controls=int(np.isin(train_labels, CONTROLS).sum()), outer_source_fits=1,
        algorithm_seeds=list(ALGORITHM_SEEDS), fitted_kmeans_seed=0, kmeans_consumed_by_primary_arms=False,
        ridge_fits=ridge_fits, potential_seed_arm_fits=len(ALGORITHM_SEEDS)*len(MAIN_ARMS),
        seed_prediction_sha256=array_hash(*(stacked[arm] for arm in MAIN_ARMS)),
        nonunique_swap_nearest_neighbors=sum(a["ties"] > 1 for a in audits), selections=selections,
        swaps=audits)
    return stacked, transform, audit


def h1_decision(errors: dict[str, np.ndarray], *, bootstrap_seed: int, repetitions: int):
    if set(errors) != set(MAIN_ARMS):
        raise ValueError("H1 requires exactly the registered three arms")
    if any(np.asarray(v).ndim != 2 or np.asarray(v).shape[1] != 2 for v in errors.values()):
        raise ValueError("H1 errors must have two fixed intervals per TF")
    comparisons = {ref: paired_gain_test(errors["vel_pool"], errors[ref], errors["static"],
        seed=bootstrap_seed, repetitions=repetitions) for ref in ("static", "vel_pool_Uswap")}
    return dict(detected=all(c["supported"] for c in comparisons.values()), comparisons=comparisons,
        inference_unit="TF_after_seed_prediction_averaging_then_two_interval_gain_averaging")


def one_replicate(seed: int, kind: str, rho: float, geometry: Geometry, *, bootstrap=10000,
                  cache_identical_predictions=True):
    start = time.monotonic()
    data = simulate_population(seed, kind, rho, geometry)
    pending, fold_audits = [], []
    for interval in range(2):
        for tf_index, tf in enumerate(data["labels"][:geometry.n_tfs]):
            train = data["groups"] != tf
            prediction, transform, audit = fit_outer_fold(data["s"][interval], data["u"][interval],
                data["groups"], data["ids"][interval], data["gene_names"],
                data["s"][interval+1][train], data["groups"][train], data["labels"],
                data["embeddings"], tf, geometry, cache_identical_predictions=cache_identical_predictions)
            pending.append((interval, tf_index, tf, prediction, transform))
            fold_audits.append(dict(interval=interval, **audit))
    # All outer predictions are now fixed in memory. Only this scoring pass
    # accesses each fold's held-out target; other folds legally trained on it.
    errors = {arm: np.empty((geometry.n_tfs, 2)) for arm in MAIN_ARMS}
    for interval, tf_index, tf, prediction, transform in pending:
        rows = data["groups"] == tf
        truth = transform.state(data["s"][interval+1][rows]).mean(0, keepdims=True)
        for arm in MAIN_ARMS:
            errors[arm][tf_index, interval] = average_predictions_then_errors(prediction[arm], truth)[0]
    decision = h1_decision(errors, bootstrap_seed=seed, repetitions=bootstrap)
    row = dict(seed=seed, kind=kind, rho=rho, n_tfs=geometry.n_tfs, detected=decision["detected"],
        input_sha256=data["input_sha256"], errors_sha256=array_hash(*(errors[a] for a in MAIN_ARMS)),
        outer_source_fits=len(fold_audits), ridge_fits=sum(a["ridge_fits"] for a in fold_audits),
        potential_seed_arm_fits=sum(a["potential_seed_arm_fits"] for a in fold_audits),
        nonunique_swap_nearest_neighbors=sum(a["nonunique_swap_nearest_neighbors"] for a in fold_audits),
        runtime_seconds=time.monotonic()-start)
    for ref, stats in decision["comparisons"].items():
        for name in ("gain", "ci_low", "ci_high", "wilcoxon_p", "supported", "independent_TFs", "denominator_floors"):
            row[f"{ref}_{name}"] = stats[name]
        row[f"{ref}_interval0_gain"], row[f"{ref}_interval1_gain"] = stats["by_interval"]
    for fold in fold_audits:
        fold.update(replicate_seed=seed, kind=kind, rho=rho)
    return row, fold_audits


def binomial_rate(successes: int, repetitions: int):
    if not 0 <= successes <= repetitions or repetitions < 1:
        raise ValueError("Invalid binomial totals")
    interval = binomtest(successes, repetitions).proportion_ci(confidence_level=.95, method="exact")
    return dict(events=int(successes), repetitions=int(repetitions), rate=float(successes/repetitions),
        ci_low=float(interval.low), ci_high=float(interval.high), interval="95% Clopper-Pearson")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    p.add_argument("--n-tfs", type=int, default=8)
    p.add_argument("--replicates", type=int, default=2)
    p.add_argument("--rho-grid", default="0,0.5,0.9")
    p.add_argument("--cells", type=int, default=12)
    p.add_argument("--genes", type=int, default=12)
    p.add_argument("--dimensions", type=int, default=6)
    p.add_argument("--clusters", type=int, default=4)
    p.add_argument("--embedding-dimensions", type=int, default=8)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261017)
    p.add_argument("--qc-manifest", type=Path)
    p.add_argument("--no-prediction-cache", action="store_true")
    return p


def validate_args(args):
    rho = tuple(float(x) for x in args.rho_grid.split(","))
    if rho != RHO_GRID:
        raise ValueError("The fixed latent-correlation grid is 0,0.5,0.9")
    if args.replicates < 1 or args.bootstrap < 100 or args.seed < 0:
        raise ValueError("Invalid repetition count, bootstrap budget or seed")
    if args.mode == "formal":
        if args.replicates < 200 or args.bootstrap < 10000:
            raise ValueError("Formal design calibration requires >=200 repetitions and >=10000 bootstraps")
        if args.qc_manifest is None:
            raise ValueError("Formal calibration requires the technical-QC common-TF manifest")
        qc = json.loads(args.qc_manifest.read_text())
        tfs = qc.get("H1_TFs", qc.get("h1_eligible_tfs", []))
        if len(tfs) != args.n_tfs or len(set(tfs)) != len(tfs) or qc.get("biological_effects_read") is not False:
            raise ValueError("QC manifest must bind n_tfs to unique H1_TFs without biological effect reads")
        if "H1_TFs" in qc and qc.get("status") != "TECHNICAL_COHORT_QC_COMPLETE":
            raise ValueError("Current H1_TFs schema requires a complete technical-cohort QC status")
    geometry = Geometry(n_tfs=args.n_tfs, cells=args.cells, genes=args.genes,
        dimensions=args.dimensions, clusters=args.clusters, embedding_dimensions=args.embedding_dimensions)
    geometry.validate()
    return geometry, rho


def main():
    args = parser().parse_args()
    geometry, rhos = validate_args(args)
    inputs = [Path(__file__), Path(__file__).with_name("w4_streams.py"),
        Path(__file__).with_name("w4_arms.py"), Path(__file__).with_name("w4_artifacts.py")]
    if args.qc_manifest is not None:
        inputs.append(args.qc_manifest)
    run = Artifacts(args.output, seed=args.seed, inputs=inputs)
    run.json("config.json", dict(mode=args.mode, geometry=geometry.__dict__, rho_grid=list(rhos),
        replicates=args.replicates, bootstrap=args.bootstrap, algorithm_seeds=list(ALGORITHM_SEEDS),
        primary_arms=list(MAIN_ARMS), prediction_cache=not args.no_prediction_cache,
        qc_manifest=str(args.qc_manifest) if args.qc_manifest else None,
        formal_does_not_mean_empirical_RENGE_power=True))
    rows = []
    try:
        # JSONL records also carry common execution/provenance context.
        with (run.directory / "fold_audits.jsonl").open("x") as audit_file:
            for kind in ("positive", "null"):
                for rho in rhos:
                    for replicate in range(args.replicates):
                        row, audits = one_replicate(args.seed+replicate, kind, rho, geometry,
                            bootstrap=args.bootstrap, cache_identical_predictions=not args.no_prediction_cache)
                        rows.append(row)
                        for audit in audits:
                            audit_file.write(json.dumps(dict(run.context,
                                runtime_seconds=time.monotonic()-run.started, **audit), allow_nan=False) + "\n")
                        audit_file.flush()
                        run.json("status.json", dict(status="running", kind=kind, rho=rho,
                            completed_in_scenario=replicate+1, completed_replicates=len(rows)))
                        print(json.dumps(dict(kind=kind, rho=rho, replicate=replicate,
                            detected=row["detected"], runtime_seconds=row["runtime_seconds"])), flush=True)
        with (run.directory / "replicates.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        rates = []
        for kind in ("positive", "null"):
            for rho in rhos:
                subset = [r for r in rows if r["kind"] == kind and r["rho"] == rho]
                rates.append(dict(kind=kind, rho=rho, metric="detection" if kind == "positive" else "false_positive",
                    **binomial_rate(sum(r["detected"] for r in subset), len(subset))))
        summary = dict(status="complete", mode=args.mode, complete_H1_decision_design=True,
            adequate_formal_replication_count=args.replicates >= 200,
            formal_QC_count_bound=args.mode == "formal", empirical_RENGE_power_estimate=False,
            rates=rates, geometry=geometry.__dict__, primary_arms=list(MAIN_ARMS), algorithm_seeds=list(ALGORITHM_SEEDS),
            source_geometry_fit="outer training source TFs and controls only; shared unsupervised outer geometry in inner CV",
            response_fit="inner leave-one-training-TF selection, scaler refit inside each inner fold",
            scoring="average seed predictions, score in each fold geometry, average two interval gains within TF",
            h1_rule="both comparisons: 95% TF bootstrap CI lower>0, one-sided Wilcoxon p<.05, both interval means>0",
            random_effects="rho is pre-tanh latent innovation correlation; null drift has same rho but is independent of U",
            common_random_numbers_across_scenarios=True, independent_cells_across_time=True,
            mid_snapshot_reused_as_adjacent_target_and_source=True,
            predictions_frozen_in_memory_before_scoring=True,
            prediction_cache_equivalence="only identical arm feature blocks reuse fits; kmeans centers unused; swap ties recomputed per seed",
            synthetic_embeddings="independent random vectors, zero controls; not measured ESM2 embeddings",
            simulation_geometry_matches_registered_PCA50_HVG2000=geometry.dimensions == 50 and geometry.genes == 2000,
            biological_effects_read=False, gpu_used=False,
            csv_sha256=sha256(run.directory / "replicates.csv"),
            fold_audit_sha256=sha256(run.directory / "fold_audits.jsonl"))
        run.json("summary.json", summary)
        lines = ["# W4 complete-H1 synthetic design calibration", "",
            f"Mode: **{args.mode}**; {args.n_tfs} TFs; {args.replicates} repetitions per scenario; "
            f"{args.bootstrap} TF bootstrap resamples per comparison.", "",
            "This runs the two-interval outer-LOTO/inner-CV H1 decision design. "
            "A smoke run is not a power/FPR pass. Even formal runs estimate rates only under "
            "the declared simulation geometry/noise, not empirical RENGE power.", "",
            "| Scenario | Latent rho | Events / repetitions | Rate | 95% exact binomial interval |",
            "|---|---:|---:|---:|---:|"]
        for rate in rates:
            lines.append(f"| {rate['metric']} | {rate['rho']:g} | {rate['events']}/{rate['repetitions']} | "
                f"{rate['rate']:.4f} | [{rate['ci_low']:.4f}, {rate['ci_high']:.4f}] |")
        lines += ["", "Four algorithmic seeds are averaged as predictions, not treated as biological repeats. "
            "Intervals are averaged within TF; both primary comparisons must pass CI, Wilcoxon "
            "and positive-interval checks. The middle snapshot is shared by adjacent intervals, "
            "but cells at distinct times are independently sampled.", "",
            "Fixed source k-means centers are unused by the primary arms. Only identical per-arm "
            "feature matrices reuse a ridge fit; tied U-swap choices are generated separately for every seed.", "",
            "All simulation seeds, fold audits, per-repetition comparisons and provenance hashes "
            "are stored alongside this report. No biological replication data or GPU were used.", ""]
        (run.directory / "RESULTS.md").write_text("\n".join(lines))
        run.json("status.json", dict(status="complete"))
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        run.json("status.json", dict(status="failed", error=f"{type(error).__name__}: {error}"))
        raise


if __name__ == "__main__":
    main()
