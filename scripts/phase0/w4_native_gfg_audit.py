"""W4-0: CPU-only re-audit of existing native GFG target-domain fits.

No model fit or inference is performed. Saved velocities are checked against
the original summaries; analytical controls are reconstructed from the frozen
source moments, normalizers, and source-only U interventions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import anndata as ad
import numpy as np
from scipy import sparse

from veloroute.artifacts import Run, load_config, save_csv, save_json, sha256
from w3_7_native_train import raw_variants, row_normalization, velocity_comparison


VARIANTS = {
    "V0_real": "Original source U and S.",
    "V3_local_U": (
        "Mutual-kNN source-S U exchange; RENGE within condition, MeRLin within "
        "Day0 sample, not clone; preserves recipient selected-gene U depth."
    ),
    "V4_global_U": (
        "Global bijective cyclic exchange among positive-U-depth cells; zero-U "
        "cells retained; preserves recipient selected-gene U depth."
    ),
    "V7_zero_U": "All U set to zero; S, source graph and normalizers unchanged.",
}


def original_run(dataset: str, seed: int) -> Path:
    gpu = seed if dataset == "renge" else 3
    return Path(f"outputs/w3_7_native_{dataset}_gpu{gpu}_20261014_v1") / f"seed{seed}"


def registered_inputs(config: dict) -> list[Path]:
    root = Path(config["artifact_root"])
    inputs = []
    for dataset in ("renge", "merlin"):
        inputs += [root / dataset / "input_v1" / name for name in ("ready.json", "train.h5ad")]
        for seed in config["training_seeds"]:
            inputs += [root / dataset / f"seed{seed}_v1" / name for name in
                       ("ready.json", "final.pth", "normalization.npz", "train_source_velocities.npz")]
            inputs += [original_run(dataset, seed) / name for name in
                       ("summary.json", "U_sensitivity.csv", "controls.json", "provenance.json")]
    return inputs


def audit_dataset(config: dict, dataset: str) -> tuple[list[dict], dict]:
    root = Path(config["artifact_root"]) / dataset
    source_ready = json.loads((root / "input_v1/ready.json").read_text())
    source_path = root / "input_v1/train.h5ad"
    if source_ready["dataset"] != dataset or source_ready["sha256"] != sha256(source_path):
        raise ValueError("Frozen source identity/hash changed")
    if source_ready.get("future_counts_read", True) or source_ready.get("excluded_count_rows_read", True):
        raise ValueError("Source artifact does not establish a source-only contract")
    source = ad.read_h5ad(source_path)
    u = sparse.csr_matrix(source.layers["raw_U"]).toarray().astype(float)
    s = sparse.csr_matrix(source.layers["raw_S"]).toarray().astype(float)
    normalized_u = sparse.csr_matrix(source.layers["unspliced"]).toarray()
    factors = row_normalization(u, normalized_u)
    operator = source.obsp["w3_moment_operator"]
    ms = np.asarray(source.layers["Ms"], dtype=np.float32)
    mu0 = np.asarray(source.layers["Mu"], dtype=np.float32)
    strata = (source.obs["condition"].astype(str).to_numpy() if dataset == "renge"
              else np.repeat("MeRLin_Day0", source.n_obs))
    variants, controls = raw_variants(u, s, source.obsm["X_pca"], strata,
        seed=config["control_seed"], neighbors=config["local_neighbors"])
    gamma = np.sum(mu0 * ms, axis=0) / np.maximum(np.sum(ms * ms, axis=0), 1e-12)
    analytical = {}
    for variant, counts in variants.items():
        mu = (operator @ sparse.csr_matrix(counts * factors[:, None])).toarray().astype(np.float32)
        if variant == "V0_real":
            np.testing.assert_allclose(mu, mu0, rtol=1e-5, atol=1e-5)
        analytical[variant] = mu - gamma * ms
    rows = []
    for seed in config["training_seeds"]:
        directory = root / f"seed{seed}_v1"
        ready = json.loads((directory / "ready.json").read_text())
        original = original_run(dataset, seed)
        summary = json.loads((original / "summary.json").read_text())
        if ready != summary or ready["status"] != "NATIVE_TRAINING_AND_U_AUDIT_COMPLETE":
            raise ValueError("Original training completion records disagree")
        if (ready["dataset"], ready["seed"], ready["cells"], ready["genes"]) != (
                dataset, seed, source.n_obs, source.n_vars):
            raise ValueError("Native training/source identity mismatch")
        if sha256(directory / "final.pth") != ready["checkpoint_sha256"]:
            raise ValueError("Native checkpoint changed after the original audit")
        original_controls = json.loads((original / "controls.json").read_text())
        for key in ("local_order", "global_order"):
            if controls[key] != original_controls[key]:
                raise ValueError("Reconstructed U intervention differs from the original audit")
        with np.load(directory / "train_source_velocities.npz", allow_pickle=False) as velocities:
            np.testing.assert_array_equal(velocities["source_ids"], source.obs_names.to_numpy(dtype=str))
            np.testing.assert_array_equal(velocities["genes"], source.var_names.to_numpy(dtype=str))
            np.testing.assert_allclose(velocities["gamma"], gamma, rtol=1e-6, atol=1e-7)
            for variant in config["variants"]:
                native = velocity_comparison(velocities["V0_real"], velocities[variant])
                reference = velocity_comparison(analytical["V0_real"], analytical[variant])
                registered = next(r for r in ready["sensitivity"] if r["variant"] == variant)
                for key, value in native.items():
                    np.testing.assert_allclose(value, registered[key], rtol=1e-8, atol=1e-10)
                np.testing.assert_allclose(reference["cos_median"], registered["analytic_cos_median"],
                                           rtol=1e-6, atol=1e-6)
                rows.append(dict(dataset=dataset, seed=seed, variant=variant,
                    **{f"native_{k}": v for k, v in native.items()},
                    **{f"analytic_{k}": v for k, v in reference.items()},
                    original_summary=str(original / "summary.json"),
                    saved_velocities=str(directory / "train_source_velocities.npz")))
    details = dict(cells=source.n_obs, genes=source.n_vars, source=str(source_path),
        source_sha256=source_ready["sha256"], local_control=controls["local"],
        global_moved_fraction=controls["global_moved_fraction"],
        source_only_contract_verified=True, checkpoints_verified=len(config["training_seeds"]),
        saved_velocities_reproduced_original_summaries=True,
        training_seeds=config["training_seeds"], original_epochs=10)
    return rows, details


def result_markdown(rows: list[dict], metadata: dict) -> str:
    lines = ["# W4-0: native GFG source-U sensitivity audit", "",
        "Completed native target-domain fits were re-audited on CPU; no model was retrained.", "",
        "Native columns show the range of per-seed cell-median cos/norm ratio and global relative L2. "
        "The analytical reference is identical across training seeds.", "",
        "| Dataset | U intervention | Native cosine | Native norm ratio | Native relative L2 | "
        "Analytic cosine | Analytic norm ratio | Analytic relative L2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in ("renge", "merlin"):
        for variant in VARIANTS:
            subset = [r for r in rows if r["dataset"] == dataset and r["variant"] == variant]
            ranges = []
            for metric in ("cos_median", "norm_ratio_median", "relative_l2"):
                values = [r[f"native_{metric}"] for r in subset]
                ranges.append(f"{min(values):.6f}–{max(values):.6f}")
            analytic = [f"{subset[0][f'analytic_{metric}']:.6f}" for metric in
                        ("cos_median", "norm_ratio_median", "relative_l2")]
            lines.append("| " + " | ".join([dataset, variant, *ranges, *analytic]) + " |")
    lines += ["", "## Interpretation", "",
        "Under these target-domain native training settings, the output direction changes little "
        "under U interventions. Norm and L2 changes are not zero. This is an estimator-sensitivity "
        "result, not evidence that U has no biological information or that downstream prediction cannot improve.",
        "", "The analytical reference is Mu − gamma·Ms with gamma fitted by full-source moment OLS "
        "and frozen across interventions; it is not W4's raw-count two-tail estimator. "
        "S moments, graph, and recipient normalizers are fixed. All valid-direction fractions are "
        "reported in metrics.csv. These are training-source audits, not held-out prediction results.",
        "", "## Provenance", "",
        f"Git commit: `{metadata['git_commit']}`; seeds: `{metadata['seed']}`; "
        f"runtime: {metadata['runtime_seconds']:.3f} seconds.",
        "", "Exact commands, all source paths and SHA256 hashes are in summary.json/provenance.json; "
        "per-seed values and original summary/velocity paths are in metrics.csv.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/w3_7_native_gfg_20261014.json")
    parser.add_argument("--output", default="outputs/w4_native_gfg_audit_v1")
    args = parser.parse_args()
    started = time.monotonic()
    config = load_config(args.config)
    if config["variants"] != list(VARIANTS) or config["training_seeds"] != [0, 1, 2]:
        raise ValueError("This closeout audits exactly the registered four variants and three seeds")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    scripts = [Path(__file__), Path(__file__).with_name("w3_7_native_train.py"),
               Path(__file__).with_name("w3_validation.py")]
    inputs = [Path(args.config), *scripts, *registered_inputs(config)]
    with Run(args.output, stage="W4_0_existing_native_GFG_CPU_reaudit", kind="development",
             config=dict(config, audit_git_commit=git_commit, command=sys.argv),
             inputs=inputs, seed=config["training_seeds"]) as run:
        rows, datasets = [], {}
        for dataset in ("renge", "merlin"):
            values, audit = audit_dataset(config, dataset)
            rows.extend(values)
            datasets[dataset] = audit
        save_csv(run.directory / "metrics.csv", rows)
        metadata = dict(git_commit=git_commit, seed=config["training_seeds"],
            control_seed=config["control_seed"], command=[sys.executable, *sys.argv],
            data_hashes={r["path"]: r["sha256"] for r in run.provenance["inputs"]},
            runtime_seconds=time.monotonic() - started)
        summary = dict(status="W4_0_NATIVE_GFG_CPU_REAUDIT_COMPLETE", **metadata,
            datasets=datasets, variant_definitions=VARIANTS, metrics=rows,
            analytical_reference="Frozen full-training-source moment OLS gamma, Mu - gamma*Ms; not W4 two-tail gamma",
            new_training_performed=False, gpu_used=False, future_counts_read=False,
            downstream_prediction_evaluated=False)
        save_json(run.directory / "summary.json", summary)
        (run.directory / "RESULTS.md").write_text(result_markdown(rows, metadata))
        run.provenance.update(metadata)
    print(json.dumps(dict(status=summary["status"], output=str(Path(args.output).resolve()),
        rows=len(rows), datasets=datasets, runtime_seconds=metadata["runtime_seconds"])))


if __name__ == "__main__":
    main()
