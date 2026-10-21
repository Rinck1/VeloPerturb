"""Fixed-recipe RENGE L2: native CellFlow with source-population covariates.

No held-out destination is accepted by the prediction function. Cross-condition
shuffle uses a derangement of TRAINING source tokens, then a deterministic wrong
training-source reference at the single held-out condition. This is deliberately
not described as a symmetric held-out condition permutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from w4_artifacts import Artifacts, sha256
from w15a_cellflow_adapter import (TokenNormalizer, add_cellflow_paths, array_hash,
    condition_inputs, predict_cellflow_tau)
from w15a_cellflow_train import create_solver, fit_groups


VARIANTS = ("cf_static", "cf_Ssum", "cf_directU", "cf_kin",
            "cf_kin_shuf_perm0", "cf_kin_shuf_perm1", "cf_kin_shuf_perm2")
SHUFFLE_SCOPE = "training_source_derangement_plus_heldout_training_reference_replacement"


def load_fold(pack_directory, fold_name):
    # Lazy import permits isolated CPU fixtures while the shared pack is prepared.
    from w15c_renge_data import load_fold as read_fold
    return read_fold(pack_directory, fold_name)


def variant_spec(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unregistered RENGE L2 variant: {variant}")
    if variant.startswith("cf_kin_shuf_perm"):
        return dict(variant=variant, arm="cf_kin_shuf", shuffle_seed=int(variant[-1]))
    return dict(variant=variant, arm=variant, shuffle_seed=None)


def training_derangement(groups, seed):
    """Derange unique source CONDITIONS; no held-out input is an argument."""
    labels = [str(g["label"]) for g in groups]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Training token derangement needs at least two unique conditions")
    times = np.asarray([g["time_hours"] for g in groups], dtype=float)
    if not np.isfinite(times).all() or not np.all(times == times[0]):
        raise ValueError("Training token derangement requires a fixed biological source time")
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        mapping = rng.permutation(len(groups))
        if np.all(mapping != np.arange(len(groups))):
            return mapping
    raise ValueError("Could not construct a training-condition derangement")


def token_normalizer(values):
    kinds = ("kin", "Ssum", "directU", "cell_kin", "cell_directU")
    mean = {k: np.asarray(values["mean_" + k]) for k in kinds}
    scale = {k: np.asarray(values["scale_" + k]) for k in kinds}
    for kind in kinds:
        if (mean[kind].ndim != 1 or scale[kind].shape != mean[kind].shape
                or not np.isfinite(mean[kind]).all()
                or not np.isfinite(scale[kind]).all() or (scale[kind] <= 0).any()):
            raise ValueError("Invalid training-only token normalizer")
    return TokenNormalizer(mean, scale, ())


def source_condition(group, normalizer, arm, *, replacement=None):
    """Construct one source-side condition; shuffle never reads its real token."""
    z = np.asarray(group["z"])
    if z.ndim != 2 or not len(z) or not np.isfinite(z).all():
        raise ValueError("Source states must be a finite nonempty matrix")
    embedding = np.asarray(group["embedding"])
    if embedding.ndim != 1 or not len(embedding) or not np.isfinite(embedding).all():
        raise ValueError("Real ESM perturbation embedding must be a finite vector")
    time = float(group["time_hours"])
    population = {}
    if arm in ("cf_kin", "cf_Ssum", "cf_directU"):
        kind = {"cf_kin": "kin", "cf_Ssum": "Ssum", "cf_directU": "directU"}[arm]
        value = np.asarray(group["population_" + kind])
        if value.shape != (z.shape[1],):
            raise ValueError("Population token dimension differs from source state")
        population[kind] = value
    return condition_inputs(dict(z=z, population=population), normalizer, arm,
        time_hours=time, condition_delta_hours=24., perturbation_embedding=embedding,
        shuffled_population_token=replacement)


def prepare_conditions(groups, source, norm, variant):
    """Build training batches and one source-only held-out conditioning object."""
    spec = variant_spec(variant)
    if not groups:
        raise ValueError("RENGE fold has no training source groups")
    labels = [str(g["label"]) for g in groups]
    if len(set(labels)) != len(labels) or str(source["label"]) in labels:
        raise ValueError("Training conditions must be unique and exclude held-out TF")
    dimension = np.asarray(source["z"]).shape[1]
    embedding_dimension = np.asarray(source["embedding"]).shape
    for group in groups:
        if (np.asarray(group["z"]).ndim != 2 or np.asarray(group["z"]).shape[1] != dimension
                or np.asarray(group["embedding"]).shape != embedding_dimension):
            raise ValueError("Fold source/ESM dimensions are inconsistent")
        target = np.asarray(group["target"])
        if target.ndim != 2 or not len(target) or target.shape[1] != dimension or not np.isfinite(target).all():
            raise ValueError("Invalid training-group target")
        if float(group["time_hours"]) != float(source["time_hours"]):
            raise ValueError("A RENGE fold must have a single biological source time")
    normalizer = token_normalizer(norm)
    replacements = [None] * len(groups)
    heldout_replacement = None
    audit = dict(shuffle_seed=spec["shuffle_seed"], symmetric_heldout_permutation=False)
    if spec["shuffle_seed"] is not None:
        mapping = training_derangement(groups, spec["shuffle_seed"])
        bank = np.stack([normalizer.apply("kin", g["population_kin"]) for g in groups])
        replacements = list(bank[mapping])
        heldout_index = int(np.random.default_rng(spec["shuffle_seed"]).integers(len(groups)))
        heldout_replacement = bank[heldout_index]
        audit.update(scope=SHUFFLE_SCOPE, training_labels=labels,
            training_donor_labels=[labels[int(i)] for i in mapping],
            heldout_training_reference_label=labels[heldout_index],
            training_token_bank_sha256=array_hash(bank),
            heldout_replacement_sha256=array_hash(heldout_replacement),
            heldout_source_token_used_in_training=False,
            heldout_source_token_used_for_replacement=False)
    prepared = [dict(z=np.asarray(g["z"], dtype="float32"),
        target=np.asarray(g["target"], dtype="float32"),
        condition=source_condition(g, normalizer, spec["arm"], replacement=replacement))
        for g, replacement in zip(groups, replacements)]
    condition = source_condition(source, normalizer, spec["arm"], replacement=heldout_replacement)
    return prepared, condition, audit


def validate_configuration(config, *, engineering_fixture=False):
    if engineering_fixture:
        if int(config["steps"]) > 2 or int(config["batch_size"]) > 8:
            raise ValueError("CPU engineering fixture must use at most two tiny updates")
    elif (config["width"], config["depth"], config["steps"],
          config["learning_rate"], config["batch_size"]) != (128, 2, 5000, .0003, 128):
        raise ValueError("RENGE L2 inherits the fixed 128x2 / 5000-step CellFlow recipe")
    if config.get("mean_anchor", False) or config.get("gfg_jvp", False):
        raise ValueError("RENGE CellFlow uses neither a mean anchor nor GFG/JVP")


def score_rows(source):
    indices = np.asarray(source["score_indices"])
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
            or indices.min() < 0 or indices.max() >= len(source["z"])
            or len(set(indices.tolist())) != len(indices)):
        raise ValueError("Frozen L1 comparison source indices are invalid")
    ids = np.asarray(source["cell_ids"], dtype=str)
    if len(ids) != len(source["z"]) or len(set(ids.tolist())) != len(ids):
        raise ValueError("Source IDs are invalid")
    return indices, ids[indices]


def train_predict(pack, fold, config_path, output, variant, seed, *, engineering_fixture=False):
    """Train native OTFM and freeze one held-out source-only 24h forecast."""
    spec = variant_spec(variant)
    config_path, pack = Path(config_path), Path(pack)
    config = json.loads(config_path.read_text())
    validate_configuration(config, engineering_fixture=engineering_fixture)
    fm, training_groups, source, norm = load_fold(pack, fold)
    if int(fm["target_day"]) - int(fm["source_day"]) != 1:
        raise ValueError("This fixed RENGE transfer predicts adjacent 24-hour intervals only")
    expected_time = 24. * (int(fm["source_day"]) - 2)
    if float(source["time_hours"]) != expected_time or str(source["label"]) != str(fm["TF"]):
        raise ValueError("Held-out condition/biological-time metadata mismatch")
    groups, condition, shuffle = prepare_conditions(training_groups, source, norm, variant)
    indices, source_ids = score_rows(source)
    inputs = [config_path, Path(__file__), Path(__file__).with_name("w15a_cellflow_train.py"),
              Path(__file__).with_name("w15a_cellflow_adapter.py")]
    for candidate in (Path(__file__).with_name("w15c_renge_data.py"), pack / "manifest.json",
                      pack / "study_manifest.json", pack / "folds" / fold / "manifest.json"):
        if candidate.is_file():
            inputs.append(candidate)
    run = Artifacts(output, seed=seed, inputs=inputs)
    run.json("study_manifest.json", dict(fold=fold, variant=variant, seed=seed,
        pack_directory=str(pack.resolve()), fold_metadata=fm, config=config,
        engineering_fixture=engineering_fixture, training_conditions=[str(g["label"]) for g in training_groups],
        source_ids_sha256=array_hash(source_ids), shuffle=shuffle,
        time_hours=expected_time, delta_hours=24., tau_end=1., mean_anchor=False,
        OT_cost="expression_PCA_S_only", heldout_future_used=False))
    add_cellflow_paths()
    import jax
    from flax import serialization
    dimension = source["z"].shape[1]
    solver = create_solver(config, {k: v[:1] for k, v in groups[0]["condition"].items()}, dimension, seed)
    history = fit_groups(solver, groups, config, seed,
        callback=lambda row: run.json("status.json", dict(status="training", fold=fold,
            variant=variant, seed=seed, **row)))
    if not history or int(history[-1]["step"]) != int(config["steps"]):
        raise ValueError("CellFlow training did not finish the fixed update budget")
    model_path = run.directory / "model.msgpack"
    model_path.write_bytes(serialization.to_bytes(solver.vf_state_inference))
    # Exercise the same serialized checkpoint that later consumers will use.
    solver.vf_state_inference = serialization.from_bytes(solver.vf_state_inference, model_path.read_bytes())
    run.json("training_history.json", dict(history=history, steps=int(config["steps"])))
    run.json("status.json", dict(status="source_only_prediction", fold=fold, variant=variant, seed=seed))
    prediction = predict_cellflow_tau(solver, np.asarray(source["z"])[indices],
        {k: v[indices] for k, v in condition.items()}, tau_end=1., per_particle=True)
    if prediction.shape != (len(indices), dimension) or not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite or invalid native CellFlow ODE forecast")
    path = run.directory / "predictions.npz"
    np.savez_compressed(path, candidates=prediction[:, None, :].astype("float32"),
        q=np.ones((len(indices), 1), dtype="float32"), source_ids=source_ids)
    result = dict(status="RENGE_L2_PREDICTIONS_FROZEN", fold=fold, variant=variant,
        arm=spec["arm"], seed=seed, shuffle_seed=spec["shuffle_seed"], shuffle=shuffle,
        path=str(path.resolve()), sha256=sha256(path), model_sha256=sha256(model_path),
        parameters=int(sum(p.size for p in jax.tree.leaves(solver.vf_state.params))),
        steps=int(config["steps"]), dimensions=dimension, source_cells=len(indices),
        time_hours=expected_time, delta_hours=24., tau_end=1., mean_anchor=False,
        solver="Tsit5", rtol=1e-5, atol=1e-5, training_groups=len(groups),
        heldout_future_used=False, engineering_fixture=engineering_fixture,
        config_sha256=sha256(config_path), backend=str(jax.devices()))
    run.json("prediction_manifest.json", result)
    run.json("summary.json", result)
    run.json("status.json", dict(status="complete", fold=fold, variant=variant, seed=seed))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("pack", "fold", "configuration", "output", "variant"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(train_predict(args.pack, args.fold, args.configuration, args.output,
        args.variant, args.seed), indent=2))
