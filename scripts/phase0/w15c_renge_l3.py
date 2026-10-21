"""RENGE L3: frozen official S-only OT backbone plus source kinetic routing.

The twelve architecture/training choices are inherited from W15A, not tuned on
RENGE. Perturbation ESM embeddings enter only the state-field condition stream.
The learned kinetic encoder/router is conditioned only on the source S state
and local kinetic (or direct-U) representation. Each expert is integrated as an
independent trajectory with its source routing weight held fixed.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from w4_artifacts import Artifacts, sha256
from w15a_cellflow_adapter import add_cellflow_paths, biological_fourier

add_cellflow_paths()

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training.train_state import TrainState
from cellflow.solvers.utils import ema_update
from cellflow.utils import match_linear

from w15a_cellflow_moe import MoEConfig
from w15a_cellflow_moe_official import make_official_ot_train_step, official_moe_from_native
from w15a_cellflow_moe_real_predict import make_candidate_integrator
from w15a_cellflow_moe_real_train import require_complete_synthetic_gate
from w15a_cellflow_train import create_solver


VARIANTS = ("moe_static", "moe_kin", "moe_directU", "moe_kin_shuf_perm0",
            "moe_kin_shuf_perm1", "moe_kin_shuf_perm2")
SEEDS = (0, 1, 2)
DEFAULT_SYNTHETIC_GATE = "/data/yuchang/veloroute_w15a/l3_population_frozen_plan_v1"


def load_fold(pack_directory, fold_name):
    """Root-owned source-only test pack API; held-out targets are never requested."""
    from w15c_renge_data import load_fold as read
    return read(pack_directory, fold_name)


def variant_spec(variant):
    if variant not in VARIANTS:
        raise ValueError("Unregistered RENGE L3 variant")
    shuffled = variant.startswith("moe_kin_shuf_perm")
    return dict(arm="moe_kin_shuf" if shuffled else variant,
                shuffle_seed=int(variant[-1]) if shuffled else None)


def source_state_conditions(count, time_hours, embedding, delta_hours=24.):
    embedding = np.asarray(embedding, dtype="float32")
    if embedding.ndim == 2 and embedding.shape[0] == 1:
        embedding = embedding[0]
    if embedding.ndim != 1 or not embedding.size or not np.isfinite(embedding).all():
        raise ValueError("Perturbation embedding must be a finite nonempty vector")
    return dict(perturbation=jnp.broadcast_to(jnp.asarray(embedding), (count, 1, len(embedding))),
        bio_time=jnp.broadcast_to(jnp.asarray(biological_fourier(time_hours), dtype=jnp.float32), (count, 1, 8)),
        bio_delta=jnp.broadcast_to(jnp.asarray(biological_fourier(delta_hours), dtype=jnp.float32), (count, 1, 8)))


def kinetic_input(pack, normalizer, variant):
    """Permutation is complete within this source cohort and training-seed free."""
    spec = variant_spec(variant)
    if spec["arm"] == "moe_static":
        return np.zeros_like(pack["cell_kin"], dtype="float32")
    kind = "cell_directU" if spec["arm"] == "moe_directU" else "cell_kin"
    mean = np.asarray(normalizer["mean_" + kind])
    scale = np.asarray(normalizer["scale_" + kind])
    if mean.shape != (pack["z"].shape[1],) or scale.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid training-source kinetic normalizer")
    value = (np.asarray(pack[kind]) - mean) / scale
    if spec["arm"] == "moe_kin_shuf":
        identity = str(pack.get("source_key", pack["label"])) + "|" + str(float(pack["time_hours"]))
        source_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "little")
        order = np.random.default_rng(np.random.SeedSequence([spec["shuffle_seed"], source_seed])).permutation(len(value))
        value = value[order]
    value = np.asarray(value, dtype="float32")
    if value.shape != np.shape(pack["z"]) or not np.isfinite(value).all():
        raise ValueError("Invalid local kinetic feature shape or values")
    return value


def create_model(configuration, dimensions, seed, embedding):
    condition = source_state_conditions(1, 0., embedding)
    native_condition = dict(condition, population_token=jnp.zeros((1, 1, dimensions)),
                            cell_token=jnp.zeros((1, 1, dimensions)))
    native = create_solver(configuration, native_condition, dimensions, seed).vf
    return official_moe_from_native(native, config=MoEConfig(dimensions=dimensions))


def fit_group_arrays(groups, configuration, seed, *, callback=None):
    """Low-level array primitive, also exercised by explicitly synthetic CPU tests."""
    first = groups[0]
    model = create_model(configuration, first["z"].shape[1], seed, first["embedding"])
    initial = model.init(jax.random.PRNGKey(seed), jnp.asarray(first["z"][:1]),
        jnp.asarray(first["kinetic_input"][:1]), jnp.asarray(first["z"][:1]), .5,
        source_state_conditions(1, first["time_hours"], first["embedding"]))["params"]
    state = TrainState.create(apply_fn=model.apply, params=initial,
                             tx=optax.adam(configuration["learning_rate"]))
    inference_params = initial
    match_fn = functools.partial(match_linear, epsilon=configuration["ot_epsilon"], scale_cost="mean",
                                  max_iterations=configuration["ot_max_iterations"])
    step = make_official_ot_train_step(model, match_fn)
    rng = np.random.default_rng(seed)
    history, started = [], time.monotonic()
    for update in range(1, configuration["steps"] + 1):
        group = groups[int(rng.integers(len(groups)))]
        count = configuration["batch_size"]
        si = rng.choice(len(group["z"]), count, replace=len(group["z"]) < count)
        ti = rng.choice(len(group["target"]), count, replace=len(group["target"]) < count)
        state, metrics = step(state, jax.random.fold_in(jax.random.PRNGKey(seed), update),
            jnp.asarray(group["z"][si]), jnp.asarray(group["target"][ti]),
            jnp.asarray(group["kinetic_input"][si]),
            source_state_conditions(count, group["time_hours"], group["embedding"]))
        inference_params = (state.params if configuration.get("ema", 1.) == 1. else
            ema_update(inference_params, state.params, configuration["ema"]))
        if update == 1 or update % configuration.get("log_interval", 250) == 0 or update == configuration["steps"]:
            values = {key: float(value) for key, value in metrics.items()
                      if key not in ("source_indices", "target_indices")}
            if not np.isfinite(list(values.values())).all():
                raise FloatingPointError("Nonfinite RENGE L3 task update")
            row = dict(step=update, training_elapsed_seconds=time.monotonic()-started, **values)
            history.append(row)
            if callback is not None:
                callback(row)
    return model, inference_params, history, initial


def predict_arrays(model, params, source, normalizer, variant):
    """Compute q on the full source, then integrate canonical score cells only."""
    if "target" in source:
        raise ValueError("Held-out source pack must not contain a future target")
    z = np.asarray(source["z"], dtype="float32")
    indices = np.asarray(source["score_indices"])
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or
            len(indices) == 0 or len(set(indices.tolist())) != len(indices) or
            indices.min() < 0 or indices.max() >= len(z)):
        raise ValueError("Invalid canonical score indices")
    ids = np.asarray(source["cell_ids"], dtype=str)
    if len(ids) != len(z) or len(set(ids.tolist())) != len(ids):
        raise ValueError("Source identities are missing or duplicated")
    features = kinetic_input(source, normalizer, variant)
    q = np.asarray(model.apply(dict(params=params), jnp.asarray(z), jnp.asarray(features), method=model.route))
    if q.shape != (len(z), model.config.experts) or not np.isfinite(q).all() or (q < 0).any():
        raise ValueError("Nonfinite or invalid source routing weights")
    np.testing.assert_allclose(q.sum(1), 1., rtol=1e-5, atol=1e-6)
    integrate = make_candidate_integrator(model)
    condition = source_state_conditions(len(indices), source["time_hours"], source["embedding"])
    candidates = integrate(params, jnp.asarray(z[indices]), condition, 1.)
    return dict(candidates=np.asarray(candidates, dtype="float32"), q=q[indices],
        q_cohort=np.broadcast_to(q.mean(0), (len(indices), q.shape[1])).copy(),
        q_uniform=np.full((len(indices), q.shape[1]), 1./q.shape[1], dtype="float32"),
        source_ids=ids[indices], source_indices=indices,
        full_source_population_q=q.mean(0), full_source_cells=np.asarray(len(z)))


def _registered_configuration(path, gate):
    configuration = json.loads(Path(path).read_text())
    inherited_path = Path(configuration.get("inherited_config_path", path))
    inherited_hash = sha256(inherited_path)
    if (configuration.get("inherited_config_sha256", inherited_hash) != inherited_hash or
            gate.get("l2_configuration_sha256") != inherited_hash):
        raise ValueError("RENGE L3 must inherit the exact synthetic-tested W15A recipe")
    inherited = json.loads(inherited_path.read_text())
    recipe = ("width", "depth", "condition_dim", "time_freqs", "time_width", "time_depth",
              "pooling", "learning_rate", "ot_epsilon", "ot_max_iterations", "ema",
              "batch_size", "steps", "log_interval")
    if any(configuration.get(key) != inherited.get(key) for key in recipe):
        raise ValueError("RENGE L3 cannot change the inherited architecture/training recipe")
    fixed = dict(width=128, depth=2, steps=5000, learning_rate=.0003, batch_size=128)
    if any(configuration.get(key) != value for key, value in fixed.items()):
        raise ValueError("RENGE L3 training architecture and budget are frozen")
    return configuration


def train_predict(pack_directory, fold_name, config_path, output, *, variant, seed,
                  synthetic_gate=DEFAULT_SYNTHETIC_GATE):
    """Fit a single held-out TF/day fold and freeze source-only predictions."""
    spec = variant_spec(variant)
    if seed not in SEEDS:
        raise ValueError("Registered RENGE L3 training seeds are 0/1/2")
    gate = require_complete_synthetic_gate(synthetic_gate)
    configuration = _registered_configuration(config_path, gate)
    fm, groups, source, normalizer = load_fold(pack_directory, fold_name)
    if "target" in source:
        raise ValueError("Held-out source must be separated from its future target")
    if not groups or any(np.shape(group["z"])[1:] != (50,) for group in groups) or np.shape(source["z"])[1:] != (50,):
        raise ValueError("RENGE L3 requires training groups and frozen PCA50 state")
    if float(source["time_hours"]) != 24.*(int(fm["source_day"])-2) or int(fm["target_day"]) != int(fm["source_day"])+1:
        raise ValueError("RENGE biological time must be source day minus day2, with a 24h adjacent target")
    embedding_dim = np.asarray(source["embedding"]).size
    for group in groups:
        if np.asarray(group["embedding"]).size != embedding_dim:
            raise ValueError("All perturbations must share one ESM embedding dimension")
        if np.shape(group["target"])[1:] != (50,) or not len(group["target"]):
            raise ValueError("Missing training-only target distribution")
    prepared = [dict(group, kinetic_input=kinetic_input(group, normalizer, variant)) for group in groups]
    output = Path(output)
    inputs = [Path(config_path), Path(synthetic_gate)/"summary.json", Path(__file__),
              Path(__file__).with_name("w15a_cellflow_moe.py"),
              Path(__file__).with_name("w15a_cellflow_moe_official.py")]
    inherited_path = Path(configuration.get("inherited_config_path", config_path))
    if inherited_path.resolve() != Path(config_path).resolve():
        inputs.append(inherited_path)
    for name in ("manifest.json", "study_manifest.json"):
        if (Path(pack_directory)/name).is_file():
            inputs.append(Path(pack_directory)/name)
    run = Artifacts(output, seed=seed, inputs=inputs)
    model, params, history, initial = fit_group_arrays(prepared, configuration, seed,
        callback=lambda row: run.json("status.json", dict(status="training", fold=fold_name,
                                                          variant=variant, seed=seed, **row)))
    (output/"model.msgpack").write_bytes(serialization.to_bytes(params))
    run.json("training_history.json", dict(history=history))
    # Restore the actual saved artifact before any forecast, exercising deployment.
    restored = serialization.from_bytes(initial, (output/"model.msgpack").read_bytes())
    predictions = predict_arrays(model, restored, source, normalizer, variant)
    np.savez_compressed(output/"predictions.npz", **predictions)
    manifest = dict(status="RENGE_L3_SOURCE_PREDICTIONS_FROZEN", fold=fold_name, variant=variant,
        seed=seed, TF=fm["TF"], source_day=fm["source_day"], target_day=fm["target_day"],
        path=str((output/"predictions.npz").resolve()), sha256=sha256(output/"predictions.npz"),
        prediction_sha256=sha256(output/"predictions.npz"), model_sha256=sha256(output/"model.msgpack"),
        heldout_future_used=False, heldout_targets_read=False,
        full_source_q_before_score_selection=True, source_routing_fixed=True,
        independent_expert_ODEs=True, experts=model.config.experts,
        cohort_q_and_uniform_q_reuse_identical_candidates=True, **spec)
    run.json("prediction_manifest.json", manifest)
    result = dict(status="RENGE_L3_TRAIN_PREDICT_COMPLETE", fold=fold_name, variant=variant,
        seed=seed, config=configuration, synthetic_gate=gate, fold_metadata=fm,
        parameters=sum(value.size for value in jax.tree_util.tree_leaves(params)),
        embedding_dimensions=embedding_dim, all_training_steps=history[-1]["step"],
        state_dimension=50, experts=model.config.experts, solver="Tsit5", tau_end=1.,
        biological_delta_hours=24., mean_anchor=False, JVP=False, heldout_future_used=False,
        heldout_targets_read=False, prediction_manifest=manifest,
        initial_parameters_sha256=hashlib.sha256(serialization.to_bytes(initial)).hexdigest(),
        backend=[str(device) for device in jax.devices()])
    run.json("summary.json", result)
    run.json("status.json", dict(status="complete"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", required=True); parser.add_argument("--fold", required=True)
    parser.add_argument("--configuration", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--synthetic-gate", default=DEFAULT_SYNTHETIC_GATE)
    args = parser.parse_args()
    print(json.dumps(train_predict(args.pack, args.fold, args.configuration, args.output,
        variant=args.variant, seed=args.seed, synthetic_gate=args.synthetic_gate), indent=2))
