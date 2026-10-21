"""Gated L3 real-data training; frozen core and active synthetic are unchanged.

Real training is impossible until the complete registered 120-fit synthetic grid
has finished and passed its predeclared engineering gate. Low-level array fitting
is also used by explicit CPU synthetic fixtures, never as a real-data gate bypass.
"""
from __future__ import annotations

import argparse
import csv
import functools
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

from w15a_cellflow_moe import MoEConfig, require_l2_freeze
from w15a_cellflow_moe_official import make_official_ot_train_step, official_moe_from_native
from w15a_cellflow_train import create_solver, load_npz


VARIANTS = ("moe_static", "moe_kin", "moe_directU", "moe_kin_shuf_perm0",
            "moe_kin_shuf_perm1", "moe_kin_shuf_perm2")
SEEDS = (0, 1, 2)
HERE = Path(__file__).resolve().parent


def registered_jobs():
    return [dict(variant=variant, seed=seed, **variant_spec(variant))
            for variant in VARIANTS for seed in SEEDS]


def variant_spec(variant):
    if variant not in VARIANTS:
        raise ValueError("Unregistered L3 variant")
    shuffled = variant.startswith("moe_kin_shuf_perm")
    return dict(arm="moe_kin_shuf" if shuffled else variant,
                shuffle_seed=int(variant[-1]) if shuffled else None)


def source_state_conditions(count, time_hours, delta_hours=24.):
    return dict(perturbation=jnp.ones((count, 1, 1), dtype=jnp.float32),
        bio_time=jnp.broadcast_to(jnp.asarray(biological_fourier(time_hours), dtype=jnp.float32), (count, 1, 8)),
        bio_delta=jnp.broadcast_to(jnp.asarray(biological_fourier(delta_hours), dtype=jnp.float32), (count, 1, 8)))


def require_complete_synthetic_gate(directory):
    """Never open partial scores: a completed full-grid summary is the first gate."""
    directory = Path(directory)
    path = directory / "summary.json"
    if not path.is_file():
        raise ValueError("Complete 120-fit L3 synthetic summary is required before real-data training")
    summary = json.loads(path.read_text())
    if (summary.get("status") != "L3_POPULATION_SYNTHETIC_GRID_COMPLETE" or
            summary.get("mode") != "frozen_L2_full_20_positive_20_null" or
            summary.get("completed_fits") != 120 or
            summary.get("full_positive_null_grid_completed") is not True or
            summary.get("population_composition_learning_gate_passed") is not True):
        raise ValueError("Complete registered L3 synthetic grid has not passed its engineering gate")
    from w15a_cellflow_moe_synthetic import verify_plan, _summarize
    plan = verify_plan(directory)
    if (plan["mode"] != "frozen_L2_full_20_positive_20_null" or
            plan["repetition_seeds"] != list(range(20)) or
            plan["scenario_order"] != ["positive", "null"] or
            plan["arms"] != ["moe_static", "moe_kin", "moe_kin_shuf"]):
        raise ValueError("L3 synthetic design is not the complete registered 20+20 grid")
    with (directory / "all_repetitions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {(scenario, repetition, arm) for scenario in ("positive", "null")
                for repetition in range(20) for arm in plan["arms"]}
    identities = [(row["scenario"], int(row["repetition"]), row["arm"]) for row in rows]
    if len(identities) != 120 or set(identities) != expected:
        raise ValueError("Synthetic fit grid incomplete or duplicated")
    for row in rows:
        for key in ("repetition",):
            row[key] = int(row[key])
        for key in ("mean_energy", "mean_mode_proportion_TV", "mean_effective_experts"):
            row[key] = float(row[key])
        fit = directory / "fits" / row["scenario"] / f"repetition{row['repetition']:02d}" / row["arm"]
        fit_summary = json.loads((fit / "summary.json").read_text())
        history = json.loads((fit / "training_history.json").read_text())
        if (not history or history[-1]["step"] != plan["model_configuration"]["steps"] or
                sha256(fit / "model.msgpack") != row["model_sha256"] or
                fit_summary["model_sha256"] != row["model_sha256"] or
                float(fit_summary["mean_energy"]) != row["mean_energy"]):
            raise ValueError("Synthetic completed-fit artifact differs from registered budget/checkpoint")
    checked = _summarize(plan, rows)
    if checked["population_composition_learning_gate_passed"] is not True:
        raise ValueError("Recomputed frozen synthetic gate did not pass")
    return dict(directory=str(directory.resolve()), summary_sha256=sha256(path),
        plan_sha256=sha256(directory / "plan.json"), repetitions_sha256=sha256(directory / "all_repetitions.csv"),
        completed_fits=120, engineering_gate_passed=True,
        l2_configuration_sha256=plan["l2_freeze"]["configuration_sha256"])


def sidecar_api():
    """Root-owned source-only feature API; no raw counts are read in this module."""
    from w15a_l3_source_sidecar import load_sidecar, load_normalizer
    return load_sidecar, load_normalizer


def sidecar_metadata(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = manifest["training_groups"] + manifest["development_sources"]
    if len({row["source_key"] for row in rows}) != len(rows):
        raise ValueError("L3 sidecar source keys must be unique across train/development")
    return manifest


def read_feature_cohort(directory, source_key):
    manifest = sidecar_metadata(directory)
    matches = [row for row in manifest["training_groups"] + manifest["development_sources"]
               if row["source_key"] == source_key]
    if len(matches) != 1:
        raise ValueError("L3 source cohort is missing or ambiguous")
    row = matches[0]
    if sha256(row["path"]) != row["sha256"]:
        raise ValueError("L3 source-only feature sidecar changed")
    load_sidecar, _ = sidecar_api()
    pack = load_sidecar(directory, source_key)
    required = ("cell_ids", "z", "cell_kin", "cell_directU", "cell_kin_Umean", "cell_kin_Uzero")
    if any(key not in pack for key in required) or "target" in pack:
        raise ValueError("Missing local feature sidecar fields or forbidden target field")
    n, dimensions = pack["z"].shape
    if len(pack["cell_ids"]) != n or len(set(pack["cell_ids"].tolist())) != n:
        raise ValueError("Sidecar source identity mismatch")
    for key in required[2:]:
        if np.shape(pack[key]) != (n, dimensions) or not np.isfinite(pack[key]).all():
            raise ValueError("Local feature shape or finite-value mismatch")
    return pack, row


def read_feature_normalizer(directory):
    _, load_normalizer = sidecar_api()
    result = load_normalizer(directory)
    for kind in ("cell_kin", "cell_directU"):
        if not np.isfinite(result["mean_"+kind]).all() or not np.isfinite(result["scale_"+kind]).all():
            raise ValueError("Nonfinite feature normalizer")
        if (np.asarray(result["scale_"+kind]) <= 0).any():
            raise ValueError("Local feature scales must be positive")
    return result


def velocity_input(pack, normalizer, variant, source_key, *, intervention=None):
    """Full-cohort inputs are constructed BEFORE cell sampling and source OT."""
    spec = variant_spec(variant)
    if spec["arm"] == "moe_static":
        if intervention is not None:
            raise ValueError("U interventions are only defined for frozen moe_kin inference")
        return np.zeros_like(pack["cell_kin"], dtype="float32")
    if intervention is not None and spec["arm"] != "moe_kin":
        raise ValueError("U interventions are only defined for frozen moe_kin inference")
    if spec["arm"] == "moe_directU":
        key, kind = "cell_directU", "cell_directU"
    else:
        key, kind = {None: "cell_kin", "Umean": "cell_kin_Umean", "Uzero": "cell_kin_Uzero"}[intervention], "cell_kin"
    value = (np.asarray(pack[key]) - normalizer["mean_"+kind]) / normalizer["scale_"+kind]
    if spec["arm"] == "moe_kin_shuf":
        # Stable source-key hash, independent from Python hash randomization and training seed.
        import hashlib
        source_seed = int.from_bytes(hashlib.sha256(source_key.encode()).digest()[:4], "little")
        order = np.random.default_rng(np.random.SeedSequence([spec["shuffle_seed"], source_seed])).permutation(len(value))
        value = value[order]
    return np.asarray(value, dtype="float32")


def load_training_groups(sidecar, variant):
    manifest = sidecar_metadata(sidecar)
    normalizer = read_feature_normalizer(sidecar)
    groups = []
    for row in manifest["training_groups"]:
        pack, actual = read_feature_cohort(sidecar, row["source_key"])
        if actual != row or sha256(row["parent_path"]) != row["parent_sha256"]:
            raise ValueError("Parent source/target training snapshot changed")
        parent = load_npz(row["parent_path"])
        if not np.array_equal(parent["cell_ids"], pack["cell_ids"]):
            raise ValueError("Sidecar/parent training source IDs differ")
        np.testing.assert_allclose(parent["z"], pack["z"], rtol=1e-6, atol=2e-5)
        groups.append(dict(z=np.asarray(pack["z"], dtype="float32"), target=np.asarray(parent["target"], dtype="float32"),
            kinetic_input=velocity_input(pack, normalizer, variant, row["source_key"]),
            time_hours=float(row["time_hours"]), source_key=row["source_key"]))
    return groups


def create_model(configuration, dimensions, seed):
    condition = source_state_conditions(1, 0.)
    native_condition = dict(condition, population_token=jnp.zeros((1, 1, dimensions)),
                            cell_token=jnp.zeros((1, 1, dimensions)))
    native = create_solver(configuration, native_condition, dimensions, seed).vf
    return official_moe_from_native(native, config=MoEConfig(dimensions=dimensions))


def fit_group_arrays(groups, configuration, seed, *, callback=None):
    """Executable array-only primitive used by real gate and CPU toy fixtures."""
    dimensions = groups[0]["z"].shape[-1]
    model = create_model(configuration, dimensions, seed)
    first = groups[0]
    initial = model.init(jax.random.PRNGKey(seed), jnp.asarray(first["z"][:1]),
        jnp.asarray(first["kinetic_input"][:1]), jnp.asarray(first["z"][:1]), .5,
        source_state_conditions(1, first["time_hours"]))
    state = TrainState.create(apply_fn=model.apply, params=initial["params"], tx=optax.adam(configuration["learning_rate"]))
    inference_params = initial["params"]
    matching = functools.partial(match_linear, epsilon=configuration["ot_epsilon"], scale_cost="mean",
                                  max_iterations=configuration["ot_max_iterations"])
    step = make_official_ot_train_step(model, matching)
    rng = np.random.default_rng(seed)
    history, started = [], time.monotonic()
    for update in range(1, configuration["steps"]+1):
        group = groups[int(rng.integers(len(groups)))]
        count = configuration["batch_size"]
        si = rng.choice(len(group["z"]), count, replace=len(group["z"])<count)
        ti = rng.choice(len(group["target"]), count, replace=len(group["target"])<count)
        state, metrics = step(state, jax.random.fold_in(jax.random.PRNGKey(seed), update),
            jnp.asarray(group["z"][si]), jnp.asarray(group["target"][ti]), jnp.asarray(group["kinetic_input"][si]),
            source_state_conditions(count, group["time_hours"]))
        inference_params = (state.params if configuration.get("ema", 1.) == 1. else
            ema_update(inference_params, state.params, configuration["ema"]))
        if update == 1 or update % configuration.get("log_interval", 250) == 0 or update == configuration["steps"]:
            values = {key:float(value) for key,value in metrics.items() if key not in ("source_indices", "target_indices")}
            if not np.isfinite(list(values.values())).all():
                raise FloatingPointError("Nonfinite L3 task update")
            record = dict(step=update, training_elapsed_seconds=time.monotonic()-started, **values)
            history.append(record)
            if callback is not None:
                callback(record)
    return model, inference_params, history, initial["params"]


def train_real(sidecar, parent_pack, configuration_path, freeze_record, synthetic,
                output, *, variant, seed):
    """No GPU/real-data training until BOTH fixed recipe and full synthetic pass."""
    job = variant_spec(variant)
    if seed not in SEEDS:
        raise ValueError("L3 real training is registered for seeds 0/1/2")
    gate = require_complete_synthetic_gate(synthetic)  # before source/target loading or accelerator use
    from w15a_cellflow_registered_grid import validate_freeze
    configuration, freeze = validate_freeze(configuration_path, freeze_record, parent_pack)
    require_l2_freeze(freeze, configuration_path=configuration_path,
                      adapter_path=HERE / "w15a_cellflow_moe_official.py")
    if gate["l2_configuration_sha256"] != freeze["configuration_sha256"]:
        raise ValueError("Synthetic and real L3 must inherit the same frozen L2 recipe")
    manifest = sidecar_metadata(sidecar)
    if len(manifest["training_groups"]) != 15 or {row["replicate"] for row in manifest["training_groups"]} != {"r1", "r2", "r3"}:
        raise ValueError("L3 requires all fifteen r1-r3 adjacent training transitions")
    groups = load_training_groups(sidecar, variant)
    if any(group["z"].shape[-1] != 50 for group in groups):
        raise ValueError("Real L3 uses the frozen PCA50 state space")
    output = Path(output)
    run = Artifacts(output, seed=seed, inputs=[Path(configuration_path), Path(freeze_record),
        Path(sidecar)/"manifest.json", Path(sidecar)/"normalizer.npz", Path(synthetic)/"summary.json",
        Path(__file__), HERE/"w15a_cellflow_moe.py", HERE/"w15a_cellflow_moe_official.py"])
    model, params, history, initial = fit_group_arrays(groups, configuration, seed,
        callback=lambda row: run.json("status.json", dict(status="training", variant=variant, seed=seed, **row)))
    (output/"model.msgpack").write_bytes(serialization.to_bytes(params))
    run.json("training_history.json", dict(history=history))
    result = dict(status="L3_REAL_DATA_TRAINED", variant=variant, seed=seed, **job,
        config=configuration, dimensions=50, experts=8, engineering_fixture=False,
        sidecar_directory=str(Path(sidecar).resolve()), sidecar_manifest_sha256=sha256(Path(sidecar)/"manifest.json"),
        sidecar_normalizer_sha256=sha256(Path(sidecar)/"normalizer.npz"),
        parent_pack=str(Path(parent_pack).resolve()), freeze_commit=freeze["freeze_commit"], synthetic_gate=gate,
        model_sha256=sha256(output/"model.msgpack"),
        initial_parameters_sha256=__import__("hashlib").sha256(serialization.to_bytes(initial)).hexdigest(),
        parameters=sum(value.size for value in jax.tree_util.tree_leaves(params)),
        all_training_steps=history[-1]["step"], heldout_targets_read=False, backend=[str(d) for d in jax.devices()])
    run.json("summary.json", result); run.json("status.json", dict(status="complete"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", required=True); parser.add_argument("--parent-pack", required=True)
    parser.add_argument("--configuration", required=True); parser.add_argument("--freeze-record", required=True)
    parser.add_argument("--synthetic", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True); parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(train_real(args.sidecar,args.parent_pack,args.configuration,args.freeze_record,
        args.synthetic,args.output,variant=args.variant,seed=args.seed), indent=2))
