"""Predeclared population-composition synthetic check for the frozen L3 recipe.

This is not a fate-identification test. Source S has the SAME empirical law in
every group, while observed source kappa composition changes between groups.
Targets depend on group composition only in the positive scenario. Pairing is
always official S-only OT; truth labels never enter pairing, model inputs or loss.

Without a committed L2 freeze, only an explicitly requested tiny CPU engineering
smoke is permitted. It never reports the full 20-positive/20-null gate as passed.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.distance import cdist

from w15a_cellflow_adapter import add_cellflow_paths, biological_fourier
add_cellflow_paths()

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training.train_state import TrainState
from cellflow.solvers.utils import ema_update
from cellflow.utils import match_linear

from w15a_cellflow_moe import MoEConfig, arm_input, require_l2_freeze
from w15a_cellflow_moe_official import make_official_ot_train_step, official_moe_from_native
from w15a_cellflow_train import create_solver


ARMS = ("moe_static", "moe_kin", "moe_kin_shuf")
TRAIN_RATIOS = (.1, .3, .7, .9)
TEST_RATIOS = (.2, .4, .6, .8)
SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def engineering_configuration(steps=2):
    if not isinstance(steps, int) or not 1 <= steps <= 20:
        raise ValueError("Explicit engineering smoke is bounded to 1..20 steps per arm")
    return dict(width=8, depth=1, condition_dim=4, time_freqs=8, time_width=8,
        time_depth=1, pooling="mean", learning_rate=.001, ot_epsilon=.1,
        ot_max_iterations=30, ema=1., batch_size=8, steps=steps, log_interval=1,
        evaluation_source_cells=16, evaluation_target_cells=16)


def _code_hashes():
    return {str((SCRIPT_DIRECTORY / name).resolve()): sha256(SCRIPT_DIRECTORY / name)
            for name in (Path(__file__).name, "w15a_cellflow_moe.py", "w15a_cellflow_moe_official.py",
                         "w15a_cellflow_adapter.py", "w15a_cellflow_train.py")}


def prepare_plan(output, *, engineering_smoke=False, steps=2,
                 l2_configuration=None, freeze_record=None):
    """Write every budget, seed and decision threshold BEFORE any fits/scores."""
    output = Path(output)
    if output.exists():
        raise ValueError("Synthetic plan output must be a new directory")
    if engineering_smoke:
        if l2_configuration is not None or freeze_record is not None:
            raise ValueError("Engineering smoke must not masquerade as a frozen L2 recipe")
        configuration = engineering_configuration(steps)
        mode = "engineering_smoke_not_full_gate"
        seeds, count = [0], 16
        freeze = None
    else:
        if l2_configuration is None or freeze_record is None:
            raise ValueError("Full synthetic grid requires a committed L2 freeze")
        record = json.loads(Path(freeze_record).read_text())
        require_l2_freeze(record, configuration_path=l2_configuration,
                          adapter_path=SCRIPT_DIRECTORY / "w15a_cellflow_moe_official.py")
        configuration = json.loads(Path(l2_configuration).read_text())
        mode = "frozen_L2_full_20_positive_20_null"
        seeds = list(range(20))
        count = max(int(configuration.get("evaluation_source_cells", 512)),
                    int(configuration.get("evaluation_target_cells", 512)),
                    int(configuration["batch_size"]))
        freeze = dict(record_path=str(Path(freeze_record).resolve()),
            record_sha256=sha256(freeze_record), configuration_path=str(Path(l2_configuration).resolve()),
            configuration_sha256=sha256(l2_configuration), freeze_commit=record["freeze_commit"])
    output.mkdir(parents=True)
    plan = dict(kind="L3_POPULATION_COMPOSITION_ENGINEERING_CHECK", mode=mode,
        model_configuration=configuration, l2_freeze=freeze, scenario_order=["positive", "null"],
        repetition_seeds=seeds, arms=list(ARMS), dimensions=50, experts=8,
        cells_per_cohort=count, training_composition_ratios=list(TRAIN_RATIOS),
        test_composition_ratios=list(TEST_RATIOS), source_state_sd=.1,
        kinetic_component_amplitude=1., kinetic_noise_sd=.01, target_mode_amplitude=2.,
        target_noise_sd=.1, null_target_high_probability=.5,
        same_source_empirical_distribution_within_each_split=True,
        source_kinetics_independent_of_source_state=True, target_sampling_unpaired=True,
        training_pairing="official_state_only_OT_sample_joint", truth_in_pairing_or_loss=False,
        loss="mean_i_sum_k_qik_sum_d_squared_error_plus_0.01_KL_batch_mean_q_uniform",
        shuffle="one_full_source_cohort_permutation_per_repetition_group_before_source_OT_reindexing",
        shuffle_seed_offset=200000, model_seed_offset=10000, numpy_training_seed_offset=300000,
        primary_evaluation="exact_weighted_empirical_mixture_V_energy_equal_test_cohort_average",
        prediction_tau_end=1., prediction_solver="Tsit5_PID_rtol_atol_1e-5",
        source_biological_hours=5., biological_condition_delta_hours=24.,
        bootstrap_repetitions=1000, bootstrap_seed=123, confidence=.95,
        gate=dict(positive_real_vs_static_relative_gain_median_must_exceed=0.,
            null_suspicious_if_mean_relative_gain_CI_low_exceeds=0.,
            shuffle_is_composition_preserving_equivalence_check_not_superiority_control=True,
            engineering_smoke_may_pass_full_gate=False, biological_significance_claim=False),
        code_sha256=_code_hashes())
    _json(output / "plan.json", plan)
    _json(output / "plan_manifest.json", dict(status="PLAN_FROZEN_BEFORE_FITTING",
        plan_sha256=sha256(output / "plan.json"), scores_read_before_preparation=False))
    return plan


def verify_plan(directory):
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text())
    manifest = json.loads((directory / "plan_manifest.json").read_text())
    if sha256(directory / "plan.json") != manifest["plan_sha256"]:
        raise ValueError("Synthetic plan changed after preparation")
    for path, digest in plan["code_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"Synthetic implementation changed after plan freeze: {path}")
    if plan["mode"] == "frozen_L2_full_20_positive_20_null":
        frozen = plan["l2_freeze"]
        if sha256(frozen["record_path"]) != frozen["record_sha256"]:
            raise ValueError("L2 freeze record changed")
        record = json.loads(Path(frozen["record_path"]).read_text())
        require_l2_freeze(record, configuration_path=frozen["configuration_path"],
                          adapter_path=SCRIPT_DIRECTORY / "w15a_cellflow_moe_official.py")
    return plan


def generate_scenario(plan, scenario, repetition):
    """Generate known truth, then expose it ONLY in a separate scoring mapping."""
    if scenario not in ("positive", "null"):
        raise ValueError("Unknown synthetic scenario")
    rng = np.random.default_rng(int(repetition))
    count, dimensions = plan["cells_per_cohort"], plan["dimensions"]
    result = {}
    for split, ratios in (("train", plan["training_composition_ratios"]),
                          ("test", plan["test_composition_ratios"])):
        source = rng.normal(0, plan["source_state_sd"], (count, dimensions)).astype("float32")
        cohorts = []
        for index, ratio in enumerate(ratios):
            high = np.zeros(count, dtype=bool)
            high[:int(np.floor(count * ratio + .5))] = True
            high = high[rng.permutation(count)]
            kinetic = rng.normal(0, plan["kinetic_noise_sd"], (count, dimensions))
            kinetic[:, 0] += np.where(high, 1., -1.) * plan["kinetic_component_amplitude"]
            target_probability = ratio if scenario == "positive" else plan["null_target_high_probability"]
            target_high = np.zeros(count, dtype=bool)
            target_high[:int(np.floor(count * target_probability + .5))] = True
            target_high = target_high[rng.permutation(count)]
            target = rng.normal(0, plan["target_noise_sd"], (count, dimensions))
            target[:, 0] += np.where(target_high, 1., -1.) * plan["target_mode_amplitude"]
            cohorts.append(dict(group=f"{split}_{index}", source=source.copy(),
                kinetic=kinetic.astype("float32"), target=target.astype("float32"),
                truth=dict(source_component_high=high, target_component_high=target_high,
                    nominal_source_high_probability=ratio,
                    nominal_target_high_probability=target_probability)))
        result[split] = cohorts
    return result


def training_arrays(cohort, arm, *, shuffle_seed):
    """No truth input to the model or OT: source, observed kappa, target snapshot."""
    inputs = arm_input(arm, cohort["kinetic"],
        permutation=np.random.default_rng(shuffle_seed).permutation(len(cohort["source"])))
    return dict(source=jnp.asarray(cohort["source"]), kinetic_input=inputs,
                target=jnp.asarray(cohort["target"]))


def state_conditions(count, plan):
    """Identical condition values across all groups; no group ID leak."""
    return dict(perturbation=jnp.ones((count, 1, 1), dtype=jnp.float32),
        bio_time=jnp.broadcast_to(jnp.asarray(biological_fourier(plan["source_biological_hours"]),
            dtype=jnp.float32), (count, 1, 8)),
        bio_delta=jnp.broadcast_to(jnp.asarray(biological_fourier(plan["biological_condition_delta_hours"]),
            dtype=jnp.float32), (count, 1, 8)))


def make_tsit5_mixture_predictor(model):
    """Every expert/particle follows its own Tsit5 trajectory; q computed once.

    All parameters are runtime arguments, not baked-in JIT constants. Each k is a
    static closure so XLA can discard unselected head outputs. No shared averaged
    velocity is integrated. Production tolerances match L2's frozen wrapper.
    """
    import diffrax
    solvers = []
    for expert in range(model.config.experts):
        def one(params, x, condition, tau_end, expert=expert):
            condition = {key: value[None] for key, value in condition.items()}
            def vf(tau, state, parameters):
                return model.apply(dict(params=parameters), state[None], tau, condition,
                                    method=model.velocities)[0, expert]
            solution = diffrax.diffeqsolve(diffrax.ODETerm(vf), solver=diffrax.Tsit5(),
                t0=0., t1=tau_end, dt0=None, y0=x, args=params,
                stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5))
            return solution.ys[0]
        solvers.append(jax.jit(jax.vmap(one, in_axes=(None, 0, 0, None))))
    def predict(params, source, kinetics, condition, *, tau_end=1.):
        q = model.apply(dict(params=params), source, kinetics, method=model.route)
        if tau_end == 0:
            candidates = jnp.repeat(source[:, None], model.config.experts, axis=1)
        else:
            candidates = jnp.stack([solve(params, source, condition, jnp.asarray(tau_end))
                                    for solve in solvers], axis=1)
        return dict(candidates=np.asarray(candidates), q=np.asarray(q), routing_evaluations=1)
    return predict


def blocked_weighted_energy(candidates, q, target, block=256):
    """Exact empirical-mixture V energy without [N*K,N*K,D] memory blow-up."""
    candidates, q, target = [np.asarray(value, dtype="float64") for value in (candidates, q, target)]
    if candidates.ndim != 3 or q.shape != candidates.shape[:2] or target.ndim != 2:
        raise ValueError("Invalid empirical mixture shapes")
    if target.shape[-1] != candidates.shape[-1] or not len(q) or not len(target):
        raise ValueError("Empty or incompatible empirical mixture")
    if not all(np.isfinite(value).all() for value in (candidates, q, target)):
        raise ValueError("Nonfinite empirical mixture")
    if (q < 0).any() or not np.allclose(q.sum(1), 1, atol=1e-6):
        raise ValueError("Mixture q must be row-stochastic")
    source = candidates.reshape(-1, candidates.shape[-1])
    weights = q.ravel() / len(q)
    cross = within = target_within = 0.
    for first in range(0, len(source), block):
        x, w = source[first:first+block], weights[first:first+block]
        for second in range(0, len(target), block):
            cross += w @ cdist(x, target[second:second+block]).sum(1) / len(target)
        for second in range(0, len(source), block):
            within += w @ cdist(x, source[second:second+block]) @ weights[second:second+block]
    for first in range(0, len(target), block):
        for second in range(0, len(target), block):
            target_within += cdist(target[first:first+block], target[second:second+block]).sum() / len(target)**2
    return float(2 * cross - within - target_within)


def _score(prediction, cohort):
    candidates, q = prediction["candidates"], prediction["q"].astype("float64")
    mean = np.einsum("nk,nkd->d", q, candidates) / len(q)
    high = float(np.mean(np.sum(q * (candidates[:, :, 0] > 0), axis=1)))
    observed = float(np.mean(cohort["truth"]["target_component_high"]))
    average_q = q.mean(0)
    entropy = -np.sum(average_q * np.log(np.maximum(average_q, 1e-30)))
    energy = blocked_weighted_energy(candidates, q, cohort["target"])
    return dict(energy=energy, energy_per_sqrt_dimension=energy/np.sqrt(candidates.shape[-1]),
        mean_mse=float(np.mean((mean-cohort["target"].mean(0))**2)),
        predicted_high_mode_probability=high, observed_high_mode_probability=observed,
        mode_proportion_TV=abs(high-observed), effective_experts=float(np.exp(entropy)),
        population_q=average_q.tolist())


def _write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _summarize(plan, repetition_rows):
    rng = np.random.default_rng(plan["bootstrap_seed"])
    comparisons = []
    for scenario in plan["scenario_order"]:
        for arm in ("moe_kin", "moe_kin_shuf"):
            differences = []
            relative = []
            for repetition in plan["repetition_seeds"]:
                values = {row["arm"]: row["mean_energy"] for row in repetition_rows
                          if row["scenario"] == scenario and row["repetition"] == repetition}
                difference = values["moe_static"] - values[arm]
                differences.append(difference)
                relative.append(difference / max(values["moe_static"], 1e-12))
            x = np.asarray(relative)
            samples = x[rng.integers(len(x), size=(plan["bootstrap_repetitions"], len(x)))].mean(1)
            alpha = (1-plan["confidence"])/2
            low, high = np.quantile(samples, (alpha, 1-alpha))
            comparisons.append(dict(scenario=scenario, arm=arm, control="moe_static",
                relative_gains=relative, absolute_energy_differences=differences,
                mean_relative_gain=float(x.mean()), median_relative_gain=float(np.median(x)),
                bootstrap_mean_ci_low=float(low), bootstrap_mean_ci_high=float(high)))
    full = plan["mode"] == "frozen_L2_full_20_positive_20_null"
    real = {row["scenario"]: row for row in comparisons if row["arm"] == "moe_kin"}
    return dict(full_positive_null_grid_completed=full,
        population_composition_learning_gate_passed=(
            real["positive"]["median_relative_gain"] > 0 and real["null"]["bootstrap_mean_ci_low"] <= 0) if full else None,
        comparisons=comparisons, shuffle_superiority_required=False,
        biological_velocity_specificity_established=False, individual_fate_identified=False,
        false_positive_rate_calibrated=False)


def run_plan(directory, *, acknowledge_full_run=False):
    directory = Path(directory)
    plan = verify_plan(directory)
    engineering = plan["mode"] == "engineering_smoke_not_full_gate"
    if not engineering and not acknowledge_full_run:
        raise ValueError("Full synthetic execution requires explicit acknowledgement after L2 freeze")
    if engineering and any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("Unfrozen engineering smoke is CPU-only")
    if (directory / "summary.json").exists() or (directory / "fits").exists():
        raise ValueError("Existing synthetic results cannot be overwritten or silently resumed")
    started = time.monotonic()
    configuration = plan["model_configuration"]
    dummy = state_conditions(1, plan)
    dummy_full = dict(dummy, population_token=jnp.zeros((1, 1, 50)), cell_token=jnp.zeros((1, 1, 50)))
    native = create_solver(configuration, dummy_full, 50, 0).vf
    model = official_moe_from_native(native, config=MoEConfig())
    matching = functools.partial(match_linear, epsilon=configuration["ot_epsilon"], scale_cost="mean",
                                max_iterations=configuration["ot_max_iterations"])
    step = make_official_ot_train_step(model, matching)
    predict = make_tsit5_mixture_predictor(model)
    repetition_rows, test_rows = [], []
    for scenario in plan["scenario_order"]:
        for repetition in plan["repetition_seeds"]:
            data = generate_scenario(plan, scenario, repetition)
            initial_group = data["train"][0]
            initial = model.init(jax.random.PRNGKey(plan["model_seed_offset"]+repetition),
                jnp.asarray(initial_group["source"][:1]), jnp.asarray(initial_group["kinetic"][:1]),
                jnp.asarray(initial_group["source"][:1]), .5, dummy)
            initial_bytes = serialization.to_bytes(initial["params"])
            parameters = sum(value.size for value in jax.tree_util.tree_leaves(initial["params"]))
            for arm in plan["arms"]:
                fit_directory = directory / "fits" / scenario / f"repetition{repetition:02d}" / arm
                fit_directory.mkdir(parents=True)
                state = TrainState.create(apply_fn=model.apply, params=initial["params"],
                    tx=optax.adam(configuration["learning_rate"]))
                inference_params = initial["params"]
                groups = [training_arrays(group, arm,
                    shuffle_seed=plan["shuffle_seed_offset"]+repetition*100+index)
                    for index, group in enumerate(data["train"])]
                batches = np.random.default_rng(plan["numpy_training_seed_offset"]+repetition)
                history = []
                for update in range(1, configuration["steps"]+1):
                    group = groups[int(batches.integers(len(groups)))]
                    count = configuration["batch_size"]
                    si = batches.choice(len(group["source"]), count, replace=len(group["source"])<count)
                    ti = batches.choice(len(group["target"]), count, replace=len(group["target"])<count)
                    state, metrics = step(state,
                        jax.random.fold_in(jax.random.PRNGKey(plan["model_seed_offset"]+repetition), update),
                        group["source"][si], group["target"][ti], group["kinetic_input"][si],
                        state_conditions(count, plan))
                    inference_params = (state.params if configuration.get("ema", 1.) == 1. else
                        ema_update(inference_params, state.params, configuration["ema"]))
                    if update == 1 or update % configuration.get("log_interval", 250) == 0 or update == configuration["steps"]:
                        values = {key:float(value) for key,value in metrics.items()
                                  if key not in ("source_indices", "target_indices")}
                        if not np.isfinite(list(values.values())).all():
                            raise FloatingPointError("Nonfinite synthetic training metrics")
                        history.append(dict(step=update, **values))
                (fit_directory / "model.msgpack").write_bytes(serialization.to_bytes(inference_params))
                _json(fit_directory / "training_history.json", history)
                scores = []
                for index, cohort in enumerate(data["test"]):
                    inputs = training_arrays(cohort, arm,
                        shuffle_seed=plan["shuffle_seed_offset"]+repetition*100+50+index)
                    prediction = predict(inference_params, inputs["source"], inputs["kinetic_input"],
                        state_conditions(len(inputs["source"]), plan), tau_end=plan["prediction_tau_end"])
                    path = fit_directory / f"{cohort['group']}_predictions.npz"
                    np.savez_compressed(path, candidates=prediction["candidates"], q=prediction["q"])
                    score = _score(prediction, cohort)
                    row = dict(scenario=scenario, repetition=repetition, arm=arm, group=cohort["group"],
                        source_high_probability=cohort["truth"]["nominal_source_high_probability"],
                        source_cells=len(inputs["source"]), target_cells=len(cohort["target"]), **score)
                    scores.append(score); test_rows.append(row)
                summary = dict(scenario=scenario, repetition=repetition, arm=arm,
                    mean_energy=float(np.mean([row["energy"] for row in scores])),
                    mean_mode_proportion_TV=float(np.mean([row["mode_proportion_TV"] for row in scores])),
                    mean_effective_experts=float(np.mean([row["effective_experts"] for row in scores])),
                    final_kinetic_gradient_norm=history[-1]["kinetic_gradient_norm"],
                    final_router_gradient_norm=history[-1]["router_gradient_norm"], parameters=parameters,
                    initial_parameter_sha256=hashlib.sha256(initial_bytes).hexdigest(),
                    model_sha256=sha256(fit_directory / "model.msgpack"))
                repetition_rows.append(summary); _json(fit_directory / "summary.json", summary)
                _json(directory / "status.json", dict(status="running", completed_fits=len(repetition_rows),
                    expected_fits=2*len(plan["repetition_seeds"])*len(plan["arms"])))
    _write_csv(directory / "all_repetitions.csv", repetition_rows)
    _write_csv(directory / "all_test_cohorts.csv", test_rows)
    result = dict(status="L3_POPULATION_ENGINEERING_SMOKE_COMPLETE" if engineering else "L3_POPULATION_SYNTHETIC_GRID_COMPLETE",
        mode=plan["mode"], runtime_seconds=time.monotonic()-started, backend=[str(d) for d in jax.devices()],
        completed_fits=len(repetition_rows), plan_sha256=sha256(directory / "plan.json"),
        true_pairing_used=False, real_biological_data_read=False, **_summarize(plan, repetition_rows))
    _json(directory / "summary.json", result); _json(directory / "status.json", dict(status="complete"))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "smoke"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--l2-configuration")
    parser.add_argument("--freeze-record")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--acknowledge-full-run", action="store_true")
    args = parser.parse_args()
    if args.action in ("prepare", "smoke"):
        prepare_plan(args.output, engineering_smoke=args.action=="smoke", steps=args.steps,
            l2_configuration=args.l2_configuration, freeze_record=args.freeze_record)
    if args.action in ("run", "smoke"):
        result = run_plan(args.output, acknowledge_full_run=args.acknowledge_full_run)
        print(json.dumps({key:result[key] for key in ("status", "completed_fits", "mode", "runtime_seconds")}, indent=2))
