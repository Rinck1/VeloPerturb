"""Tiny CPU-only actual-CellFlow L3 integration sanity, NOT a scientific run.

All four equal-parameter arms train on synthetic arrays using official state-only
OT. This checks executable plumbing, finite task gradients, serialization and
weighted independent-expert endpoints, not velocity advantage or the 20+20 gate.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from w15a_cellflow_adapter import add_cellflow_paths, biological_fourier
add_cellflow_paths()

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training.train_state import TrainState
from cellflow.utils import match_linear

from w15a_cellflow_moe import ARMS, MoEConfig, arm_input, integrate_mixture, weighted_energy
from w15a_cellflow_moe_official import (make_official_ot_train_step, official_moe_from_native,
                                        official_source_record)
from w15a_cellflow_train import create_solver


def run_sanity(output, *, steps=20):
    if any(device.platform != "cpu" for device in jax.devices()):
        raise ValueError("This sanity is CPU-only and must not compete for GPUs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if not 1 <= steps <= 100:
        raise ValueError("Bounded engineering sanity accepts only 1..100 updates")
    started = time.monotonic()
    rng = np.random.default_rng(0)
    n, d = 16, 50
    source = jnp.asarray(rng.normal(0, .25, (n, d)), dtype=jnp.float32)
    target = jnp.asarray(rng.normal(0, .25, (n, d)) + .15, dtype=jnp.float32)
    kinetic = jnp.asarray(rng.normal(0, .2, (n, d)), dtype=jnp.float32)
    direct_u = jnp.asarray(rng.normal(0, .2, (n, d)), dtype=jnp.float32)
    condition = dict(perturbation=jnp.ones((n, 1, 1)),
        bio_time=jnp.broadcast_to(jnp.asarray(biological_fourier(5.), dtype=jnp.float32), (n, 1, 8)),
        bio_delta=jnp.broadcast_to(jnp.asarray(biological_fourier(24.), dtype=jnp.float32), (n, 1, 8)))
    full = dict(condition, population_token=jnp.zeros((n, 1, d)), cell_token=jnp.zeros((n, 1, d)))
    # A deliberately tiny inherited architecture for engineering ONLY, not CF-3.
    config = dict(width=16, depth=1, condition_dim=8, time_freqs=8, time_width=16,
        time_depth=1, pooling="mean", learning_rate=.001, ot_epsilon=.1, ot_max_iterations=30)
    native = create_solver(config, {k:v[:1] for k,v in full.items()}, d, 0).vf
    model = official_moe_from_native(native, config=MoEConfig())
    initial = model.init(jax.random.PRNGKey(0), source, kinetic, source, .2, condition)
    initial_bytes = serialization.to_bytes(initial["params"])
    parameter_count = sum(value.size for value in jax.tree_util.tree_leaves(initial["params"]))
    match_fn = functools.partial(match_linear, epsilon=.1, scale_cost="mean", max_iterations=30)
    step = make_official_ot_train_step(model, match_fn)
    permutation = np.random.default_rng(1).permutation(n)
    source_record = official_source_record()
    rows = []
    for arm in ARMS:
        features = arm_input(arm, kinetic, direct_u=direct_u, permutation=permutation)
        state = TrainState.create(apply_fn=model.apply, params=initial["params"], tx=optax.adam(.001))
        if serialization.to_bytes(state.params) != initial_bytes:
            raise AssertionError("Arm initial parameters differ")
        history = []
        for update in range(steps):
            state, metrics = step(state, jax.random.PRNGKey(update + 10), source, target, features, condition)
            metrics = {key: float(value) for key, value in metrics.items()
                       if key not in ("source_indices", "target_indices")}
            if not np.isfinite(list(metrics.values())).all():
                raise FloatingPointError("Nonfinite L3 engineering step")
            history.append(dict(step=update + 1, **metrics))
        path = output / f"{arm}.msgpack"
        path.write_bytes(serialization.to_bytes(state.params))
        restored = serialization.from_bytes(initial["params"], path.read_bytes())
        prediction = integrate_mixture(model, dict(params=restored), source, features,
                                       condition, steps=4, tau_end=.3)
        artifact = output / f"{arm}_predictions.npz"
        np.savez_compressed(artifact, candidates=np.asarray(prediction["candidates"]), q=np.asarray(prediction["q"]))
        score = weighted_energy(prediction["candidates"], prediction["q"], target)
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite weighted mixture score")
        rows.append(dict(arm=arm, initial_loss=history[0]["loss"], final_loss=history[-1]["loss"],
            final_kinetic_gradient_norm=history[-1]["kinetic_gradient_norm"],
            final_router_gradient_norm=history[-1]["router_gradient_norm"],
            parameters=parameter_count, synthetic_tensor_energy=score,
            model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            prediction_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(), history=history))
        (output / "status.json").write_text(json.dumps(dict(status="running", arms_completed=len(rows))))
    if official_source_record() != source_record:
        raise AssertionError("Official source changed during integration sanity")
    result = dict(status="L3_OFFICIAL_CPU_ENGINEERING_SANITY_COMPLETE", runtime_seconds=time.monotonic()-started,
        backend=[str(device) for device in jax.devices()], steps_per_arm=steps, experts=8, dimensions=50,
        arms=rows, backbone_engineering_config=config, official_source=source_record,
        real_data_read=False, formal_L3_grid_started=False, L2_static_configuration_frozen=False,
        biological_velocity_advantage_tested=False, twenty_positive_twenty_null_gate_completed=False,
        conditional_time=.3, initial_parameter_sha256=hashlib.sha256(initial_bytes).hexdigest())
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "status.json").write_text(json.dumps(dict(status="complete")))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    result = run_sanity(args.output, steps=args.steps)
    print(json.dumps(dict(status=result["status"], output=str(Path(args.output).resolve()),
        runtime_seconds=result["runtime_seconds"], dimensions=50, experts=8,
        arms_completed=len(result["arms"]), biological_velocity_advantage_tested=False), indent=2))
