"""Four paired-truth toy systems for ENGINEERING, not real-data G1 evidence."""
from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import torch
from torch.nn import functional as F

from .artifacts import Run, save_csv, save_json
from .metrics import energy_distance
from .model import (ModelConfig, VeloRoute, expert_regression_loss, load_checkpoint,
                    routing_loss, save_checkpoint)

SYSTEMS = {"single_no_increment", "bimodal_position", "bimodal_velocity", "single_endpoint_path_velocity"}


def generate(system, n, seed):
    if system not in SYSTEMS:
        raise ValueError("Unknown synthetic system")
    generator = torch.Generator().manual_seed(seed)
    z0 = torch.randn(n, 2, generator=generator)*.15
    hidden = torch.randint(2, (n,), generator=generator)
    velocity = torch.zeros(n, 2)
    velocity[:, 0] = hidden.float()*2-1
    condition = torch.zeros(n, 2)
    condition[:, 0] = 1.
    if system in {"bimodal_velocity", "single_endpoint_path_velocity"}:
        z0.zero_()  # exact same S, different V: no hidden state can leak through position
    modes = (z0[:, 0] > 0).long() if system == "bimodal_position" else hidden
    if system == "single_no_increment":
        modes = torch.zeros(n, dtype=torch.long)
    return {"z0": z0, "velocity": velocity, "condition": condition, "modes": modes}


def true_path(system, batch, s):
    """Return state and instantaneous physical rate, with a one-day interval."""
    z0, modes = batch["z0"], batch["modes"]
    s = torch.as_tensor(s, dtype=z0.dtype).expand(len(z0))
    direction = modes.float()*2-1
    rate = torch.zeros_like(z0)
    displacement = torch.zeros_like(z0)
    rate[:, 1] = .1
    displacement[:, 1] = .1*s
    if system == "single_no_increment":
        rate[:, 0] = .3
        displacement[:, 0] = .3*s
    elif system == "single_endpoint_path_velocity":
        rate[:, 0] = direction*torch.pi*torch.cos(torch.pi*s)
        displacement[:, 0] = direction*torch.sin(torch.pi*s)
    else:
        rate[:, 0] = direction*1.5
        displacement[:, 0] = direction*1.5*s
    return z0+displacement, rate


def _subset(batch, indices):
    return {key: value[indices] for key, value in batch.items()}


def train_one(system, seed, config):
    torch.manual_seed(seed)
    n_experts = 1 if system == "single_no_increment" else 2
    cfg = ModelConfig(state_dim=2, velocity_dim=2, condition_dim=2, hidden_dim=config["hidden_dim"],
                      router_hidden_dim=config["hidden_dim"], time_hidden_dim=16, residual_blocks=1,
                      expert_rank=8, n_experts=n_experts, top_k=n_experts)
    model = VeloRoute(cfg)
    train = generate(system, config["train_cells"], seed+10000)
    test = generate(system, config["test_cells"], seed+20000)
    generator = torch.Generator().manual_seed(seed+30000)
    optimizer = torch.optim.Adam(model.field.parameters(), lr=config["learning_rate"])
    trace = []
    for step in range(config["expert_steps"]):
        indices = torch.randint(len(train["z0"]), (config["batch_size"],), generator=generator)
        batch = _subset(train, indices)
        s = torch.rand(len(indices), generator=generator)
        state, rate = true_path(system, batch, s)
        # Ground-truth responsibilities/path rates are AVAILABLE ONLY IN THIS TOY.
        weights = F.one_hot(batch["modes"], n_experts).float()
        optimizer.zero_grad(set_to_none=True)
        loss = expert_regression_loss(model, state, 4+s, batch["condition"], rate, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.field.parameters(), 1.)
        optimizer.step()
        if step == 0 or (step+1) % 50 == 0:
            trace.append({"system": system, "seed": seed, "stage": "experts", "step": step+1, "loss": loss.item()})
    static = VeloRoute(replace(cfg, use_velocity=False))
    static.field.load_state_dict(model.field.state_dict())
    static.router.load_state_dict(model.router.state_dict())  # same initialization before either router is fitted
    for candidate in (model, static):
        candidate.field.requires_grad_(False)
        candidate.field.zero_grad(set_to_none=True)
    optimizers = [torch.optim.Adam(candidate.router.parameters(), lr=config["learning_rate"])
                  for candidate in (model, static)]
    for step in range(config["router_steps"] if n_experts > 1 else 0):
        indices = torch.randint(len(train["z0"]), (config["batch_size"],), generator=generator)
        batch = _subset(train, indices)
        weights = F.one_hot(batch["modes"], n_experts).float()
        for name, candidate, optimizer in zip(("real", "static"), (model, static), optimizers):
            optimizer.zero_grad(set_to_none=True)
            loss = routing_loss(candidate, batch["z0"], batch["velocity"], batch["condition"], weights)
            loss.backward()
            optimizer.step()
            if step == 0 or (step+1) % 50 == 0:
                trace.append({"system": system, "seed": seed, "stage": f"router_{name}", "step": step+1, "loss": loss.item()})
    return model.eval(), static.eval(), test, trace


def run_synthetic(config, output, config_path):
    if config.get("kind") != "synthetic" or not set(config["systems"]) <= SYSTEMS:
        raise ValueError("Synthetic runner accepts only declared toy systems")
    torch.set_num_threads(config["cpu_threads"])
    metrics, traces, checks = [], [], []
    with Run(output, stage="core_network_acceptance", kind="synthetic", config=config,
             inputs=[config_path], seed=config["seeds"]) as run:
        for system in config["systems"]:
            for seed in config["seeds"]:
                print(f"Training synthetic {system}, seed {seed}", flush=True)
                model, static, test, trace = train_one(system, seed, config)
                traces.extend(trace)
                true_endpoint, _ = true_path(system, test, 1.)
                true_midpoint, _ = true_path(system, test, .5)
                permutation = torch.randperm(len(test["z0"]), generator=torch.Generator().manual_seed(seed+40000))
                arms = [("real_velocity_router", model, test["velocity"]),
                        ("static_router_frozen_same_experts", static, test["velocity"]),
                        ("velocity_shuffled_at_inference", model, test["velocity"][permutation])]
                arm_rows, predictions = {}, {}
                for name, candidate, velocity in arms:
                    result = candidate.predict(test["z0"], velocity, test["condition"],
                                               max_step=config["max_step_days"], return_trajectory=True,
                                               generator=torch.Generator().manual_seed(seed+50000))
                    index = min(range(len(result.times)), key=lambda i: abs(result.times[i]-4.5))
                    if abs(result.times[index]-4.5) > 1e-6:
                        raise ValueError("Toy protocol requires an exact 4.5-day snapshot")
                    usage = result.probabilities.mean(0)
                    row = {"system": system, "seed": seed, "arm": name,
                           "routing_accuracy": float((result.probabilities.argmax(-1) == test["modes"]).float().mean()),
                           "paired_endpoint_mse": float((result.endpoint-true_endpoint).square().mean()),
                           "paired_midpoint_mse": float((result.trajectory[index]-true_midpoint).square().mean()),
                           "endpoint_energy_distance": energy_distance(result.endpoint.numpy(), true_endpoint.numpy()),
                           "effective_experts": float((-(usage*usage.clamp_min(1e-12).log()).sum()).exp()),
                           "total_parameters": sum(p.numel() for p in candidate.parameters()),
                           "trainable_parameters_at_routing_stage": sum(p.numel() for p in candidate.parameters() if p.requires_grad),
                           "evidence_kind": "paired_truth_synthetic_not_G1"}
                    metrics.append(row)
                    arm_rows[name] = row
                    predictions[name] = result
                checkpoint = run.directory / f"{system}_seed{seed}.pt"
                save_checkpoint(checkpoint, model, metadata={"kind": "synthetic", "system": system, "seed": seed,
                                                            "truth_modes_used": True, "G1": "not_run"})
                restored, _ = load_checkpoint(checkpoint)
                restored.eval()
                reference = predictions["real_velocity_router"]
                reloaded = restored.predict(test["z0"], test["velocity"], test["condition"],
                                             max_step=config["max_step_days"], modes=reference.modes)
                roundtrip_error = float((reference.endpoint-reloaded.endpoint).abs().max())
                checks.append({"system": system, "seed": seed, "check": "checkpoint_roundtrip",
                               "passed": roundtrip_error < 1e-6, "value": roundtrip_error})
                limits = config["acceptance"]
                real = arm_rows["real_velocity_router"]
                if system in {"bimodal_velocity", "single_endpoint_path_velocity"}:
                    for control in ("static_router_frozen_same_experts", "velocity_shuffled_at_inference"):
                        advantage = real["routing_accuracy"]-arm_rows[control]["routing_accuracy"]
                        passed = (real["routing_accuracy"] >= limits["hidden_state_router_accuracy_minimum"]
                                  and advantage >= limits["hidden_state_advantage_minimum"])
                        checks.append({"system": system, "seed": seed, "check": f"velocity_route_vs_{control}",
                                       "passed": passed, "value": advantage})
                elif system == "bimodal_position":
                    value = arm_rows["static_router_frozen_same_experts"]["routing_accuracy"]
                    checks.append({"system": system, "seed": seed, "check": "position_static_routing",
                                   "passed": value >= limits["position_static_router_accuracy_minimum"], "value": value})
                else:
                    error = float((predictions["real_velocity_router"].endpoint-predictions["velocity_shuffled_at_inference"].endpoint).abs().max())
                    checks.append({"system": system, "seed": seed, "check": "K1_velocity_invariance_not_a_gate",
                                   "passed": error <= limits["no_increment_velocity_invariance_atol"], "value": error})
                np.savez_compressed(run.directory / f"{system}_seed{seed}_predictions.npz",
                                    z0=test["z0"].numpy(), velocity=test["velocity"].numpy(),
                                    true_mode=test["modes"].numpy(), true_endpoint=true_endpoint.numpy(),
                                    predicted_endpoint=reference.endpoint.numpy(), mode_probabilities=reference.probabilities.numpy())
                print({"system": system, "seed": seed, "routing_accuracy": real["routing_accuracy"],
                       "endpoint_mse": real["paired_endpoint_mse"]}, flush=True)
        save_csv(run.directory / "metrics.csv", metrics)
        save_csv(run.directory / "training_trace.csv", traces)
        save_csv(run.directory / "acceptance_checks.csv", checks)
        passed = all(row["passed"] for row in checks)
        decision = {"status": "ENGINEERING_PASS" if passed else "ENGINEERING_FAIL", "kind": "synthetic",
                    "stage": "core_network_acceptance", "G1": "not_run", "checks": len(checks),
                    "passed_checks": sum(row["passed"] for row in checks), "real_data_training_authorized": False}
        save_json(run.directory / "decision.json", decision)
        lines = ["# 核心网络合成验收", "", f"判决：{decision['status']}，{decision['passed_checks']}/{len(checks)} 项检查通过。",
                 "", "范围：四类已知真值系统，独立训练/测试样本，每类 3 seeds；不是 RENGE，不是 G1。",
                 "", "| 系统 | real 路由准确率（seed 均值） | static 路由准确率 | shuffled 路由准确率 |", "| --- | ---: | ---: | ---: |"]
        for system in config["systems"]:
            values = [np.mean([r["routing_accuracy"] for r in metrics if r["system"] == system and r["arm"] == arm])
                      for arm in ("real_velocity_router", "static_router_frozen_same_experts", "velocity_shuffled_at_inference")]
            lines.append(f"| {system} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")
        lines.extend(["", "解释边界：", "",
                      "- 专家用合成真实模式/路径率监督，然后冻结；static/real router 使用相同专家与相同容量预算。真实未配对数据没有这种真值。",
                      "- shuffled 是推理期干预，不是正式 G1 的训练与推理全通路置换臂。",
                      "- 相同 S、不同 velocity 的路由区分能力不等于群体分布增益；静态混合也可能匹配终态边际。",
                      "- 单终态/不同路径系统可有两个路径专家，但不能据此宣称两个终态命运模式。",
                      "- K=1 的 velocity 不变性不是可靠度门控关闭；本版没有可靠度门控或自动模式数恢复。",
                      "- 所有 checkpoint 明确标记 synthetic；不能解锁正式 RENGE 训练。"])
        (run.directory / "RESULTS.md").write_text("\n".join(lines)+"\n")
    return decision
