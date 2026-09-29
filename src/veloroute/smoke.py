"""Full configured dimensions: gradients, coupling, RK4 and checkpoint roundtrip."""
from dataclasses import asdict

import torch

from .artifacts import Run, save_csv, save_json
from .coupling import build_responsibilities
from .model import ModelConfig, VeloRoute, flow_matching_loss, load_checkpoint, routing_loss, save_checkpoint


def run_core_smoke(config, output, config_path, seed=7):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    cfg = ModelConfig(**config["model"])
    model = VeloRoute(cfg)
    n = 32
    z0 = torch.randn(n, cfg.state_dim)*.1
    velocity = torch.randn(n, cfg.velocity_dim)
    condition = torch.randn(1, cfg.condition_dim).expand(n, -1).clone()
    # Engineering inputs only; no RENGE matrices are opened by this runner.
    target = z0 + torch.randn(n, cfg.state_dim)*.05
    labels = ["synthetic_condition"]*n
    with Run(output, stage="configured_core_smoke", kind="engineering", config=config,
             inputs=[config_path], seed=seed) as run:
        coupling = build_responsibilities(model, z0, target, velocity, condition,
                                           source_conditions=labels, target_conditions=labels,
                                           lambda_dynamic=.2, iterations=500)
        weights = coupling["source_responsibilities"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        rows = []
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            field_loss = flow_matching_loss(model, z0, target, condition, weights, t0=4., t1=5.)
            route_loss = routing_loss(model, z0, velocity, condition, weights)
            loss = field_loss+route_loss
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            rows.append({"step": step+1, "field_loss": field_loss.item(), "routing_loss": route_loss.item()})
        model.eval()
        result = model.predict(z0, velocity, condition, return_trajectory=True,
                               generator=torch.Generator().manual_seed(seed))
        save_checkpoint(run.directory/"core_smoke.pt", model,
                        metadata={"kind": "engineering", "real_data_used": False, "G1": "not_run"})
        restored, _ = load_checkpoint(run.directory/"core_smoke.pt")
        restored.eval()
        again = restored.predict(z0, velocity, condition, modes=result.modes)
        roundtrip_error = float((again.endpoint-result.endpoint).abs().max())
        if roundtrip_error > 1e-6:
            raise RuntimeError("Checkpoint predictions do not reproduce")
        summary = {"status": "ENGINEERING_PASS", "state_dim": cfg.state_dim,
                   "condition_dim": cfg.condition_dim, "n_experts": cfg.n_experts,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "router_parameters": sum(p.numel() for p in model.router.parameters()),
                   "field_parameters": sum(p.numel() for p in model.field.parameters()),
                   "prediction_shape": list(result.endpoint.shape), "RK4_steps": len(result.times)-1,
                   "checkpoint_roundtrip_max_error": roundtrip_error,
                   "coupling_row_marginal_error": coupling["row_marginal_error"],
                   "coupling_column_marginal_error": coupling["column_marginal_error"],
                   "device": "CPU", "torch_version": str(torch.__version__), "G1": "not_run"}
        save_json(run.directory/"summary.json", summary)
        save_csv(run.directory/"metrics.csv", rows)
        (run.directory/"RESULTS.md").write_text("# Configured core smoke test\n\nAll inputs are synthetic random tensors. Three optimization steps, responsibility construction, RK4 prediction and checkpoint reload completed.\n\n```json\n"+__import__("json").dumps(summary, indent=2)+"\n```\n")
    return summary
