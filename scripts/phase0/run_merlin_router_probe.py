"""Held-out-clone router increment probe on the frozen MeRLin Day21 audit.

Program scores are expression-derived, so this is explicitly a diagnostic of
incremental predictability, not an independent biological endpoint claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--steps", type=int, default=800); ap.add_argument("--seed", type=int, default=20260923); args = ap.parse_args()
    data = np.load(Path(args.input), allow_pickle=False)
    z, v, sv, y = data["z"].astype("float32"), data["velocity"].astype("float32"), data["shuffled_velocity"].astype("float32"), data["program_scores"].astype("float32")
    groups = data["clone_ids"].astype(str)
    rng = np.random.default_rng(args.seed)
    unique = np.unique(groups); rng.shuffle(unique); held = set(unique[:max(1, len(unique)//5)])
    test = np.array([g in held for g in groups]); train = ~test
    import torch
    from torch import nn
    torch.manual_seed(args.seed); device = torch.device(args.device)
    def fit(x):
        mean, scale = x[train].mean(0), x[train].std(0).clip(min=1e-4)
        ym, ys = y[train].mean(0), y[train].std(0).clip(min=1e-4)
        xt = torch.as_tensor((x-mean)/scale, device=device)
        yt = torch.as_tensor((y-ym)/ys, device=device)
        model = nn.Sequential(nn.Linear(x.shape[1], 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, y.shape[1])).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        for _ in range(args.steps):
            pred = model(xt[train]); loss = (pred-yt[train]).square().mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = model(xt).cpu().numpy()*ys+ym
        mse = float(np.mean((pred[test]-y[test])**2)); base = float(np.mean((y[test]-y[train].mean(0))**2))
        return dict(mse=mse, normalized_mse=mse/max(base,1e-12), test_cells=int(test.sum()), test_clones=len(held))
    result = {"status":"MERLIN_DAY21_ROUTER_INCREMENT_PROBE_COMPLETE", "scope":"held_out_clone_diagnostic", "seed": args.seed, "models": {
        "z_only": fit(z), "z_plus_gfg_velocity": fit(np.c_[z,v]), "z_plus_shuffled_velocity": fit(np.c_[z,sv])},
        "expression_derived_targets": True, "formal_claim": False}
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True); (out/"summary.json").write_text(json.dumps(result, indent=2)); (out/"RESULTS.md").write_text("# MeRLin router increment probe\n\n"+json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
