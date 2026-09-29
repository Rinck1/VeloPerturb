"""Held-out-clone MeRLin probe: can Day0 GFG velocity predict Day21 fate programs?

This is the first longitudinal velocity experiment.  GFG is prepared on train
clones only; validation/confirmation GFG calls use frozen train normalization.
Day21 expression is used only to construct held-out outcome labels, never as a
source feature.  The result is diagnostic and does not approve the full router.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from veloroute.artifacts import sha256, save_csv, save_json
from veloroute.contracts import FitScope
from veloroute.full_model import FullConfig, FullVeloRoute
from veloroute.latent import FrozenSplicingTransform, load_pack
from veloroute.preprocess import load_usa


CHECKPOINT = Path("/data/yuchang/GFG/results/mousebrain_graphbatch_directed_v1_seed0/final.pth")
CHECKPOINT_SHA256 = "a9d9fa39063312230e4ba64309e9264ba3c2cb7ac549f04544b366df57c1340b"
GTF = Path("/data/yuchang/veloroute_kang_20260914/relocated/gencode32/gencode.v32.primary_assembly.annotation.gtf.gz")
PROGRAMS = Path("/data/yuchang/veloroute_ucheck_20260915/merlin_programs.json")


def read_clones(path: Path) -> dict[str, str]:
    with path.open() as handle:
        return {r["cell_barcode"].split("-", 1)[0]: r["clone_barcode"] for r in csv.DictReader(handle)}


def read_symbols(path: Path) -> dict[str, str]:
    result = {}
    with gzip.open(path, "rt", errors="ignore") as handle:
        for line in handle:
            if line.startswith("#") or "\tgene\t" not in line:
                continue
            fields = line.rstrip("\n").split("\t")
            attrs = {}
            for item in fields[8].split(";"):
                item = item.strip()
                if " " in item:
                    key, value = item.split(" ", 1)
                    attrs[key] = value.strip('"')
            if attrs.get("gene_id") and attrs.get("gene_name"):
                result.setdefault(attrs["gene_id"].split(".")[0], attrs["gene_name"])
    return result


def sidecar(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def parse_source_id(cell_id: str) -> tuple[str, str]:
    _, run, barcode = cell_id.split(":", 2)
    return run, barcode


def parse_target_id(cell_id: str) -> str:
    _, _, barcode = cell_id.split(":", 2)
    return barcode


def raw_gene_us(payload, rows: list[dict[str, str]], gene_index: np.ndarray) -> np.ndarray:
    barcodes, genes, mats, _ = payload
    lookup = {str(b).split("-", 1)[0]: i for i, b in enumerate(barcodes)}
    ix = np.asarray([lookup[parse_source_id(r["cell_id"])[1]] for r in rows], dtype=int)
    # Match prepare_gfg_inputs: library size is computed on the full S matrix
    # before selecting the frozen 2,000-gene panel.
    full_s = mats[0][ix].tocsr()
    s_sparse = full_s[:, gene_index].tocsr()
    u_sparse = mats[1][ix][:, gene_index].tocsr()
    depth = np.asarray(full_s.sum(axis=1)).ravel().astype("float32")
    scale = 10000.0 / np.maximum(depth, 1.0)
    s = s_sparse.multiply(scale[:, None]).toarray().astype("float32")
    u = u_sparse.multiply(scale[:, None]).toarray().astype("float32")
    return np.concatenate((u, s), axis=1)


def target_programs(payload, rows: list[dict[str, str]], program_defs: dict, gene_symbols: dict) -> tuple[np.ndarray, list[str]]:
    barcodes, genes, mats, _ = payload
    lookup = {str(b).split("-", 1)[0]: i for i, b in enumerate(barcodes)}
    ix = np.asarray([lookup[parse_target_id(r["cell_id"])] for r in rows], dtype=int)
    # The target pack is S-only; keep the expression-derived outcome on the
    # same measured modality rather than adding future ambiguous counts.
    expression = mats[0][ix].tocsr().astype("float64")
    depth = np.asarray(expression.sum(axis=1)).ravel()
    expression = expression.multiply((1e4 / np.maximum(depth, 1.0))[:, None]).tocsr()
    expression.data = np.log1p(expression.data)
    symbol_to_ix: dict[str, list[int]] = {}
    for i, gene in enumerate(genes):
        symbol_to_ix.setdefault(gene_symbols.get(str(gene).split(".")[0], ""), []).append(i)
    names, scores = [], []
    for name, definition in program_defs.items():
        wanted = {str(x).split()[0].replace("*", "") for x in definition if not str(x).startswith("**")}
        selected = [i for symbol in wanted for i in symbol_to_ix.get(symbol, [])]
        if len(selected) < 3:
            continue
        scores.append(np.asarray(expression[:, selected].mean(axis=1)).ravel())
        names.append(name)
    return np.stack(scores, axis=1).astype("float32"), names


def clone_targets(rows: list[dict[str, str]], programs: np.ndarray) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["clone_id"], []).append(i)
    return {clone: programs[ix].mean(0) for clone, ix in groups.items()}


def outcome_for_source(rows: list[dict[str, str]], clone_y: dict[str, np.ndarray], width: int) -> np.ndarray:
    return np.stack([clone_y.get(row["clone_id"], np.zeros(width, dtype="float32")) for row in rows]).astype("float32")


def metrics(y: np.ndarray, pred: np.ndarray, rows: list[dict[str, str]]) -> dict:
    groups = np.asarray([r["clone_id"] for r in rows])
    cell_mse = float(np.mean((pred - y) ** 2))
    clone_values = []
    for clone in np.unique(groups):
        ix = groups == clone
        clone_values.append(np.mean((pred[ix] - y[ix]) ** 2))
    baseline = float(np.mean((y - y.mean(0)) ** 2))
    return {"cell_mse": cell_mse, "clone_equal_mse": float(np.mean(clone_values)),
            "normalized_cell_mse": cell_mse / max(baseline, 1e-12), "n_cells": int(len(y)),
            "n_clones": int(len(np.unique(groups)))}


def fit_eval(x_train, y_train, rows_train, x_eval, y_eval, rows_eval):
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(x_train, y_train)
    return model, metrics(y_eval, model.predict(x_eval), rows_eval)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("/data/yuchang/veloroute_ucheck_20260915"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260929)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("GFG checkpoint hash mismatch")
    transform = FrozenSplicingTransform.load(args.fold / "transform.npz")
    # Read each USA matrix once.  Re-reading a 60k-gene MatrixMarket file per
    # gene or per split silently turns this diagnostic into an hours-long I/O job.
    payload_308 = load_usa(args.root / "quant_merlin308" / "af_quant")
    genes_308 = list(payload_308[1])
    gene_index = np.asarray([genes_308.index(g) for g in transform.gene_ids], dtype=int)
    program_defs = json.loads(PROGRAMS.read_text())["signatures"]
    symbols = read_symbols(GTF)

    packs, sidecars = {}, {}
    # Confirmation remains sealed until the validation result is inspected.
    roles = ("train", "validation")
    for role in roles:
        packs[role] = load_pack(args.fold / f"{role}_source.npz", expected_side="source")[0]
        sidecars[role] = sidecar(args.fold / f"{role}_source_clones.csv")
    target_rows = {role: sidecar(args.fold / f"{role}_target_clones.csv") for role in roles}
    target_programs_by_role = {}
    clone_y_by_role = {}
    program_names = None
    for role, rows_target in target_rows.items():
        target_programs_by_role[role], names = target_programs(payload_308, rows_target, program_defs, symbols)
        if program_names is None:
            program_names = names
        elif names != program_names:
            raise ValueError("Program definitions differ across target roles")
        clone_y_by_role[role] = clone_targets(rows_target, target_programs_by_role[role])
    y = {role: outcome_for_source(sidecars[role], clone_y_by_role[role], len(program_names)) for role in sidecars}

    # Reconstruct gene-level Day0 U/S only for the registered source cells.
    gene_us = {}
    source_runs = sorted({parse_source_id(row["cell_id"])[0] for role in roles for row in sidecars[role]})
    source_payloads = {run: load_usa(args.root / f"quant_merlin{run[-3:]}" / "af_quant") for run in source_runs}
    for role in roles:
        by_run = {}
        for i, row in enumerate(sidecars[role]):
            by_run.setdefault(parse_source_id(row["cell_id"])[0], []).append((i, row))
        out = np.zeros((len(sidecars[role]), 2 * len(gene_index)), dtype="float32")
        for run, items in by_run.items():
            rows = [row for _, row in items]
            values = raw_gene_us(source_payloads[run], rows, gene_index)
            for j, (i, _) in enumerate(items): out[i] = values[j]
        gene_us[role] = out

    config = FullConfig(state_dim=50, representation_dim=64, esm_dim=2560, gfg_genes=len(gene_index),
                        gfg_hidden=(256, 512, 512, 256), gfg_gene_dim=8, gfg_codes=32,
                        dynamics_backend="gfg", joint_dynamics=False, use_intrinsic=False,
                        use_gate=False, use_noise=False, max_experts=8, initial_active=1)
    model = FullVeloRoute(config).to(torch.device(args.device))
    model.dynamics.core.load_pretrained(CHECKPOINT, expected_sha256=CHECKPOINT_SHA256)
    model.eval()
    device = torch.device(args.device)
    train_ids = packs["train"]["cell_ids"].tolist()
    model.dynamics.prepare(torch.as_tensor(gene_us["train"], device=device),
                           torch.as_tensor(transform.components, dtype=torch.float32, device=device),
                           cell_ids=train_ids, scope=FitScope(frozenset(train_ids)))

    gfg = {}
    with torch.no_grad():
        for role in roles:
            vals = torch.as_tensor(gene_us[role], device=device)
            z = torch.as_tensor(packs[role]["z"], dtype=torch.float32, device=device)
            chunks = []
            for start in range(0, len(vals), 256): chunks.append(model.dynamics(z[start:start + 256], vals[start:start + 256])[1].cpu().numpy())
            gfg[role] = np.concatenate(chunks).astype("float32")
    rng = np.random.default_rng(args.seed)
    shuffled = {role: values.copy() for role, values in gene_us.items()}
    for role, values in shuffled.items():
        for run in source_runs:
            ix = np.asarray([parse_source_id(row["cell_id"])[0] == run for row in sidecars[role]])
            if ix.sum() > 1:
                values[ix, :len(gene_index)] = values[rng.permutation(np.flatnonzero(ix)), :len(gene_index)]
    gfg_shuffled = {}
    with torch.no_grad():
        for role in roles:
            vals = torch.as_tensor(shuffled[role], device=device)
            z = torch.as_tensor(packs[role]["z"], dtype=torch.float32, device=device)
            chunks = [model.dynamics(z[s:s + 256], vals[s:s + 256])[1].cpu().numpy() for s in range(0, len(vals), 256)]
            gfg_shuffled[role] = np.concatenate(chunks).astype("float32")

    rows = []
    for eval_role in ("validation",):
        x_train = packs["train"]["z"]
        x_eval = packs[eval_role]["z"]
        feature_sets = {
            "z_only": (x_train, x_eval),
            "z_plus_simple_velocity": (np.c_[x_train, packs["train"]["velocity"]], np.c_[x_eval, packs[eval_role]["velocity"]]),
            "z_plus_gfg_velocity": (np.c_[x_train, gfg["train"]], np.c_[x_eval, gfg[eval_role]]),
            "z_plus_gfg_shuffled_U": (np.c_[x_train, gfg_shuffled["train"]], np.c_[x_eval, gfg_shuffled[eval_role]]),
        }
        for name, (xt, xe) in feature_sets.items():
            _, metric = fit_eval(xt, y["train"], sidecars["train"], xe, y[eval_role], sidecars[eval_role])
            rows.append({"eval_role": eval_role, "arm": name, **metric})
    save_csv(args.output / "metrics.csv", rows)
    summary = {"status": "MERLIN_DAY0_GFG_FATE_PROBE_COMPLETE", "formal_claim": False,
               "checkpoint_sha256": CHECKPOINT_SHA256, "programs": program_names,
               "train_cells": len(sidecars["train"]), "validation_cells": len(sidecars["validation"]),
               "confirmation_sealed": True, "metrics": rows,
               "target_programs_expression_derived": True,
               "interpretation_rule": "Validation-only screening: GFG is a candidate only if it beats z-only and shuffled-U; confirmation remains sealed."}
    save_json(args.output / "summary.json", summary)
    (args.output / "RESULTS.md").write_text("# MeRLin Day0 GFG fate probe\n\n" + json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
