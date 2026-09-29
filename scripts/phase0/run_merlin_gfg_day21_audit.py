"""Bounded GPU audit of GFG velocity against MeRLin Day21 clone programs.

This is an exploratory same-time diagnostic, not the longitudinal endpoint test.
It reads only the existing Day21 USA matrix and clone assignments, computes a
training-only PCA and GFG decoder-JVP velocity, and compares clone residual
association against row-shuffled U.  Outputs are provenance-bound and are not
used as a formal multimodality claim.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from veloroute.artifacts import sha256
from veloroute.preprocess import load_usa


CHECKPOINT = Path("/data/yuchang/GFG/results/mousebrain_graphbatch_directed_v1_seed0/final.pth")
CHECKPOINT_SHA256 = "a9d9fa39063312230e4ba64309e9264ba3c2cb7ac549f04544b366df57c1340b"
ROOT = Path("/data/yuchang/veloroute_ucheck_20260915")
USA = ROOT / "quant_merlin308" / "af_quant"
CLONES = ROOT / "clones" / "SRR33960308" / "cell_clone_assignments.csv"
GTF = Path("/data/yuchang/veloroute_kang_20260914/relocated/gencode32/gencode.v32.primary_assembly.annotation.gtf.gz")
PROGRAMS = ROOT / "merlin_programs.json"


def read_clones(path: Path):
    result = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            result[row["cell_barcode"]] = row["clone_barcode"]
    return result


def read_gene_symbols(path: Path):
    result = {}
    with gzip.open(path, "rt", errors="ignore") as handle:
        for line in handle:
            if line.startswith("#") or "\tgene\t" not in line:
                continue
            fields = line.rstrip("\n").split("\t")
            values = {}
            for item in fields[8].split(";"):
                item = item.strip()
                if not item or " " not in item:
                    continue
                key, value = item.split(" ", 1)
                values[key] = value.strip('"')
            if values.get("gene_id") and values.get("gene_name"):
                result.setdefault(values["gene_id"].split(".")[0], values["gene_name"])
    return result


def residual_clone_score(values, groups, *, min_clone=3):
    """Between-clone residual energy / total residual energy."""
    unique, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    keep = counts[inverse] >= min_clone
    if keep.sum() < 3:
        return dict(score=None, n_cells=int(keep.sum()), n_clones=0)
    x = values[keep]
    gi = inverse[keep]
    centered = x - x.mean(0, keepdims=True)
    means = np.zeros((len(unique), x.shape[1]), dtype=np.float64)
    np.add.at(means, gi, x)
    denom = np.bincount(gi, minlength=len(unique)).clip(min=1)[:, None]
    means /= denom
    between = means[gi] - x.mean(0, keepdims=True)
    total = float(np.square(centered).sum())
    return dict(score=float(np.square(between).sum() / max(total, 1e-12)),
                n_cells=int(keep.sum()), n_clones=int(len(np.unique(gi))))


def clone_velocity_rows(v, z, groups, programs):
    rows = []
    design = np.c_[np.ones(len(z)), z]
    residual = v - design @ np.linalg.lstsq(design, v, rcond=None)[0]
    for name, values in [("velocity", v), ("velocity_residual", residual), ("z", z), ("program", programs)]:
        s = residual_clone_score(values, groups)
        rows.append(dict(feature=name, **s))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-cells", type=int, default=12000)
    ap.add_argument("--genes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("GFG checkpoint hash mismatch")
    barcodes, genes, matrices, meta = load_usa(USA)
    spliced, unspliced, ambiguous = [m.tocsr() for m in matrices]
    clone_map = read_clones(CLONES)
    keep = np.array([b in clone_map for b in barcodes])
    depth = np.asarray(spliced.sum(1)).ravel() + np.asarray(unspliced.sum(1)).ravel()
    keep &= depth >= 500
    indices = np.flatnonzero(keep)
    rng = np.random.default_rng(args.seed)
    if len(indices) > args.max_cells:
        indices = np.sort(rng.choice(indices, args.max_cells, replace=False))
    barcodes = np.asarray(barcodes)[indices]
    s = spliced[indices]
    u = unspliced[indices]
    clone_ids = np.array([clone_map[b] for b in barcodes])
    sf = s.multiply((1e4 / np.maximum(np.asarray(s.sum(1)).ravel(), 1.0))[:, None]).tocsr()
    sf.data = np.log1p(sf.data)
    variances = np.asarray(sf.multiply(sf).mean(0) - np.square(sf.mean(0))).ravel()
    genes_n = min(args.genes, len(variances))
    selected = np.argsort(variances)[-genes_n:]
    from sklearn.decomposition import PCA
    pca = PCA(50, svd_solver="randomized", random_state=args.seed)
    x_selected = sf[:, selected].toarray().astype("float32")
    z = pca.fit_transform(x_selected).astype("float32")
    gene_us = np.concatenate([u[:, selected].toarray(), s[:, selected].toarray()], axis=1).astype("float32")
    symbols = read_gene_symbols(GTF)
    selected_symbols = [symbols.get(str(genes[j]).split(".")[0], "") for j in selected]
    program_defs = json.loads(PROGRAMS.read_text())["signatures"]
    program_names = list(program_defs)
    programs, program_hits = [], {}
    for name in program_names:
        wanted = {str(x).split(" ", 1)[0].replace("*", "") for x in program_defs[name]}
        hit = [i for i, symbol in enumerate(selected_symbols) if symbol in wanted]
        program_hits[name] = len(hit)
        programs.append(x_selected[:, hit].mean(1) if hit else np.zeros(len(x_selected), dtype="float32"))
    programs = np.stack(programs, axis=1).astype("float32")

    import torch
    from veloroute.contracts import FitScope
    from veloroute.full_model import FullConfig, FullVeloRoute
    device = torch.device(args.device)
    cfg = FullConfig(state_dim=50, representation_dim=64, esm_dim=2560,
                     gfg_genes=genes_n, gfg_hidden=(256, 512, 512, 256),
                     gfg_gene_dim=8, gfg_codes=32, dynamics_backend="gfg",
                     joint_dynamics=False, use_intrinsic=False, use_gate=False,
                     use_noise=False, max_experts=8, initial_active=1)
    model = FullVeloRoute(cfg).to(device)
    model.dynamics.core.load_pretrained(CHECKPOINT, expected_sha256=CHECKPOINT_SHA256)
    model.eval()
    ids = [str(x) for x in barcodes]
    model.dynamics.prepare(torch.as_tensor(gene_us, device=device),
                           torch.as_tensor(pca.components_, dtype=torch.float32, device=device),
                           cell_ids=ids, scope=FitScope(frozenset(ids)))
    vs = []; rho = []; native = []
    with torch.no_grad():
        for start in range(0, len(z), 256):
            zz = torch.as_tensor(z[start:start+256], device=device)
            gu = torch.as_tensor(gene_us[start:start+256], device=device)
            rr, vv, ro = model.dynamics(zz, gu)
            vs.append(vv.cpu().numpy()); rho.append(ro.cpu().numpy()); native.append(float(model.dynamics.last_native_loss))
    v = np.concatenate(vs); rho = np.concatenate(rho)
    shuffled = gene_us.copy(); shuffled[:, :genes_n] = shuffled[rng.permutation(len(shuffled)), :genes_n]
    sv = []
    with torch.no_grad():
        for start in range(0, len(z), 256):
            zz = torch.as_tensor(z[start:start+256], device=device)
            gu = torch.as_tensor(shuffled[start:start+256], device=device)
            sv.append(model.dynamics(zz, gu)[1].cpu().numpy())
    sv = np.concatenate(sv)
    rows = clone_velocity_rows(v, z, clone_ids, programs)
    rows_shuf = clone_velocity_rows(sv, z, clone_ids, programs)
    for row in rows_shuf: row["feature"] = "shuffled_" + row["feature"]
    rows.extend(rows_shuf)
    design = np.c_[np.ones(len(z)), z]
    def residualize(values):
        return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]
    def velocity_program_similarity(values):
        a, b = residualize(programs), residualize(values)
        a /= np.maximum(np.linalg.norm(a, axis=0, keepdims=True), 1e-8)
        b /= np.maximum(np.linalg.norm(b, axis=0, keepdims=True), 1e-8)
        return np.abs(a.T @ b).max(axis=1).tolist()
    real_similarity = velocity_program_similarity(v)
    shuffled_similarity = velocity_program_similarity(sv)
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out / "velocity.npz", barcodes=barcodes, clone_ids=clone_ids,
                        z=z, velocity=v, shuffled_velocity=sv, rho=rho, program_scores=programs)
    summary = dict(status="MERLIN_DAY21_GFG_AUDIT_COMPLETE", scope="same_time_exploratory",
                   cells=int(len(z)), clone_count=int(len(np.unique(clone_ids))), genes=int(genes_n),
                   checkpoint_sha256=CHECKPOINT_SHA256, median_rho=float(np.median(rho)),
                   native_loss_mean=float(np.mean(native)), real=rows[:4], shuffled=rows[4:8],
                   program_names=program_names, program_hits=program_hits,
                   velocity_program_max_abs_corr=real_similarity,
                   shuffled_velocity_program_max_abs_corr=shuffled_similarity,
                   multimodality_claim=False,
                   note="Day21 clone association is not a future prediction; Day0 runs are still required.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "RESULTS.md").write_text("# MeRLin Day21 GFG audit\n\n" + json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
