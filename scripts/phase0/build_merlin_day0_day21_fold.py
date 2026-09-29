"""Build a clone-held-out MeRLin Day0 -> Day21 source/target fold.

The fold is intentionally small and explicit: Day0 S/U is source input,
Day21 spliced expression is target output, and clone IDs are retained only in
sidecar metadata for evaluation.  No Day21 U/S/velocity is exported.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from veloroute.artifacts import save_csv, save_json, sha256
from veloroute.contracts import FitScope
from veloroute.latent import FrozenSplicingTransform, dense_counts, normalized_counts, save_pack
from veloroute.preprocess import load_usa


ROOT = Path("/data/yuchang/veloroute_ucheck_20260915")
DEFAULT_OUTPUT = Path("/data/yuchang/veloroute_merlin_day0_day21_fold_20260929")


def norm_barcode(value: str) -> str:
    return value.split("-", 1)[0]


def read_clone_map(path: Path, *, strict_single_barcode: bool = True) -> dict[str, str]:
    with path.open() as handle:
        rows = csv.DictReader(handle)
        result = {}
        for row in rows:
            # The extractor keeps dominant_fraction>=0.5 rows, including cells
            # with competing barcode hypotheses.  Longitudinal fate labels use
            # only unambiguous n_barcodes==1 assignments by default.
            if strict_single_barcode and int(row.get("n_barcodes", "1")) != 1:
                continue
            cell, clone = norm_barcode(row["cell_barcode"]), row["clone_barcode"]
            if cell and clone:
                result[cell] = clone
        return result


def load_subset(root: Path, run: str, clone_map: dict[str, str], keep_clones: set[str], minimum_total_umi: int):
    q = root / f"quant_merlin{run[-3:]}" / "af_quant"
    barcodes, genes, mats, _ = load_usa(q)
    normalized = np.asarray([norm_barcode(b) for b in barcodes], dtype=str)
    keep = np.array([b in clone_map and clone_map[b] in keep_clones for b in normalized], dtype=bool)
    ix = np.flatnonzero(keep)
    if not len(ix):
        raise ValueError(f"No cells for {run} in requested clone set")
    s, u = mats[0][ix].tocsr(), mats[1][ix].tocsr()
    valid = (np.asarray(s.sum(axis=1)).ravel() + np.asarray(u.sum(axis=1)).ravel()) >= minimum_total_umi
    valid &= np.asarray(s.sum(axis=1)).ravel() > 0
    s, u, ix = s[valid], u[valid], ix[valid]
    ids = np.asarray([f"day0:{run}:{b}" if run[-3:] in {"310", "311"} else f"day21:{run}:{b}" for b in normalized[ix]], dtype=str)
    clones = np.asarray([clone_map[b] for b in normalized[ix]], dtype=str)
    return genes, s, u, ids, clones


def choose_genes(s: sparse.csr_matrix, n_genes: int, target_sum: float = 10000.0) -> np.ndarray:
    totals = np.asarray(s.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("Zero-spliced source cell encountered")
    sf = s.multiply((target_sum / totals)[:, None]).tocsr().astype(np.float64)
    sf.data = np.log1p(sf.data)
    mean = np.asarray(sf.mean(axis=0)).ravel()
    second = np.asarray(sf.multiply(sf).mean(axis=0)).ravel()
    variance = np.maximum(second - mean * mean, 0.0)
    support = np.asarray((s > 0).sum(axis=0)).ravel()
    candidates = np.flatnonzero(support >= min(10, s.shape[0]))
    if len(candidates) < n_genes:
        raise ValueError(f"Only {len(candidates)} genes have sufficient support for {n_genes}")
    order = np.argsort(-variance[candidates], kind="stable")
    return candidates[order[:n_genes]]


def target_latent(transform: FrozenSplicingTransform, s: sparse.csr_matrix) -> np.ndarray:
    values = dense_counts(s)
    normalized, _ = normalized_counts(values, np.zeros_like(values), transform.target_sum)
    return ((np.log1p(normalized[:, transform.selected]) - transform.mean) @ transform.components.T).astype(np.float32)


def write_sidecar(path: Path, ids, clones, runs, role, side):
    rows = [{"cell_id": str(i), "clone_id": str(c), "run": str(r), "role": role, "side": side} for i, c, r in zip(ids, clones, runs)]
    save_csv(path, rows, fields=["cell_id", "clone_id", "run", "role", "side"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--genes", type=int, default=2000)
    ap.add_argument("--components", type=int, default=50)
    ap.add_argument("--day0-run", choices=("SRR33960310", "SRR33960311"), default="SRR33960310",
                     help="One sequencing run only; 310/311 are technical repeats of the same Day0 library.")
    ap.add_argument("--minimum-total-umi", type=int, default=100)
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    clone_maps = {run: read_clone_map(args.root / "clones" / run / "cell_clone_assignments.csv") for run in ("SRR33960308", args.day0_run)}
    raw_shared = set(clone_maps[args.day0_run].values()) & set(clone_maps["SRR33960308"].values())
    # Load QC-passing cells first; clone splitting happens only after both
    # sides are observed so clones with no cells surviving QC cannot enter a
    # nominal split and distort the independent-unit count.
    source_full = []
    source_genes = None
    for run in (args.day0_run,):
        genes, s, u, ids, clones = load_subset(args.root, run, clone_maps[run], raw_shared, args.minimum_total_umi)
        source_genes = genes if source_genes is None else source_genes
        if list(genes) != list(source_genes):
            raise ValueError("Day0 gene order mismatch")
        run_arr = np.repeat(run, len(ids))
        source_full.append((s, u, ids, clones, run_arr))
    genes, target_s, target_u, target_ids, target_clones = load_subset(args.root, "SRR33960308", clone_maps["SRR33960308"], raw_shared, args.minimum_total_umi)
    if list(genes) != list(source_genes):
        raise ValueError("Day0/Day21 gene order mismatch")
    target_run = np.repeat("SRR33960308", len(target_ids))
    effective_shared = set(target_clones.tolist())
    for _, _, _, clones, _ in source_full:
        effective_shared &= set(clones.tolist())
    shared = sorted(effective_shared)
    if len(shared) < 100:
        raise ValueError(f"Insufficient QC-passing shared clones: {len(shared)}")
    rng = np.random.default_rng(args.seed)
    shuffled = np.asarray(shared, dtype=str); rng.shuffle(shuffled)
    n = len(shuffled); n_val = max(1, int(round(n * 0.20))); n_conf = max(1, int(round(n * 0.20)))
    clone_splits = {
        "train": set(shuffled[: n - n_val - n_conf]),
        "validation": set(shuffled[n - n_val - n_conf : n - n_conf]),
        "confirmation": set(shuffled[n - n_conf :]),
    }
    if any(not clone_splits[k] for k in clone_splits):
        raise ValueError("Empty clone split")
    source_parts, target_parts = {}, {}
    for s, u, ids, clones, run_arr in source_full:
        for role, clone_set in clone_splits.items():
            mask = np.isin(clones, list(clone_set))
            source_parts.setdefault(role, []).append((s[mask], u[mask], ids[mask], clones[mask], run_arr[mask]))
    for role, clone_set in clone_splits.items():
        mask = np.isin(target_clones, list(clone_set))
        target_parts[role] = (target_s[mask], target_ids[mask], target_clones[mask], target_run[mask])
    del source_full, target_s, target_u
    gc.collect()

    train_s = sparse.vstack([x[0] for x in source_parts["train"]]).tocsr()
    train_u = sparse.vstack([x[1] for x in source_parts["train"]]).tocsr()
    train_ids = np.concatenate([x[2] for x in source_parts["train"]]).astype(str)
    selected = choose_genes(train_s, args.genes)
    selected_genes = np.asarray(source_genes, dtype=str)[selected]
    train_s, train_u = train_s[:, selected], train_u[:, selected]
    train_valid = np.asarray(train_s.sum(axis=1)).ravel() > 0
    train_s, train_u, train_ids = train_s[train_valid], train_u[train_valid], train_ids[train_valid]
    scope = FitScope(frozenset(train_ids.tolist()))
    transform = FrozenSplicingTransform.fit(
        train_s,
        train_u,
        gene_ids=selected_genes.tolist(),
        cell_ids=train_ids,
        scope=scope,
        n_genes=len(selected_genes),
        n_components=args.components,
        seed=args.seed,
    )
    transform_path = args.output / "transform.npz"
    transform.save(transform_path)
    transform_hash = sha256(transform_path)

    actual_source_cells, actual_target_cells = {}, {}
    for role in ("train", "validation", "confirmation"):
        src_s = sparse.vstack([x[0] for x in source_parts[role]]).tocsr()[:, selected]
        src_u = sparse.vstack([x[1] for x in source_parts[role]]).tocsr()[:, selected]
        src_ids = np.concatenate([x[2] for x in source_parts[role]]).astype(str)
        src_clones = np.concatenate([x[3] for x in source_parts[role]]).astype(str)
        src_runs = np.concatenate([x[4] for x in source_parts[role]]).astype(str)
        src_valid = np.asarray(src_s.sum(axis=1)).ravel() > 0
        src_s, src_u = src_s[src_valid], src_u[src_valid]
        src_ids, src_clones, src_runs = src_ids[src_valid], src_clones[src_valid], src_runs[src_valid]
        actual_source_cells[role] = int(len(src_ids))
        src_arrays = transform.transform(src_s, src_u, gene_ids=selected_genes.tolist())
        src_arrays.update(
            cell_ids=src_ids,
            conditions=np.repeat("MeRLin_BRAFi_MEKi", len(src_ids)).astype(str),
            depth=np.asarray(src_s.sum(axis=1)).ravel().astype(np.float32),
        )
        src_meta = {
            "side": "source", "role": role, "day": 0, "task": "MeRLin_Day0_to_Day21",
            "transform_hash": transform_hash, "fit_ids_hash": transform.fit_ids_hash,
            "future_target_read": False, "clone_split_hash": hashlib.sha256("|".join(sorted(clone_splits[role])).encode()).hexdigest(),
        }
        save_pack(args.output / f"{role}_source.npz", src_arrays, src_meta)
        write_sidecar(args.output / f"{role}_source_clones.csv", src_ids, src_clones, src_runs, role, "source")

        tgt_s = target_parts[role][0][:, selected]
        tgt_ids, tgt_clones, tgt_runs = target_parts[role][1:]
        tgt_valid = np.asarray(tgt_s.sum(axis=1)).ravel() > 0
        tgt_s = tgt_s[tgt_valid]
        tgt_ids, tgt_clones, tgt_runs = tgt_ids[tgt_valid], tgt_clones[tgt_valid], tgt_runs[tgt_valid]
        actual_target_cells[role] = int(len(tgt_ids))
        tgt_arrays = {
            "z": target_latent(transform, tgt_s),
            "cell_ids": tgt_ids.astype(str),
            "conditions": np.repeat("MeRLin_BRAFi_MEKi", len(tgt_ids)).astype(str),
        }
        tgt_meta = {
            "side": "target", "role": role, "day": 21, "task": "MeRLin_Day0_to_Day21",
            "transform_hash": transform_hash, "fit_ids_hash": transform.fit_ids_hash,
            "future_target_read": True, "clone_split_hash": hashlib.sha256("|".join(sorted(clone_splits[role])).encode()).hexdigest(),
        }
        save_pack(args.output / f"{role}_target.npz", tgt_arrays, tgt_meta)
        write_sidecar(args.output / f"{role}_target_clones.csv", tgt_ids, tgt_clones, tgt_runs, role, "target")

    save_csv(args.output / "clone_splits.csv", [{"clone_id": c, "role": role} for role, values in clone_splits.items() for c in sorted(values)], fields=["clone_id", "role"])
    summary = {
        "status": "MERLIN_DAY0_DAY21_FOLD_CREATED",
        "formal_modeling_approved": False,
        "day21_run": "SRR33960308",
        "day0_run": args.day0_run,
        "day0_311_not_merged": True,
        "clone_assignment_filter": "n_barcodes==1",
        "minimum_total_umi": args.minimum_total_umi,
        "day21_run_309_available": False,
        "shared_clones": len(shared),
        "clone_split_counts": {k: len(v) for k, v in clone_splits.items()},
        "source_cells": actual_source_cells,
        "target_cells": actual_target_cells,
        "selected_genes": int(len(selected_genes)),
        "components": int(args.components),
        "transform_hash": transform_hash,
        "velocity_estimator": transform.estimator,
        "gfg_not_run": True,
        "reason": "Engineering fold only; GFG velocity and router remain gated on longitudinal diagnostic results.",
    }
    save_json(args.output / "summary.json", summary)
    (args.output / "RESULTS.md").write_text("# MeRLin Day0–Day21 fold\n\n" + json.dumps(summary, indent=2) + "\n\nDay21 S-only target packs contain no future U/velocity. Clone IDs are sidecar evaluation metadata and are not model inputs.\n")


if __name__ == "__main__":
    main()
