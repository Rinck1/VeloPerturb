"""Audit the MeRLin in-vivo Day0 -> Day21 clone alignment.

This is a data-contract audit only.  It does not fit PCA, velocity, a model,
or open Day21 expression as a source-side feature.  The only question here is
whether the downloaded Day0 (SRR33960310/311) and Day21 (SRR33960308/309)
libraries share enough clone barcodes to justify a longitudinal fold.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from veloroute.preprocess import load_usa


DEFAULT_ROOT = Path("/data/yuchang/veloroute_ucheck_20260915")
DEFAULT_OUTPUT = Path("/data/yuchang/veloroute_merlin_day0_day21_alignment_20260929")


def norm_barcode(value: str) -> str:
    return value.split("-", 1)[0]


def read_clone_map(path: Path) -> dict[str, str]:
    with path.open() as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames is None or not {"cell_barcode", "clone_barcode"} <= set(rows.fieldnames):
            raise ValueError(f"Unexpected clone assignment header: {path}")
        result = {}
        for row in rows:
            cell = norm_barcode(row["cell_barcode"])
            clone = row["clone_barcode"].strip()
            if cell and clone:
                result[cell] = clone
    if len(result) == 0:
        raise ValueError(f"No clone assignments: {path}")
    return result


def file_record(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "exists": path.is_file(), "bytes": int(stat.st_size)}


def run_record(root: Path, run: str, role: str) -> tuple[dict, set[str], dict[str, int]]:
    suffix = run[-3:]
    quant = root / f"quant_merlin{suffix}" / "af_quant"
    clone_path = root / "clones" / run / "cell_clone_assignments.csv"
    record = {"run": run, "role": role, "quant_dir": str(quant), "clone_file": str(clone_path)}
    if not quant.is_dir():
        record.update({"available": False, "reason": "quantification_missing"})
        return record, set(), {}
    if not clone_path.is_file():
        record.update({"available": False, "reason": "clone_assignments_missing"})
        return record, set(), {}

    barcodes, genes, matrices, meta = load_usa(quant)
    clone_map = read_clone_map(clone_path)
    barcodes_norm = [norm_barcode(x) for x in barcodes]
    if len(set(barcodes_norm)) != len(barcodes_norm):
        raise ValueError(f"Barcode collision after suffix normalization in {run}")
    assigned = {b: clone_map[b] for b in barcodes_norm if b in clone_map}
    s, u, a = [m.tocsr() for m in matrices]
    s_total = np.asarray(s.sum(axis=1)).ravel().astype(float)
    u_total = np.asarray(u.sum(axis=1)).ravel().astype(float)
    a_total = np.asarray(a.sum(axis=1)).ravel().astype(float)
    total = s_total + u_total + a_total
    assigned_ix = np.array([b in clone_map for b in barcodes_norm], dtype=bool)
    clone_counts = Counter(assigned.values())
    record.update(
        {
            "available": True,
            "cells_quantified": int(len(barcodes_norm)),
            "genes": int(len(genes)),
            "assigned_cells": int(assigned_ix.sum()),
            "assigned_fraction": float(assigned_ix.mean()),
            "unique_clones_assigned": int(len(clone_counts)),
            "multi_cell_clones": int(sum(v >= 2 for v in clone_counts.values())),
            "median_spliced_umi": float(np.median(s_total)),
            "median_unspliced_umi": float(np.median(u_total)),
            "median_unspliced_fraction": float(np.median(u_total / np.maximum(total, 1.0))),
            "pooled_unspliced_fraction": float(u_total.sum() / max(total.sum(), 1.0)),
            "quant_meta": str(quant / "quant.json"),
            "inputs": [file_record(quant / "quant.json"), file_record(clone_path)],
        }
    )
    return record, set(clone_counts), clone_counts


def pair_record(a: str, b: str, clone_sets: dict[str, set[str]], counts: dict[str, dict[str, int]]) -> dict:
    x, y = clone_sets.get(a, set()), clone_sets.get(b, set())
    overlap = x & y
    per_x = counts.get(a, {})
    per_y = counts.get(b, {})
    return {
        "run_a": a,
        "run_b": b,
        "clones_a": int(len(x)),
        "clones_b": int(len(y)),
        "shared_clones": int(len(overlap)),
        "shared_fraction_of_a": float(len(overlap) / max(len(x), 1)),
        "shared_fraction_of_b": float(len(overlap) / max(len(y), 1)),
        "shared_day0_cells": int(sum(per_x.get(c, 0) for c in overlap)),
        "shared_day21_cells": int(sum(per_y.get(c, 0) for c in overlap)),
        "shared_clone_ids": sorted(overlap),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    specs = [
        ("SRR33960310", "day0_in_vivo"),
        ("SRR33960311", "day0_in_vivo"),
        ("SRR33960308", "day21_treated_in_vivo"),
        ("SRR33960309", "day21_treated_in_vivo"),
    ]
    records, clone_sets, clone_counts = [], {}, {}
    for run, role in specs:
        record, clones, counts = run_record(args.root, run, role)
        records.append(record)
        if record.get("available"):
            clone_sets[run] = clones
            clone_counts[run] = counts

    pairs = []
    for d0 in ("SRR33960310", "SRR33960311"):
        for d21 in ("SRR33960308", "SRR33960309"):
            if d0 in clone_sets and d21 in clone_sets:
                pairs.append(pair_record(d0, d21, clone_sets, clone_counts))
    if "SRR33960310" in clone_sets and "SRR33960311" in clone_sets:
        pairs.append(pair_record("SRR33960310", "SRR33960311", clone_sets, clone_counts))
    if "SRR33960308" in clone_sets and "SRR33960309" in clone_sets:
        pairs.append(pair_record("SRR33960308", "SRR33960309", clone_sets, clone_counts))

    d0 = [p for p in pairs if p["run_a"] in {"SRR33960310", "SRR33960311"} and p["run_b"] == "SRR33960308"]
    best_overlap = max((p["shared_clones"] for p in d0), default=0)
    available_d0 = sum(r.get("available", False) and r["role"] == "day0_in_vivo" for r in records)
    available_d21 = sum(r.get("available", False) and r["role"] == "day21_treated_in_vivo" for r in records)
    result = {
        "status": "ALIGNMENT_AUDIT_COMPLETE",
        "available_day0_runs": int(available_d0),
        "available_day21_runs": int(available_d21),
        "best_day0_to_day21_308_shared_clones": int(best_overlap),
        "longitudinal_candidate": bool(best_overlap >= 100),
        "merge_day0_replicates": False,
        "merge_day21_replicates": False,
        "formal_modeling_approved": False,
        "reason": "Build fold only after confirming run identity and replicate semantics; use 308 single replicate until 309 is quantified.",
        "runs": records,
        "pairs": pairs,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "runs.csv").open("w", newline="") as handle:
        fields = sorted({k for r in records for k in r if k != "inputs"})
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)
    with (args.output / "pairs.csv").open("w", newline="") as handle:
        fields = [k for k in pairs[0] if k != "shared_clone_ids"] if pairs else ["run_a", "run_b", "shared_clones"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(pairs)
    (args.output / "RESULTS.md").write_text(
        "# MeRLin Day0–Day21 alignment audit\n\n"
        + json.dumps({k: result[k] for k in ("status", "available_day0_runs", "available_day21_runs", "best_day0_to_day21_308_shared_clones", "longitudinal_candidate", "formal_modeling_approved")}, indent=2)
        + "\n\nThis is a clone-ID/data-contract audit, not a model result. Day21 expression was not used as a source feature.\n"
    )


if __name__ == "__main__":
    main()
