"""Condition-level reservations independent of any response measurements."""
from __future__ import annotations

import hashlib

from .artifacts import read_csv, save_csv


def make_condition_split(guide_rows, *, seed, n_validation=4, n_confirmation=5):
    if not isinstance(seed, int) or n_validation < 1 or n_confirmation < 1:
        raise ValueError("Use an integer seed and nonempty validation/confirmation groups")
    guide_names = [r["guide_name"] for r in guide_rows]
    if len(guide_names) != len(set(guide_names)):
        raise ValueError("Duplicate guide names")
    targets = sorted({r["target_from_published_name"] for r in guide_rows if r["control_class"] == "TF"})
    if len(targets) <= n_validation + n_confirmation:
        raise ValueError("No training conditions left")
    ordered = sorted(targets, key=lambda target: hashlib.sha256(f"{seed}:{target}".encode()).hexdigest())
    test = set(ordered[:n_confirmation])
    validation = set(ordered[n_confirmation:n_confirmation+n_validation])
    rows = []
    for guide in sorted(guide_rows, key=lambda r: r["guide_name"]):
        target = guide["target_from_published_name"]
        if guide["control_class"] not in {"TF", "CTRL", "AAVS1"}:
            raise ValueError("Unknown control class")
        role = "control" if guide["control_class"] != "TF" else (
            "confirmation" if target in test else "validation" if target in validation else "train")
        rows.append({"guide_name": guide["guide_name"], "condition": target,
                     "control_class": guide["control_class"], "role": role,
                     "days": "2;3;4;5", "seed": seed,
                     "status": "reserved_condition_only_not_cell_assignment"})
    return rows


def attach_split(cells, condition_rows):
    """Reject uncalled labels. A largest-count guide is not a valid assignment."""
    mapping = {}
    for row in condition_rows:
        if row["condition"] in mapping and mapping[row["condition"]] != row["role"]:
            raise ValueError("One TF appears in multiple roles")
        mapping[row["condition"]] = row["role"]
    ids, rows = set(), []
    for cell in cells:
        if not cell.get("cell_id") or cell["cell_id"] in ids:
            raise ValueError("Missing or duplicate namespaced cell ID")
        ids.add(cell["cell_id"])
        if cell.get("label_status") not in {"validated", "published_validated"}:
            raise ValueError("Formal split requires validated condition calls")
        condition = cell.get("assigned_condition")
        if condition not in mapping:
            raise ValueError("Unrecognized called condition")
        rows.append({**cell, "split": mapping[condition]})
    return rows

