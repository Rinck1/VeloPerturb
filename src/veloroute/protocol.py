"""Formal real-data experiments remain fail-closed after the engineering waiver."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .artifacts import object_hash, sha256, utc_now

REQUIRED_CHECKS = ("su_qc_passed", "guide_calls_validated", "fit_scope_audited",
                   "velocity_transform_frozen", "confirmation_conditions_reserved",
                   "probe_budget_fixed", "permutation_protocol_fixed")


def protocol_errors(config):
    errors = []
    if config.get("task") != "ER-short":
        errors.append("First formal diagnostic task must be ER-short")
    if config.get("metric") != "energy_distance_v_statistic":
        errors.append("Exact metric implementation must be preregistered")
    if len(set(config.get("seeds", []))) < 3:
        errors.append("Need at least three distinct seeds")
    for field in ("minimum_useful_gain", "noninferiority_margin"):
        value = config.get(field)
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            errors.append(f"{field} needs a justified positive finite value")
    if not config.get("effect_margin_rationale"):
        errors.append("Missing pre-result effect-margin rationale")
    for name in REQUIRED_CHECKS:
        if config.get("readiness", {}).get(name) is not True:
            errors.append(f"Not ready: {name}")
    for name in ("split_manifest", "data_provenance", "velocity_provenance", "probe_config"):
        value = config.get(name)
        if not isinstance(value, str) or not Path(value).is_file():
            errors.append(f"Missing evidence file: {name}")
    return errors


def freeze_protocol(config):
    errors = protocol_errors(config)
    if errors:
        raise ValueError("Protocol cannot be frozen:\n" + "\n".join(errors))
    inputs = {name: {"path": str(Path(config[name]).resolve()), "sha256": sha256(config[name])}
              for name in ("split_manifest", "data_provenance", "velocity_provenance", "probe_config")}
    payload = {"config": config, "inputs": inputs, "frozen_utc": utc_now(), "kind": "development"}
    return {**payload, "freeze_hash": object_hash(payload)}


def assert_real_training_allowed(frozen, decision):
    payload = {key: value for key, value in frozen.items() if key != "freeze_hash"}
    if frozen.get("freeze_hash") != object_hash(payload):
        raise ValueError("Protocol freeze hash mismatch")
    for artifact in frozen.get("inputs", {}).values():
        if not Path(artifact["path"]).is_file() or sha256(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Frozen input changed or missing")
    if protocol_errors(frozen.get("config", {})):
        raise ValueError("Incomplete formal protocol")
    if (decision.get("status") != "GO" or decision.get("kind") != "development"
            or decision.get("stage") != "G1" or decision.get("protocol_hash") != frozen["freeze_hash"]):
        raise ValueError("Real-data model training needs G1-GO on this protocol; synthetic success is insufficient")
    return True

