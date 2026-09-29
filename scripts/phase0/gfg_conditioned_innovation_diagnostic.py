"""Condition-aware GFG velocity residual diagnostic.

Compare source-only predictors z -> v and (z, frozen condition embedding) -> v.
Predictors are fitted on train source only; validation target is read only for
the final local-direction diagnostic.  This is exploratory and does not claim
conditional mutual information is zero.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from veloroute.artifacts import save_csv, save_json
from veloroute.latent import load_pack
from gfg_innovation_diagnostic import direction_cos, gfg_velocity


FOLD = Path("outputs/veloroute_real_pipeline_20260912_v2/fold")
GENE_DIR = Path("outputs/veloroute_gfg_inputs_20260914")
CONDITIONS = Path("data/renge/conditions/esm2_3b_v1/conditions.npz")


def condition_features(labels: np.ndarray) -> np.ndarray:
    with np.load(CONDITIONS, allow_pickle=False) as payload:
        names = payload["conditions"].astype(str)
        embeddings = payload["embeddings"].astype("float32")
    index = {name: i for i, name in enumerate(names)}
    try:
        return np.stack([embeddings[index[str(label)]] for label in labels]).astype("float32")
    except KeyError as exc:
        raise ValueError(f"Condition is absent from frozen embedding table: {exc}") from exc


def crossfit_predict(x: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float]:
    prediction = np.zeros_like(y)
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_ix, test_ix in kfold.split(x):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(x[train_ix], y[train_ix])
        prediction[test_ix] = model.predict(x[test_ix])
    residual = float(np.square(y - prediction).sum())
    variance = float(np.square(y - y.mean(axis=0, keepdims=True)).sum())
    return prediction, float(1.0 - residual / max(variance, 1e-12))


def validation_predict(x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(x_train, y_train)
    return model.predict(x_validation).astype("float32")


def main() -> None:
    output = Path("outputs/velocity_gfg_conditioned_innovation_20260929")
    output.mkdir(parents=True, exist_ok=False)
    data = gfg_velocity(str(FOLD), str(GENE_DIR), str(CONDITIONS), device="cpu")
    train_source, train_velocity = data["train"]
    validation_source, validation_velocity = data["validation"]
    train_condition = condition_features(train_source["conditions"])
    validation_condition = condition_features(validation_source["conditions"])
    feature_sets = {
        "z": (train_source["z"], validation_source["z"]),
        "z_plus_condition_embedding": (
            np.c_[train_source["z"], train_condition],
            np.c_[validation_source["z"], validation_condition],
        ),
    }
    predictions = {}
    rows = []
    for name, (x_train, x_validation) in feature_sets.items():
        train_prediction, train_r2 = crossfit_predict(x_train, train_velocity)
        validation_prediction = validation_predict(x_train, train_velocity, x_validation)
        predictions[name] = (train_prediction, validation_prediction)
        rows.append({"role": "train", "predictor": name, "r2": train_r2,
                     "velocity_rms": float(np.sqrt(np.square(train_velocity).mean())),
                     "residual_rms": float(np.sqrt(np.square(train_velocity - train_prediction).mean()))})
        rows.append({"role": "validation", "predictor": name, "r2": float(1.0 - np.square(validation_velocity - validation_prediction).sum() /
                                                                       max(np.square(validation_velocity - validation_velocity.mean(axis=0)).sum(), 1e-12)),
                     "velocity_rms": float(np.sqrt(np.square(validation_velocity).mean())),
                     "residual_rms": float(np.sqrt(np.square(validation_velocity - validation_prediction).mean()))})

    # Target packs are evaluated only after all source-side predictors are fit.
    target_rows = []
    for role, source, velocity, target in (
        ("train", train_source, train_velocity, load_pack(FOLD / "train_target.npz", expected_side="target")[0]),
        ("validation", validation_source, validation_velocity, load_pack(FOLD / "validation_target.npz", expected_side="target")[0]),
    ):
        for predictor, (_, prediction) in predictions.items():
            predicted = predictions[predictor][0] if role == "train" else prediction
            residual = velocity - predicted
            for condition in sorted(set(source["conditions"])):
                source_ix = source["conditions"] == condition
                target_ix = target["conditions"] == condition
                if source_ix.sum() < 20 or target_ix.sum() < 20:
                    continue
                target_rows.append({
                    "role": role, "predictor": predictor, "condition": str(condition),
                    "source_cells": int(source_ix.sum()), "target_cells": int(target_ix.sum()),
                    "cos_velocity": float(direction_cos(source["z"][source_ix], velocity[source_ix], target["z"][target_ix]).mean()),
                    "cos_predicted": float(direction_cos(source["z"][source_ix], predicted[source_ix], target["z"][target_ix]).mean()),
                    "cos_residual": float(direction_cos(source["z"][source_ix], residual[source_ix], target["z"][target_ix]).mean()),
                })

    save_csv(output / "predictor_metrics.csv", rows)
    save_csv(output / "condition_direction_metrics.csv", target_rows)
    summary = {
        "status": "GFG_CONDITIONED_INNOVATION_COMPLETE",
        "task": "RENGE_day4_to_day5_exploratory",
        "predictor_fit_scope": "train_source_only",
        "target_used_for_fit": False,
        "confirmation_opened": False,
        "condition_input": "frozen_ESM2_condition_embeddings",
        "predictors": rows,
        "direction_metrics": target_rows,
        "interpretation": "Compare z and z+condition residuals; neither predictor R2 nor local cosine is a fate claim.",
    }
    save_json(output / "summary.json", summary)
    (output / "RESULTS.md").write_text("# Condition-aware GFG innovation diagnostic\n\n" + json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
