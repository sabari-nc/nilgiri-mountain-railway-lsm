#!/usr/bin/env python3
"""
Trains XGBoost for footprint-based landslide susceptibility modelling of the
Nilgiri Mountain Railway corridor, on the same locked predictor set and spatial
block partition used by train_ml_models.py.

Raw predicted probabilities are used for evaluation and mapping.

Required input files (in DATA_DIR):
    train_fp_final.csv
    val_fp_final.csv
    test_fp_final.csv
    predictors_used_fp_final.txt

Output files (models and statistics are written under DATA_DIR/model_runs/xgboost/):
    models/xgb_model.json                       Trained booster
    statistics/xgb_model_metrics.json           Test metrics
    statistics/xgb_feature_importance_gain.csv  Gain importance
    statistics/permutation_importance_xgb.csv   Test permutation importance
    DATA_DIR/xgb_test_predictions.csv           Test predictions (restricted)

Requirements:
    pip install xgboost scikit-learn numpy pandas

Run:
    python train_xgb_model.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

# Input configuration
# Set NMR_DATA_DIR to the train/val/test CSVs and predictors_used_fp_final.txt.
# These data are restricted; see README.md for the data restrictions.
DATA_DIR = Path(os.environ.get("NMR_DATA_DIR", "/path/to/your/data")).expanduser()
# ──────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = DATA_DIR / "model_runs" / "xgboost"
MODELS_DIR = RUN_DIR / "models"
STATS_DIR = RUN_DIR / "statistics"

RANDOM_SEED = 42
PERM_REPEATS = 10

# A stratified 25% of the training partition is held out for early stopping,
# leaving the validation partition free for threshold selection.
EARLY_STOP_FRAC = 0.25
EARLY_STOP_ROUNDS = 50
NUM_BOOST_ROUND = 4000

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc", "aucpr"],  # early stopping uses the last
    "eta": 0.05,
    "max_depth": 3,
    "min_child_weight": 5.0,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "reg_alpha": 0.5,
    "gamma": 0.0,
    "tree_method": "hist",
    "seed": RANDOM_SEED,
}


# ── HELPERS ──


def load_tables(data_dir):
    """Load the train, validation, and test partitions."""
    return (
        pd.read_csv(data_dir / "train_fp_final.csv"),
        pd.read_csv(data_dir / "val_fp_final.csv"),
        pd.read_csv(data_dir / "test_fp_final.csv"),
    )


def label_column(df):
    """Return the binary label column name ('label' or 'y')."""
    for want in ("label", "y"):
        for col in df.columns:
            if col.lower() == want:
                return col
    raise ValueError(f"No label column. Columns: {list(df.columns)}")


def locked_predictors(data_dir, tables):
    """Read the predictor list and check it against every partition."""
    path = data_dir / "predictors_used_fp_final.txt"
    if not path.exists():
        raise FileNotFoundError(f"Predictor list not found: {path}")

    preds = [p.strip() for p in path.read_text().splitlines() if p.strip()]
    preds = list(dict.fromkeys(preds))

    for name, df in tables.items():
        missing = [p for p in preds if p not in df.columns]
        if missing:
            raise ValueError(f"Predictors missing in {name}: {missing}")

    # Aspect must be circular-encoded; the raw azimuth is discontinuous at 0/360.
    if not {"aspect_sin", "aspect_cos"} <= set(preds):
        raise ValueError("aspect_sin and aspect_cos must be in the predictor list.")
    if "F2_aspect_deg" in preds:
        raise ValueError(
            "Raw aspect (F2_aspect_deg) must not be in the predictor list."
        )

    print(f"Predictors locked: {len(preds)}")
    return preds


def to_numeric(X):
    """Cast boolean and 'True'/'False' LULC indicator columns to int8."""
    X = X.copy()
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(np.int8)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            vals = set(X[c].astype(str).str.strip().str.lower().unique())
            if vals <= {"true", "false"}:
                X[c] = (X[c].astype(str).str.strip().str.lower() == "true").astype(
                    np.int8
                )

    bad = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if bad:
        raise ValueError(f"Non-numeric predictors: {bad}")
    return X


def youden_threshold(y, p, grid=199):
    """
    Threshold maximising Youden's J (TPR - FPR) on the validation partition.

    BLR, SVM, and RF use F1-maximisation. XGBoost uses Youden's J because
    F1-maximisation on the positive-dominant validation partition selects a
    degenerate low threshold. J weights sensitivity and specificity equally and
    is independent of class prevalence (Youden 1950).
    """
    best_thr, best_j = 0.5, -np.inf
    for t in np.linspace(0.01, 0.99, grid):
        tn, fp, fn, tp = confusion_matrix(
            y, (p >= t).astype(int), labels=[0, 1]
        ).ravel()
        j = tp / max(tp + fn, 1) - fp / max(fp + tn, 1)
        if j > best_j:
            best_j, best_thr = j, t
    return float(best_thr)


def evaluate(y, p, thr, name):
    """Test metrics at a fixed threshold."""
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)

    recall = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)

    m = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "threshold": float(thr),
        "balanced_acc": float(0.5 * (recall + spec)),
        "f1": float(2 * prec * recall / max(prec + recall, 1e-12)),
        "precision": float(prec),
        "recall": float(recall),
        "specificity": float(spec),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

    print(f"\n{name} — test set")
    for k, v in m.items():
        print(f"  {k:>13}: {v:.4f}" if isinstance(v, float) else f"  {k:>13}: {v}")
    return m


def permutation_importance(predict, X, y, features, repeats=PERM_REPEATS):
    """Decrease in test ROC-AUC when each predictor is permuted."""
    rng = np.random.default_rng(RANDOM_SEED)
    base = roc_auc_score(y, predict(X))

    rows = []
    for j, feat in enumerate(features):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - roc_auc_score(y, predict(Xp)))
        rows.append(
            {
                "feature": feat,
                "auc_decrease_mean": float(np.mean(drops)),
                "auc_decrease_std": float(np.std(drops)),
            }
        )

    return pd.DataFrame(rows).sort_values("auc_decrease_mean", ascending=False)


# ── MAIN ──


def main():
    np.random.seed(RANDOM_SEED)

    if not DATA_DIR.exists() or str(DATA_DIR) == "/path/to/your/data":
        raise SystemExit(
            "Set NMR_DATA_DIR to the folder containing the restricted inputs."
        )

    if DATA_DIR.resolve().is_relative_to(ROOT):
        raise SystemExit(
            "Keep the restricted inputs and training outputs outside this repository."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ──
    print("Loading data...")
    Ttr, Tva, Tte = load_tables(DATA_DIR)

    lab = label_column(Ttr)
    ytr_all = (Ttr[lab].values.astype(int) > 0).astype(int)
    yva = (Tva[lab].values.astype(int) > 0).astype(int)
    yte = (Tte[lab].values.astype(int) > 0).astype(int)

    features = locked_predictors(DATA_DIR, {"train": Ttr, "val": Tva, "test": Tte})

    Xtr_all = to_numeric(Ttr[features])
    Xva = to_numeric(Tva[features])
    Xte = to_numeric(Tte[features])

    for name, y in [("train", ytr_all), ("val", yva), ("test", yte)]:
        pos, neg = int(y.sum()), int((y == 0).sum())
        print(
            f"  {name:<5} {pos:>5} positive  {neg:>5} negative  "
            f"({100 * pos / (pos + neg):.1f}% positive)"
        )

    # ── Early-stopping holdout, taken from the training partition ──
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=EARLY_STOP_FRAC, random_state=RANDOM_SEED
    )
    idx_inner, idx_stop = next(split.split(Xtr_all, ytr_all))

    Xtr, ytr = Xtr_all.iloc[idx_inner], ytr_all[idx_inner]
    Xstop, ystop = Xtr_all.iloc[idx_stop], ytr_all[idx_stop]

    # Equalises the total gradient contribution of the two classes.
    scale_pos_weight = float((ytr == 0).sum() / max(ytr.sum(), 1))
    print(f"\n  inner train {len(Xtr)}   early-stop holdout {len(Xstop)}")
    print(f"  scale_pos_weight {scale_pos_weight:.4f}")

    # ── Train ──
    print("\nTraining XGBoost...")
    params = dict(PARAMS, scale_pos_weight=scale_pos_weight)

    dtrain = xgb.DMatrix(Xtr, label=ytr, feature_names=features)
    dstop = xgb.DMatrix(Xstop, label=ystop, feature_names=features)
    dval = xgb.DMatrix(Xva, label=yva, feature_names=features)
    dtest = xgb.DMatrix(Xte, label=yte, feature_names=features)

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dstop, "early_stop")],
        early_stopping_rounds=EARLY_STOP_ROUNDS,
        maximize=True,
        verbose_eval=False,
    )

    best_iter = int(booster.best_iteration)
    rounds = (0, best_iter + 1)
    print(f"  best iteration {best_iter}")

    p_va = booster.predict(dval, iteration_range=rounds)
    p_te = booster.predict(dtest, iteration_range=rounds)

    # ── Evaluate ──
    thr = youden_threshold(yva, p_va)
    metrics = evaluate(yte, p_te, thr, "XGBoost")
    metrics["threshold_rule"] = "Youden's J on validation"
    metrics["pr_auc_method"] = "scikit-learn average_precision_score"
    metrics["best_iteration"] = best_iter
    metrics["scale_pos_weight"] = scale_pos_weight

    # ── Importance ──
    gain = booster.get_score(importance_type="gain")
    gain_imp = pd.DataFrame(
        {
            "feature": features,
            "gain_importance": [float(gain.get(f, 0.0)) for f in features],
        }
    ).sort_values("gain_importance", ascending=False)

    print("\nXGBoost gain importance")
    print(gain_imp.to_string(index=False))

    def predict_matrix(X):
        return booster.predict(
            xgb.DMatrix(X, feature_names=features), iteration_range=rounds
        )

    perm = permutation_importance(
        predict_matrix, Xte.values.astype(float), yte, features
    )

    # ── Save ──
    booster.save_model(str(MODELS_DIR / "xgb_model.json"))

    (STATS_DIR / "xgb_model_metrics.json").write_text(
        json.dumps({"XGB": metrics, "predictors": features, "params": params}, indent=2)
        + "\n"
    )
    gain_imp.to_csv(STATS_DIR / "xgb_feature_importance_gain.csv", index=False)

    perm.assign(model="XGB")[
        ["model", "feature", "auc_decrease_mean", "auc_decrease_std"]
    ].to_csv(STATS_DIR / "permutation_importance_xgb.csv", index=False)

    # Test predictions carry inventory labels, so they stay with the restricted data.
    pd.DataFrame(
        {
            "y_true": yte,
            "p_hat": p_te,
            "y_pred": (p_te >= thr).astype(int),
        }
    ).to_csv(DATA_DIR / "xgb_test_predictions.csv", index=False)

    print(f"\nModel      -> {MODELS_DIR / 'xgb_model.json'}")
    print(f"Statistics -> {STATS_DIR}")
    print(f"Predictions -> {DATA_DIR / 'xgb_test_predictions.csv'}")


if __name__ == "__main__":
    main()
