#!/usr/bin/env python3
"""
Trains binary logistic regression, support vector machine, and random forest
for footprint-based landslide susceptibility modelling of the Nilgiri Mountain
Railway corridor. This implementation differs from train_ml_models.m; see
../README.md. XGBoost is trained by train_xgb_model.py.

Required input files (in DATA_DIR):
    train_fp_final.csv
    val_fp_final.csv
    test_fp_final.csv
    predictors_used_fp_final.txt

Output files (models and statistics are written under DATA_DIR/model_runs/python/):
    models/scaler.pkl                      Standardisation statistics
    models/blr_model.pkl                   Trained BLR
    models/svm_model_cal.pkl               SVM and Platt calibrator
    models/rf_model.pkl                    Tuned RF
    statistics/python_model_metrics.json   Test metrics per model
    statistics/rf_feature_importance.csv   RF impurity and permutation importance
    statistics/permutation_importance.csv  Test permutation importance, all models
    DATA_DIR/test_predictions.csv          Test predictions (restricted)

Requirements:
    pip install scikit-learn numpy pandas joblib

Run:
    python train_ml_models.py
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Input configuration
# Set NMR_DATA_DIR to the train/val/test CSVs and predictors_used_fp_final.txt.
# These data are restricted; see README.md for the data restrictions.
DATA_DIR = Path(os.environ.get("NMR_DATA_DIR", "/path/to/your/data")).expanduser()
# ──────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = DATA_DIR / "model_runs" / "python"
MODELS_DIR = RUN_DIR / "models"
STATS_DIR = RUN_DIR / "statistics"

RANDOM_SEED = 42
N_TREES = 800
LEAF_GRID = [1, 2, 5, 10, 20, 50]
CV_FOLDS = 5
PERM_REPEATS = 10


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
    return X.values.astype(float)


def f1_threshold(y, p, grid=999):
    """Threshold maximising F1 on the validation partition."""
    best_thr, best_f1 = 0.5, -np.inf
    for t in np.linspace(0.001, 0.999, grid):
        f1 = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
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
        "pr_auc_method": "scikit-learn average_precision_score",
        "brier": float(brier_score_loss(y, p)),
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


def fit_platt(scores, y):
    """Platt sigmoid mapping SVM decision scores to posterior probabilities."""
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=5000)
    lr.fit(np.asarray(scores).reshape(-1, 1), np.asarray(y))
    return lr


def platt_prob(platt, scores):
    """Apply a fitted Platt sigmoid."""
    return platt.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]


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
    ytr = (Ttr[lab].values.astype(int) > 0).astype(int)
    yva = (Tva[lab].values.astype(int) > 0).astype(int)
    yte = (Tte[lab].values.astype(int) > 0).astype(int)

    features = locked_predictors(DATA_DIR, {"train": Ttr, "val": Tva, "test": Tte})

    Xtr = to_numeric(Ttr[features])
    Xva = to_numeric(Tva[features])
    Xte = to_numeric(Tte[features])

    for name, y in [("train", ytr), ("val", yva), ("test", yte)]:
        pos, neg = int(y.sum()), int((y == 0).sum())
        print(
            f"  {name:<5} {pos:>5} positive  {neg:>5} negative  "
            f"({100 * pos / (pos + neg):.1f}% positive)"
        )

    # ── Standardise on training statistics ──
    scaler = StandardScaler()
    Xtrz = scaler.fit_transform(Xtr)
    Xvaz = scaler.transform(Xva)
    Xtez = scaler.transform(Xte)

    metrics, perm = {}, []

    # ── Binary logistic regression ──
    # L1 penalty, lambda by 5-fold CV on the training partition (lassoglm equivalent).
    print("\nTraining BLR...")
    blr = LogisticRegressionCV(
        Cs=100,
        cv=CV_FOLDS,
        penalty="l1",
        solver="saga",
        scoring="neg_log_loss",
        random_state=RANDOM_SEED,
        max_iter=5000,
    )
    blr.fit(Xtrz, ytr)

    p_blr_te = blr.predict_proba(Xtez)[:, 1]
    thr_blr = f1_threshold(yva, blr.predict_proba(Xvaz)[:, 1])
    metrics["BLR"] = evaluate(yte, p_blr_te, thr_blr, "BLR")

    imp = permutation_importance(
        lambda X: blr.predict_proba(X)[:, 1], Xtez, yte, features
    )
    perm.append(imp.assign(model="BLR"))

    # ── Support vector machine ──
    # Calibrate five-fold training and validation scores; exclude the test set.
    print("\nTraining SVM...")
    svm = SVC(kernel="rbf", gamma="scale", random_state=RANDOM_SEED)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    s_oof = cross_val_predict(svm, Xtrz, ytr, cv=cv, method="decision_function")

    svm.fit(Xtrz, ytr)
    s_va, s_te = svm.decision_function(Xvaz), svm.decision_function(Xtez)

    platt = fit_platt(np.concatenate([s_oof, s_va]), np.concatenate([ytr, yva]))

    p_svm_te = platt_prob(platt, s_te)
    thr_svm = f1_threshold(yva, platt_prob(platt, s_va))
    metrics["SVM"] = evaluate(yte, p_svm_te, thr_svm, "SVM")

    imp = permutation_importance(
        lambda X: platt_prob(platt, svm.decision_function(X)), Xtez, yte, features
    )
    perm.append(imp.assign(model="SVM"))

    # ── Random forest ──
    # Minimum leaf size selected on validation ROC-AUC.
    print("\nTraining RF...")
    best = {"auc": -np.inf}
    for leaf in LEAF_GRID:
        rf = RandomForestClassifier(
            n_estimators=N_TREES,
            min_samples_leaf=leaf,
            oob_score=True,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        rf.fit(Xtr, ytr)
        auc = roc_auc_score(yva, rf.predict_proba(Xva)[:, 1])
        print(f"  min_samples_leaf={leaf:<3} val ROC-AUC={auc:.5f}")
        if auc > best["auc"]:
            best = {"auc": auc, "leaf": leaf, "model": rf}

    rf = best["model"]
    print(f"  selected min_samples_leaf={best['leaf']}")

    p_rf_te = rf.predict_proba(Xte)[:, 1]
    thr_rf = f1_threshold(yva, rf.predict_proba(Xva)[:, 1])
    metrics["RF"] = evaluate(yte, p_rf_te, thr_rf, "RF")
    metrics["RF"]["min_samples_leaf"] = best["leaf"]
    metrics["RF"]["val_roc_auc"] = float(best["auc"])
    metrics["RF"]["oob_score"] = float(rf.oob_score_)

    imp_rf = permutation_importance(
        lambda X: rf.predict_proba(X)[:, 1], Xte, yte, features
    )
    perm.append(imp_rf.assign(model="RF"))

    # Impurity importance is biased toward continuous predictors, so it is
    # reported alongside permutation importance rather than in place of it.
    rf_imp = (
        pd.DataFrame(
            {"feature": features, "impurity_importance": rf.feature_importances_}
        )
        .merge(
            imp_rf[["feature", "auc_decrease_mean"]].rename(
                columns={"auc_decrease_mean": "permutation_importance"}
            ),
            on="feature",
        )
        .sort_values("permutation_importance", ascending=False)
    )
    print("\nRF predictor importance")
    print(rf_imp.to_string(index=False))

    # ── Save ──
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(blr, MODELS_DIR / "blr_model.pkl")
    joblib.dump({"svm": svm, "platt": platt}, MODELS_DIR / "svm_model_cal.pkl")
    joblib.dump(rf, MODELS_DIR / "rf_model.pkl")

    (STATS_DIR / "python_model_metrics.json").write_text(
        json.dumps({"models": metrics, "predictors": features}, indent=2) + "\n"
    )
    rf_imp.to_csv(STATS_DIR / "rf_feature_importance.csv", index=False)

    perm_df = pd.concat(perm, ignore_index=True)
    perm_df[["model", "feature", "auc_decrease_mean", "auc_decrease_std"]].to_csv(
        STATS_DIR / "permutation_importance.csv", index=False
    )

    # Test predictions carry inventory labels, so they stay with the restricted data.
    pd.DataFrame(
        {
            "y_true": yte,
            "p_blr": p_blr_te,
            "p_svm": p_svm_te,
            "p_rf": p_rf_te,
        }
    ).to_csv(DATA_DIR / "test_predictions.csv", index=False)

    print(f"\nModels     -> {MODELS_DIR}")
    print(f"Statistics -> {STATS_DIR}")
    print(f"Predictions -> {DATA_DIR / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
