"""
═══════════════════════════════════════════════════════════════
  Audio Pipeline — Model Training & Evaluation
  9 ML models • Hyperparameter tuning • Stacking ensemble
═══════════════════════════════════════════════════════════════

Models included:
  ─ Random Forest          (bagging, feature importance)
  ─ Extra Trees            (more randomised splits)
  ─ Gradient Boosting      (sequential boosting)
  ─ XGBoost                (regularised boosting)
  ─ LightGBM               (leaf-wise boosting, fast)
  ─ CatBoost               (ordered boosting, handles cats)
  ─ SVM  (RBF)             (kernel-based, great on small data)
  ─ Stacking Ensemble      (meta-learner on top of base models)
  ─ Soft Voting Ensemble   (weighted probability averaging)
"""

from __future__ import annotations

import warnings, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, StackingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
)

import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

from . import config as C

# ── Try importing CatBoost (optional) ──
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


# ═════════════════════════════════════════════════════════════
#  Model & Hyper-parameter Definitions
# ═════════════════════════════════════════════════════════════

def _base_models() -> dict:
    """Return dict of  name → unfitted estimator."""
    rs = C.RANDOM_STATE
    models = {
        "Random Forest":      RandomForestClassifier(
                                  n_estimators=200, random_state=rs, n_jobs=-1),
        "Extra Trees":        ExtraTreesClassifier(
                                  n_estimators=200, random_state=rs, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingClassifier(
                                  n_estimators=150, random_state=rs),
        "XGBoost":            xgb.XGBClassifier(
                                  n_estimators=200, eval_metric="mlogloss",
                                  random_state=rs, n_jobs=-1, verbosity=0),
        "LightGBM":           lgb.LGBMClassifier(
                                  n_estimators=200, random_state=rs,
                                  n_jobs=-1, verbose=-1),
        "SVM (RBF)":          SVC(kernel="rbf", probability=True,
                                  random_state=rs),
    }
    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostClassifier(
            iterations=200, random_state=rs, verbose=0, thread_count=-1)
    return models


def _param_grids() -> dict:
    """Hyper-parameter search spaces for tuning."""
    return {
        "Random Forest": {
            "n_estimators":  [100, 200, 300, 500],
            "max_depth":     [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
            "max_features":      ["sqrt", "log2", 0.5],
        },
        "XGBoost": {
            "n_estimators":    [100, 200, 300, 500],
            "max_depth":       [3, 5, 7, 10],
            "learning_rate":   [0.01, 0.05, 0.1, 0.2],
            "subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight":[1, 3, 5, 7],
            "gamma":           [0, 0.1, 0.2, 0.3],
            "reg_alpha":       [0, 0.01, 0.1, 1.0],
            "reg_lambda":      [0.5, 1.0, 1.5, 2.0],
        },
        "LightGBM": {
            "n_estimators":    [100, 200, 300, 500],
            "num_leaves":      [15, 31, 50, 80],
            "max_depth":       [-1, 5, 10, 20],
            "learning_rate":   [0.01, 0.05, 0.1, 0.2],
            "subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_samples":[5, 10, 20, 30],
            "reg_alpha":       [0, 0.01, 0.1, 1.0],
            "reg_lambda":      [0, 0.01, 0.1, 1.0],
        },
        "SVM (RBF)": {
            "C":     [0.1, 1, 10, 50, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        },
    }


def _ensemble_models(fitted: dict) -> dict:
    """Build ensemble models from already-fitted base estimators."""
    # pick 3 best for stacking / voting
    top3_names = list(fitted.keys())[:3]       # already sorted by F1
    estimators = [(n, fitted[n]) for n in top3_names]

    ensembles = {
        "Stacking Ensemble": StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=C.CV_FOLDS, n_jobs=-1,
        ),
        "Voting Ensemble": VotingClassifier(
            estimators=estimators,
            voting="soft", n_jobs=-1,
        ),
    }
    return ensembles


# ═════════════════════════════════════════════════════════════
#  Feature Importance Plotting
# ═════════════════════════════════════════════════════════════

_CAT_MAP = {
    "mfcc": "MFCC", "delta": "MFCC", "pitch": "Pitch",
    "jitter": "Voice Quality", "shimmer": "Voice Quality", "hnr": "Voice Quality",
    "f1_": "Formants", "f2_": "Formants", "f3_": "Formants",
    "energy": "Energy", "rms": "Energy",
    "spec_": "Spectral", "chroma": "Chroma",
    "zcr": "Speech", "pause": "Speech", "speech_rate": "Speech",
}


def _feature_category(name: str) -> str:
    n = name.lower()
    for key, cat in _CAT_MAP.items():
        if key in n:
            return cat
    return "Other"


def _plot_feature_importance(model, feat_names: list, name: str, out_dir: Path):
    if not hasattr(model, "feature_importances_"):
        return

    imp = model.feature_importances_
    fi  = pd.DataFrame({"Feature": feat_names, "Importance": imp,
                         "Category": [_feature_category(f) for f in feat_names]})
    fi  = fi.sort_values("Importance", ascending=False)
    top = fi.head(20)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Top-20 bar
    sns.barplot(x="Importance", y="Feature", data=top, hue="Category",
                dodge=False, ax=axes[0])
    axes[0].set_title(f"Top 20 Features — {name}")

    # Category totals
    cat = fi.groupby("Category")["Importance"].sum().sort_values(ascending=False)
    sns.barplot(x=cat.values, y=cat.index, palette="viridis", ax=axes[1])
    axes[1].set_title(f"Category Contribution — {name}")

    plt.tight_layout()
    slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(out_dir / f"{slug}_features.png", dpi=150)
    plt.close()


# ═════════════════════════════════════════════════════════════
#  Training Loop
# ═════════════════════════════════════════════════════════════

def train_and_evaluate(
    input_csv:  str | Path = C.FEATURES_CSV,
    model_dir:  str | Path = C.MODEL_DIR,
    eval_dir:   str | Path = C.EVAL_DIR,
    tune:       bool       = True,
    apply_smote: bool      = C.TRAIN_APPLY_SMOTE,
) -> dict:
    """
    Full training pipeline:
      1. Load CSV         2. Scale features
      3. Train 7 base models (with optional tuning)
      4. Build 2 ensembles      5. Evaluate all 9
      6. Save best model + scaler + metrics
    """
    out = Path(model_dir);  out.mkdir(parents=True, exist_ok=True)
    ev  = Path(eval_dir);   ev.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & prep ──────────────────────────────────────
    df = pd.read_csv(str(input_csv))
    drop_cols = [c for c in ("patient_id", "condition", "label") if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["label"]
    feat_names = list(X.columns)

    # Ensure model matrix is finite before scaling/SMOTE.
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().sum().sum() > 0:
        X = X.fillna(X.median(numeric_only=True))
        # Any all-NaN columns remain NaN after median fill; backfill with 0.
        X = X.fillna(0.0)

    # sequential label mapping  (XGBoost/LightGBM need 0..N-1)
    uniq = sorted(y.unique())
    lmap = {v: i for i, v in enumerate(uniq)}
    rmap = {i: v for v, i in lmap.items()}
    y_mapped = y.map(lmap)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_mapped, test_size=C.TEST_SIZE,
        stratify=y_mapped, random_state=C.RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    X_train_fit = X_train_s
    y_train_fit = y_train
    if apply_smote:
        cls, cnt = np.unique(y_train, return_counts=True)
        min_class_count = int(cnt.min()) if len(cnt) else 0
        if min_class_count > 1:
            k_neighbors = min(C.SMOTE_K_NEIGHBORS, min_class_count - 1)
            smote = SMOTE(random_state=C.RANDOM_STATE, k_neighbors=k_neighbors)
            X_train_fit, y_train_fit = smote.fit_resample(X_train_s, y_train)

            print(f"\n  Train balancing: SMOTE enabled (k_neighbors={k_neighbors})")
            before = pd.Series(y_train).value_counts().sort_index()
            after = pd.Series(y_train_fit).value_counts().sort_index()
            print("  Class counts before SMOTE:")
            for c, n in before.items():
                print(f"    class {c}: {n}")
            print("  Class counts after SMOTE:")
            for c, n in after.items():
                print(f"    class {c}: {n}")
        else:
            print("\n  Train balancing skipped: at least one class has <2 samples in train split")
    else:
        print("\n  Train balancing: SMOTE disabled")

    joblib.dump(scaler, out / "scaler.joblib")

    # ── 2. Train base models ─────────────────────────────────
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True,
                         random_state=C.RANDOM_STATE)

    grids   = _param_grids()
    results = {}           # name → {model, f1, acc, …}
    fitted  = {}           # name → trained estimator  (for ensembles)

    models = _base_models()

    header = (f"\n{'Model':<22} {'CV F1':>8} {'Test F1':>8} "
              f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'AUC':>7}")
    print("═" * 70)
    print("  MODEL TRAINING")
    print("═" * 70)
    print(header)
    print("─" * 70)

    for name, model in models.items():

        # ── optional hyper-parameter tuning ──
        if tune and name in grids:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                searcher = RandomizedSearchCV(
                    model, grids[name],
                    n_iter=C.TUNE_ITERS, scoring=C.PRIMARY_METRIC,
                    cv=cv, random_state=C.RANDOM_STATE, n_jobs=-1,
                )
                searcher.fit(X_train_fit, y_train_fit)
                model = searcher.best_estimator_
        else:
            # cross-val then full fit
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_fit, y_train_fit)

        # ── evaluate ──
        metrics = _evaluate(model, X_train_fit, y_train_fit, X_test_s, y_test,
                            name, feat_names, uniq, cv, ev)
        results[name] = metrics
        fitted[name]  = model

    # ── 3. Sort base models by F1, build ensembles ───────────
    fitted = dict(sorted(fitted.items(),
                         key=lambda kv: results[kv[0]]["f1"], reverse=True))

    for name, ens_model in _ensemble_models(fitted).items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ens_model.fit(X_train_fit, y_train_fit)

        metrics = _evaluate(ens_model, X_train_fit, y_train_fit, X_test_s, y_test,
                            name, feat_names, uniq, cv, ev)
        results[name] = metrics
        fitted[name]  = ens_model

    # ── 4. Pick best & save ──────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_f1   = results[best_name]["f1"]
    best_model = fitted[best_name]

    print("─" * 70)
    print(f"\n  ★ Best Model: {best_name}  (F1 = {best_f1:.4f})")

    joblib.dump(
        {"model": best_model, "label_map": rmap, "features": feat_names},
        out / "best_model.joblib",
    )

    # Save all results as JSON
    summary = {n: {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in m.items() if k != "model"}
               for n, m in results.items()}
    (ev / "results.json").write_text(json.dumps(summary, indent=2))

    print(f"  ✓ Saved model    → {out / 'best_model.joblib'}")
    print(f"  ✓ Saved scaler   → {out / 'scaler.joblib'}")
    print(f"  ✓ Saved metrics  → {ev / 'results.json'}")

    return {"best_model": best_name, "best_f1": best_f1, "results": summary}


# ═════════════════════════════════════════════════════════════
#  Evaluation Helper
# ═════════════════════════════════════════════════════════════

def _evaluate(model, X_tr, y_tr, X_te, y_te,
              name, feat_names, labels, cv, ev_dir) -> dict:
    """Compute full metrics, print one-liner, save confusion matrix."""
    y_pred = model.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average="weighted")
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)

    auc = float("nan")
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_te)
        try:
            n_cls = len(np.unique(y_te))
            auc = (roc_auc_score(y_te, prob[:, 1]) if n_cls == 2
                   else roc_auc_score(y_te, prob, multi_class="ovr",
                                       average="weighted"))
        except Exception:
            pass

    # CV score (on training data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_f1 = cross_val_score(model, X_tr, y_tr, cv=cv,
                                scoring=C.PRIMARY_METRIC, n_jobs=-1).mean()

    # One-liner table row
    print(f"  {name:<20} {cv_f1:>8.4f} {f1:>8.4f} "
          f"{acc:>7.4f} {prec:>7.4f} {rec:>7.4f} {auc:>7.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("True"); plt.xlabel("Predicted")
    slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(ev_dir / f"{slug}_cm.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Feature importance
    _plot_feature_importance(model, feat_names, name, ev_dir)

    return {"f1": f1, "accuracy": acc, "precision": prec,
            "recall": rec, "auc_roc": auc, "cv_f1": cv_f1}
