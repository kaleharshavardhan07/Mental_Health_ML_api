"""
═══════════════════════════════════════════════════════════════
  Audio Pipeline — Dataset Builder
  Directory walker → Feature CSV → SMOTE balancing
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

from . import config as C
from .preprocess import clean_audio
from .features import extract_features


# ═════════════════════════════════════════════════════════════
#  Build Feature CSV from Folder Structure
# ═════════════════════════════════════════════════════════════

def build_dataset(
    data_dir:   str | Path = C.SEGMENTED_DIR,
    output_csv: str | Path = C.FEATURES_CSV,
) -> pd.DataFrame:
    """
    Scan  data_dir/{condition}/*.wav  → extract features → save CSV.

    Supports flat folders  (condition/*.wav)
    and nested folders     (condition/patient/*.wav → aggregated per patient).

    Returns the resulting DataFrame.
    """
    data_path = Path(data_dir)
    out_path  = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    condition_dirs = sorted(
        d for d in data_path.iterdir()
        if d.is_dir() and d.name.lower() in C.LABELS
    )

    if not condition_dirs:
        print(f"  ✗ No condition folders found in {data_dir}")
        return pd.DataFrame()

    for cond_dir in condition_dirs:
        cond  = cond_dir.name.lower()
        label = C.LABELS[cond]
        wavs  = list(cond_dir.rglob("*.wav"))

        print(f"\n  {cond.upper():>12}  label={label}  files={len(wavs)}")

        for wav in tqdm(wavs, desc=f"    {cond}", unit="file", leave=False):
            try:
                audio, sr = clean_audio(wav)
                if audio is None or len(audio) == 0:
                    continue

                feats = extract_features(audio, sr)
                if feats is None:
                    continue

                feats["patient_id"] = wav.stem
                feats["condition"]  = cond
                feats["label"]      = label
                rows.append(feats)

            except Exception as e:
                print(f"\n    ✗ {wav.name}: {e}")

    if not rows:
        print("  ⚠  No features extracted — check your data folder.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Move identifiers to front
    id_cols = ["patient_id", "condition", "label"]
    feat_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + feat_cols]

    df.to_csv(out_path, index=False)
    print(f"\n  ✓ Saved {len(df)} samples × {len(feat_cols)} features → {out_path}")
    return df


# ═════════════════════════════════════════════════════════════
#  Class Balancing (SMOTE + Class Weights)
# ═════════════════════════════════════════════════════════════

def balance_dataset(
    input_csv:  str | Path = C.FEATURES_CSV,
    output_csv: str | Path = C.BALANCED_CSV,
) -> pd.DataFrame:
    """
    Apply SMOTE to balance classes.  Also prints class-weight
    alternative for models that support sample_weight.
    """
    df = pd.read_csv(str(input_csv))
    print(f"\n  Original shape : {df.shape}")
    print(f"  Class counts   :")
    for cls, cnt in df["label"].value_counts().sort_index().items():
        print(f"    {C.LABEL_NAMES.get(cls, cls):>12}: {cnt}")

    # ── Drop identifiers ──
    drop = [c for c in ("patient_id", "condition", "label") if c in df.columns]
    X = df.drop(columns=drop)
    y = df["label"]

    # ── Handle NaN ──
    X = X.dropna(axis=1, how="all")
    if X.isna().sum().sum() > 0:
        X = X.fillna(X.mean())

    # ── Class weights (alternative) ──
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    print("\n  Class weights (for models that accept them):")
    for c, w in zip(classes, weights):
        print(f"    {C.LABEL_NAMES.get(c, c):>12}: {w:.3f}")

    # ── SMOTE ──
    print("\n  Applying SMOTE …")
    try:
        smote = SMOTE(random_state=C.RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X, y)
    except ValueError as e:
        print(f"  ✗ SMOTE error: {e}")
        print("    (need ≥ 6 samples per class for default k_neighbors=5)")
        return df

    balanced = pd.DataFrame(X_res, columns=X.columns)
    balanced.insert(0, "label", y_res)

    # Reconstruct patient_id column
    if "patient_id" in df.columns:
        orig_ids = list(df["patient_id"].values)
        synth    = [f"SYNTH_{i}" for i in range(len(y_res) - len(orig_ids))]
        balanced.insert(0, "patient_id", orig_ids + synth)

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(out, index=False)

    print(f"\n  ✓ Balanced shape: {balanced.shape}")
    print(f"    Saved → {out}")
    for cls, cnt in pd.Series(y_res).value_counts().sort_index().items():
        print(f"      {C.LABEL_NAMES.get(cls, cls):>12}: {cnt}")

    return balanced
