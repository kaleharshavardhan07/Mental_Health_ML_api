"""
═══════════════════════════════════════════════════════════════
  Audio Pipeline — Step-by-Step Orchestrator
  Run the full pipeline or individual steps from CLI.
═══════════════════════════════════════════════════════════════

Usage:
    # Run everything
    python -m src.audio.pipeline

    # Run individual steps
    python -m src.audio.pipeline --step 1      # Segment only
    python -m src.audio.pipeline --step 2      # Extract features
    python -m src.audio.pipeline --step 3      # Balance dataset
    python -m src.audio.pipeline --step 4      # Train models
    python -m src.audio.pipeline --step 1-3    # Steps 1 through 3
"""

from __future__ import annotations

import time
import argparse
from pathlib import Path

from . import config as C


# ═════════════════════════════════════════════════════════════
#  Step Definitions
# ═════════════════════════════════════════════════════════════

def step1_segment():
    """Step 1 — Segment & normalise audio lengths."""
    from .preprocess import segment_directory

    print("\n" + "━" * 60)
    print("  STEP 1 ▸ AUDIO SEGMENTATION")
    print("━" * 60)
    print(f"  Input  : {C.RAW_AUDIO_DIR}")
    print(f"  Output : {C.SEGMENTED_DIR}")
    print(f"  Rules  : pad < {C.SEGMENT_PAD}s │ keep ≤ {C.SEGMENT_MAX}s │"
          f" split > {C.SEGMENT_MAX}s ({C.SEGMENT_OVERLAP:.0%} overlap)")

    segment_directory(C.RAW_AUDIO_DIR, C.SEGMENTED_DIR)


def step2_extract_features():
    """Step 2 — Clean audio + extract 88 features → CSV."""
    from .dataset import build_dataset

    print("\n" + "━" * 60)
    print("  STEP 2 ▸ FEATURE EXTRACTION")
    print("━" * 60)
    print(f"  Input  : {C.SEGMENTED_DIR}")
    print(f"  Output : {C.FEATURES_CSV}")
    print("  Groups : MFCC · Pitch · Voice Quality · Formants")
    print("           Energy · Spectral · Speech Patterns · Chroma")

    build_dataset(C.SEGMENTED_DIR, C.FEATURES_CSV)


def step3_balance():
    """Step 3 — Optional: export a pre-balanced CSV with SMOTE."""
    from .dataset import balance_dataset

    print("\n" + "━" * 60)
    print("  STEP 3 ▸ CLASS BALANCING (OPTIONAL EXPORT)")
    print("━" * 60)
    print(f"  Input  : {C.FEATURES_CSV}")
    print(f"  Output : {C.BALANCED_CSV}")
    print("  Note   : training now balances after split inside Step 4")

    balance_dataset(C.FEATURES_CSV, C.BALANCED_CSV)


def step4_train():
    """Step 4 — Train 9 models, tune, evaluate, save best."""
    from .train import train_and_evaluate

    print("\n" + "━" * 60)
    print("  STEP 4 ▸ MODEL TRAINING & EVALUATION")
    print("━" * 60)
    print(f"  Input  : {C.FEATURES_CSV}")
    print(f"  Balance: train-only SMOTE = {C.TRAIN_APPLY_SMOTE}")
    print(f"  Models : {C.MODEL_DIR}")
    print(f"  Eval   : {C.EVAL_DIR}")
    print("  Lineup : RF · ExtraTrees · GBDT · XGBoost · LightGBM")
    print("           CatBoost · SVM · Stacking · Voting")

    train_and_evaluate(
        C.FEATURES_CSV,
        C.MODEL_DIR,
        C.EVAL_DIR,
        tune=True,
        apply_smote=C.TRAIN_APPLY_SMOTE,
    )


STEPS = {
    1: ("Segment Audio",      step1_segment),
    2: ("Extract Features",   step2_extract_features),
    3: ("Balance Dataset",    step3_balance),
    4: ("Train Models",       step4_train),
}


# ═════════════════════════════════════════════════════════════
#  Runner
# ═════════════════════════════════════════════════════════════

def run_pipeline(steps: list[int] | None = None):
    """
    Run selected pipeline steps (default: all).

    Parameters
    ----------
    steps : list of step numbers [1-4], or None for all.
    """
    if steps is None:
        steps = list(STEPS.keys())

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║     🧠  AUDIO MENTAL HEALTH CLASSIFICATION PIPELINE    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Steps to run: {steps}")
    print(f"  Project root: {C.PROJECT_ROOT}")

    t0 = time.time()

    for i in steps:
        if i not in STEPS:
            print(f"\n  ⚠ Unknown step {i}, skipping")
            continue

        name, fn = STEPS[i]
        st = time.time()
        fn()
        elapsed = time.time() - st
        print(f"\n  ✓ Step {i} ({name}) completed in {elapsed:.1f}s")

    total = time.time() - t0
    print(f"\n{'═' * 60}")
    print(f"  ✓ Pipeline finished in {total:.1f}s")
    print(f"{'═' * 60}\n")


# ═════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════

def _parse_steps(spec: str) -> list[int]:
    """Parse '1-3' or '2' or '1,3,4' into list of ints."""
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x.strip()) for x in spec.split(",")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audio Mental Health Classification Pipeline"
    )
    parser.add_argument(
        "--step", type=str, default=None,
        help="Step(s) to run: '1', '2-4', '1,3'. Default: all."
    )
    args = parser.parse_args()

    selected = _parse_steps(args.step) if args.step else None
    run_pipeline(selected)
