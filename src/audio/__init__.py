"""
═══════════════════════════════════════════════════════════════
  src.audio — Mental Health Audio Classification Module
═══════════════════════════════════════════════════════════════

Classify mental health conditions from interview audio:
    Normal · Depression · Anxiety · ADHD · OCD

Pipeline:
    1. Preprocess    — segment, denoise, VAD, normalise
    2. Features      — 88 clinically-grounded acoustic features
    3. Balance       — SMOTE class rebalancing
    4. Train         — 9 ML models with hyper-parameter tuning
    5. Predict       — single file or folder inference

Quick start:
    from src.audio.pipeline import run_pipeline
    run_pipeline()                    # run all steps
    run_pipeline(steps=[4])           # train only

    from src.audio.predict import AudioPredictor
    p = AudioPredictor()
    print(p.predict("interview.wav"))
"""

__version__ = "2.0.0"

from .config    import LABELS, LABEL_NAMES
from .pipeline  import run_pipeline
from .predict   import AudioPredictor

__all__ = [
    "LABELS",
    "LABEL_NAMES",
    "run_pipeline",
    "AudioPredictor",
]
