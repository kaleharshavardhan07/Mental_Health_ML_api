"""
═══════════════════════════════════════════════════════════════
  Audio Pipeline — Centralized Configuration
  All constants, paths, and settings in one place.
═══════════════════════════════════════════════════════════════
"""

from pathlib import Path

# ── Project Paths ────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent
DATA_DIR        = PROJECT_ROOT / "data"
RAW_AUDIO_DIR   = DATA_DIR / "raw" / "audio"
SEGMENTED_DIR   = DATA_DIR / "raw" / "audio_segmented"
FEATURES_CSV    = DATA_DIR / "final_features" / "features.csv"
BALANCED_CSV    = DATA_DIR / "final_features" / "features_balanced.csv"
MODEL_DIR       = PROJECT_ROOT / "models" / "audio_model"
EVAL_DIR        = PROJECT_ROOT / "evaluation" / "audio"

# ── Label Mapping ────────────────────────────────────────────
LABELS = {
    "normal":     0,
    "depression": 1,
    "anxiety":    2,
    "adhd":       3,
    "ocd":        4,
}
LABEL_NAMES = {v: k.title() for k, v in LABELS.items()}

# ── Audio Settings ───────────────────────────────────────────
SAMPLE_RATE       = 16_000        # Hz  — standard for speech
SEGMENT_MAX       = 25            # seconds — split longer audio
SEGMENT_PAD       = 10            # seconds — pad shorter audio
SEGMENT_OVERLAP   = 0.5           # 50 % overlap for long clips
SEGMENT_MAX_PER_FILE = 4          # cap generated segments per source file
NOISE_REDUCE_PROP = 0.8           # noisereduce strength
VAD_AGGRESSIVENESS = 3            # webrtcvad aggressiveness (0-3)
TARGET_RMS        = 0.1           # RMS normalisation target
PRE_EMPHASIS      = 0.97          # pre-emphasis coefficient

# ── Feature Extraction ──────────────────────────────────────
N_MFCC         = 13
N_FFT          = 2048
HOP_LENGTH     = 512
PITCH_FLOOR    = 75               # Hz
PITCH_CEILING  = 600              # Hz

# ── Training ────────────────────────────────────────────────
TEST_SIZE      = 0.20
CV_FOLDS       = 5
RANDOM_STATE   = 42
PRIMARY_METRIC = "f1_weighted"
TUNE_ITERS     = 60               # RandomizedSearchCV iterations
TRAIN_APPLY_SMOTE = True          # apply SMOTE on training split only
SMOTE_K_NEIGHBORS = 5             # adjusted at runtime by min class count
