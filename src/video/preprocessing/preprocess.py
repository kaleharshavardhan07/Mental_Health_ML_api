from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoPreprocessConfig:
    raw_dir: Path
    processed_dir: Path


def preprocess_video(cfg: VideoPreprocessConfig) -> None:
    """Placeholder for video preprocessing.

    Typical steps:
    - frame extraction / sampling
    - face detection & alignment
    - tracking and temporal smoothing
    - saving per-segment frame tensors / landmarks
    """
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

