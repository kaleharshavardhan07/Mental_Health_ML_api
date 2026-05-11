from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoTrainConfig:
    processed_dir: Path
    checkpoints_dir: Path
    seed: int = 42
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 3e-4


def train_video(cfg: VideoTrainConfig) -> Path:
    """Placeholder for video model training."""
    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.checkpoints_dir / "video_model.pt"
    model_path.write_bytes(b"")
    return model_path

