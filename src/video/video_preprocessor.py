"""
Video Preprocessing Module for Mental Health Analysis
Converts raw video to .npy embeddings using ResNet50
"""

import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from pathlib import Path
from typing import Optional


class VideoPreprocessor:
    """Converts raw video files to ResNet50 feature embeddings"""

    def __init__(self, use_gpu: bool = None):
        """
        Initialize the preprocessor with ResNet50 model

        Args:
            use_gpu: Whether to use GPU (None for auto-detect)
        """
        self.device = torch.device(
            "cuda" if (use_gpu or (use_gpu is None and torch.cuda.is_available())) else "cpu"
        )

        # Load ResNet50 as feature extractor
        weights    = ResNet50_Weights.DEFAULT
        base_model = models.resnet50(weights=weights)
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model = self.model.to(self.device).eval()

        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # FPS settings per disease
        self.fps_map = {
            "adhd":       3,
            "ocd":        3,
            "anxiety":    1,
            "depression": 1
        }

        print(f"VideoPreprocessor initialized on {self.device}")

    def extract_embeddings(self, video_path: str, disease: str) -> Optional[np.ndarray]:
        """
        Extract ResNet50 embeddings from video

        Args:
            video_path: Path to input video file
            disease: Disease type ('adhd', 'ocd', 'anxiety', 'depression')

        Returns:
            numpy array of shape [T, 2048] or None if failed
        """
        disease    = disease.lower()
        target_fps = self.fps_map.get(disease, 1)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0:
            cap.release()
            raise RuntimeError("Could not read FPS from video")

        frame_interval = max(1, int(round(video_fps / target_fps)))
        features       = []
        frame_id       = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img   = self.transform(frame).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    feat = self.model(img)
                features.append(feat.squeeze().cpu().numpy())

            frame_id += 1

        cap.release()

        if not features:
            return None

        return np.array(features)  # Shape: [T, 2048]

    def preprocess(self, video_path: str, disease: str,
                   output_path: Optional[str] = None) -> dict:
        """
        Full preprocessing pipeline: video -> .npy embedding

        Args:
            video_path: Path to input video file
            disease: Disease type ('adhd', 'ocd', 'anxiety', 'depression')
            output_path: Optional path to save .npy file

        Returns:
            Result dictionary with embedding and metadata
        """
        video_path = Path(video_path)

        if not video_path.exists():
            return {"success": False, "error": f"Video not found: {video_path}"}

        try:
            print(f"Extracting embeddings for {disease} from {video_path.name}...")
            embedding = self.extract_embeddings(str(video_path), disease)

            if embedding is None:
                return {"success": False, "error": "No frames could be extracted"}

            # Save .npy if output path provided
            if output_path:
                np.save(output_path, embedding)
                print(f"Embedding saved to: {output_path}")

            return {
                "success":   True,
                "embedding": embedding,        # [T, 2048]
                "shape":     embedding.shape,
                "disease":   disease,
                "video":     str(video_path)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


# Singleton instance
_preprocessor = None


def get_preprocessor(use_gpu: bool = None) -> VideoPreprocessor:
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = VideoPreprocessor(use_gpu=use_gpu)
    return _preprocessor


def preprocess_video(video_path: str, disease: str,
                     output_path: Optional[str] = None) -> dict:
    """
    Convenience function to preprocess a video

    Args:
        video_path: Path to input video
        disease: Disease type ('adhd', 'ocd', 'anxiety', 'depression')
        output_path: Optional path to save .npy file

    Returns:
        Result dictionary
    """
    preprocessor = get_preprocessor()
    return preprocessor.preprocess(video_path, disease, output_path)