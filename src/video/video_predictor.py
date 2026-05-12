"""
Video Prediction Module for Mental Health Analysis
Uses LSTM models to predict mental health conditions from video embeddings
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
from src.video.video_preprocessor import get_preprocessor


class LSTMClassifier(nn.Module):
    """Bi-LSTM classifier — must match training architecture exactly"""

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
            # No Sigmoid — applied manually at inference
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.classifier(last_output).squeeze()


class VideoPredictor:
    """Predicts mental health conditions from video embeddings"""

    # Config per disease: (model_path, input_dim, max_len)
    DISEASE_CONFIG = {
        "adhd":       {"input_dim": 2048, "max_len": 100},
        "ocd":        {"input_dim": 2048, "max_len": 100},
        "anxiety":    {"input_dim": 2048, "max_len": 100},
        # NOTE: depression was trained with 168-dim features (not ResNet50 2048-dim).
        # It will raise a shape mismatch at inference. The scheduler catches this gracefully.
        "depression": {"input_dim": 168,  "max_len": 1000}
    }

    def __init__(self, model_dir: str = "models/video_model"):
        """
        Initialize predictor and load all disease models

        Args:
            model_dir: Path to folder containing all .pth model files
        """
        self.model_dir = Path(model_dir)
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models    = {}

        # Load all 4 models
        for disease, config in self.DISEASE_CONFIG.items():
            model_path = self.model_dir / f"{disease}_best_model.pth"

            if not model_path.exists():
                print(f"⚠️ Model not found for {disease}: {model_path}")
                continue

            model = LSTMClassifier(input_dim=config["input_dim"])
            model.load_state_dict(
                torch.load(str(model_path), map_location=self.device)
            )
            model.to(self.device).eval()
            self.models[disease] = model
            print(f"✓ Loaded model for {disease}")

        print(f"VideoPredictor initialized on {self.device}")

    def _prepare_tensor(self, embedding: np.ndarray, disease: str) -> torch.Tensor:
        """
        Pad or truncate embedding to required length and convert to tensor

        Args:
            embedding: numpy array of shape [T, input_dim]
            disease: disease name to get max_len

        Returns:
            tensor of shape [1, max_len, input_dim]
        """
        max_len = self.DISEASE_CONFIG[disease]["max_len"]

        # Fix NaN or Inf
        embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        # Pad or truncate
        if len(embedding) < max_len:
            pad       = np.zeros((max_len - len(embedding), embedding.shape[1]))
            embedding = np.vstack([embedding, pad])
        else:
            embedding = embedding[:max_len]

        # Add batch dimension [1, max_len, input_dim]
        tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        return tensor.to(self.device)

    def predict_from_embedding(self, embedding: np.ndarray,
                                disease: str) -> Dict[str, any]:
        """
        Predict from a preloaded numpy embedding

        Args:
            embedding: numpy array of shape [T, input_dim]
            disease: Disease type ('adhd', 'ocd', 'anxiety', 'depression')

        Returns:
            Prediction dictionary
        """
        disease = disease.lower()

        if disease not in self.models:
            return {"success": False, "error": f"Model not loaded for {disease}"}

        try:
            tensor = self._prepare_tensor(embedding, disease)

            with torch.no_grad():
                output      = self.models[disease](tensor)
                probability = torch.sigmoid(output).item()

            return {
                "success":      True,
                "disease":      disease,
                "probability":  round(probability, 4),
                "has_disorder": probability >= 0.5
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict_from_npy(self, npy_path: str, disease: str) -> Dict[str, any]:
        """
        Predict from a saved .npy embedding file

        Args:
            npy_path: Path to .npy embedding file
            disease: Disease type

        Returns:
            Prediction dictionary
        """
        try:
            embedding = np.load(npy_path)
            return self.predict_from_embedding(embedding, disease)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict_from_video(self, video_path: str, disease: str) -> Dict[str, any]:
        """
        Full pipeline: raw video -> prediction

        Args:
            video_path: Path to raw video file
            disease: Disease type ('adhd', 'ocd', 'anxiety', 'depression')

        Returns:
            Prediction dictionary
        """
        # Step 1: Extract embedding from video
        preprocessor = get_preprocessor()
        result       = preprocessor.preprocess(video_path, disease)

        if not result["success"]:
            return {"success": False, "error": result["error"]}

        # Step 2: Predict from embedding
        return self.predict_from_embedding(result["embedding"], disease)

    def predict_all_diseases(self, video_path: str) -> Dict[str, any]:
        """
        Run prediction for all 4 diseases from one video

        Args:
            video_path: Path to raw video file

        Returns:
            Dictionary with predictions for all diseases
        """
        predictions = {}

        for disease in self.DISEASE_CONFIG.keys():
            print(f"Predicting {disease}...")
            result = self.predict_from_video(video_path, disease)
            predictions[disease] = result

        return {
            "success":     True,
            "video":       video_path,
            "predictions": predictions
        }


# Singleton instance
_predictor = None


def get_predictor(model_dir: str = "models/video_model") -> VideoPredictor:
    global _predictor
    if _predictor is None:
        _predictor = VideoPredictor(model_dir=model_dir)
    return _predictor


def predict(video_path: str, disease: str,
            model_dir: str = "models/video_model") -> Dict[str, any]:
    """
    Convenience function: raw video -> prediction for one disease

    Args:
        video_path: Path to raw video file
        disease: Disease type ('adhd', 'ocd', 'anxiety', 'depression')
        model_dir: Path to models folder

    Returns:
        Prediction dictionary
    """
    predictor = get_predictor(model_dir=model_dir)
    return predictor.predict_from_video(video_path, disease)


def predict_all(video_path: str,
                model_dir: str = "models/video_model") -> Dict[str, any]:
    """
    Convenience function: raw video -> predictions for all diseases

    Args:
        video_path: Path to raw video file
        model_dir: Path to models folder

    Returns:
        Dictionary with all disease predictions
    """
    predictor = get_predictor(model_dir=model_dir)
    return predictor.predict_all_diseases(video_path)