"""
Video Preprocessing Module for Mental Health Analysis
Converts raw video to .npy embeddings using ResNet50 (for ADHD/OCD/Anxiety)
and MediaPipe facial features (for Depression)
"""

import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from pathlib import Path
from typing import Optional

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ mediapipe not installed. Install with: pip install mediapipe")


class VideoPreprocessor:
    """Converts raw video files to feature embeddings per disease"""

    def __init__(self, use_gpu: bool = None):
        self.device = torch.device(
            "cuda" if (use_gpu or (use_gpu is None and torch.cuda.is_available())) else "cpu"
        )

        # Load ResNet50 for ADHD, OCD, Anxiety
        weights    = ResNet50_Weights.DEFAULT
        base_model = models.resnet50(weights=weights)
        self.resnet_model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.resnet_model = self.resnet_model.to(self.device).eval()

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

        # Initialize MediaPipe for depression
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh    = mp.solutions.face_mesh
            self.mp_face_mesh_instance = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        # Target feature dimension for depression (must match training: 168)
        self.depression_feature_dim = 168

        print(f"VideoPreprocessor initialized on {self.device}")

    def _extract_depression_features_per_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract 168-dim facial features from a single frame using MediaPipe.
        Mimics CLNF feature structure used during depression model training.

        Feature breakdown (168 total):
        - 136 facial landmark x,y coordinates (68 landmarks × 2)
        - 24 action unit approximations
        - 8 gaze/head pose features

        Args:
            frame: BGR frame from OpenCV

        Returns:
            numpy array of shape (168,)
        """
        features = np.zeros(self.depression_feature_dim, dtype=np.float32)

        if not MEDIAPIPE_AVAILABLE:
            return features

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w      = rgb_frame.shape[:2]
        results   = self.mp_face_mesh_instance.process(rgb_frame)

        if not results.multi_face_landmarks:
            return features  # Return zeros if no face detected

        landmarks = results.multi_face_landmarks[0].landmark

        # --- Part 1: 68 key facial landmarks x,y (136 values) ---
        # MediaPipe gives 468 landmarks, we pick 68 standard ones
        key_indices = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
            51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
            61, 62, 63, 64, 65, 66, 67, 468
        ]

        idx = 0
        for lm_idx in key_indices[:68]:
            if lm_idx < len(landmarks):
                features[idx]     = landmarks[lm_idx].x * w  # x in pixels
                features[idx + 1] = landmarks[lm_idx].y * h  # y in pixels
            idx += 2  # idx goes up to 136

        # --- Part 2: 24 Action Unit approximations ---
        # Derived from distances between key landmark pairs
        def dist(a, b):
            return np.sqrt(
                (landmarks[a].x - landmarks[b].x) ** 2 +
                (landmarks[a].y - landmarks[b].y) ** 2
            )

        au_features = [
            dist(33,  133),   # AU01 inner brow raise
            dist(46,  53),    # AU02 outer brow raise
            dist(55,  285),   # AU04 brow lowerer
            dist(159, 145),   # AU05 upper lid raiser
            dist(386, 374),   # AU06 cheek raiser
            dist(33,  263),   # AU07 lid tightener
            dist(61,  291),   # AU10 upper lip raiser
            dist(0,   17),    # AU12 lip corner puller
            dist(61,  146),   # AU14 dimpler
            dist(78,  308),   # AU15 lip corner depressor
            dist(13,  14),    # AU17 chin raiser
            dist(78,  191),   # AU20 lip stretcher
            dist(0,   164),   # AU23 lip tightener
            dist(13,  312),   # AU24 lip pressor
            dist(61,  308),   # AU25 lips part
            dist(78,  95),    # AU26 jaw drop
            dist(159, 386),   # AU28 lip suck
            dist(33,  159),   # AU43 eyes closed (left)
            dist(263, 386),   # AU43 eyes closed (right)
            dist(70,  63),    # AU44 squint
            dist(107, 336),   # AU45 blink
            dist(55,  8),     # AU46 wink
            dist(4,   5),     # Nose movement
            dist(1,   2),     # Nose tip
        ]

        features[136:160] = np.array(au_features[:24], dtype=np.float32)

        # --- Part 3: 8 Gaze and head pose approximations ---
        # Eye center positions and head orientation estimates
        left_eye_x  = np.mean([landmarks[33].x,  landmarks[133].x])
        left_eye_y  = np.mean([landmarks[33].y,  landmarks[133].y])
        right_eye_x = np.mean([landmarks[263].x, landmarks[362].x])
        right_eye_y = np.mean([landmarks[263].y, landmarks[362].y])

        nose_x = landmarks[1].x
        nose_y = landmarks[1].y

        # Head pose approximation from nose and eye positions
        head_yaw   = (nose_x - (left_eye_x + right_eye_x) / 2) * 100
        head_pitch = (nose_y - (left_eye_y + right_eye_y) / 2) * 100
        head_roll  = (right_eye_y - left_eye_y) * 100

        # Eye openness
        left_eye_open  = dist(159, 145)
        right_eye_open = dist(386, 374)

        gaze_features = [
            left_eye_x, left_eye_y,
            right_eye_x, right_eye_y,
            head_yaw, head_pitch, head_roll,
            (left_eye_open + right_eye_open) / 2  # avg eye openness
        ]

        features[160:168] = np.array(gaze_features[:8], dtype=np.float32)

        return features

    def extract_resnet_embeddings(self, video_path: str,
                                   disease: str) -> Optional[np.ndarray]:
        """Extract ResNet50 embeddings for ADHD, OCD, Anxiety"""
        target_fps     = self.fps_map.get(disease, 1)
        cap            = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        video_fps      = cap.get(cv2.CAP_PROP_FPS)
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
                    feat = self.resnet_model(img)
                features.append(feat.squeeze().cpu().numpy())

            frame_id += 1

        cap.release()
        return np.array(features) if features else None  # [T, 2048]

    def extract_depression_embeddings(self, video_path: str) -> Optional[np.ndarray]:
        """Extract 168-dim CLNF-like features for Depression using MediaPipe"""
        target_fps     = self.fps_map["depression"]
        cap            = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        video_fps      = cap.get(cv2.CAP_PROP_FPS)
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
                feat = self._extract_depression_features_per_frame(frame)
                features.append(feat)

            frame_id += 1

        cap.release()
        return np.array(features) if features else None  # [T, 168]

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
        disease    = disease.lower()

        if not video_path.exists():
            return {"success": False, "error": f"Video not found: {video_path}"}

        try:
            print(f"Extracting embeddings for {disease} from {video_path.name}...")

            # Use different extraction based on disease
            if disease == "depression":
                embedding = self.extract_depression_embeddings(str(video_path))
            else:
                embedding = self.extract_resnet_embeddings(str(video_path), disease)

            if embedding is None:
                return {"success": False, "error": "No frames could be extracted"}

            if output_path:
                np.save(output_path, embedding)
                print(f"Embedding saved to: {output_path}")

            return {
                "success":   True,
                "embedding": embedding,
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
    preprocessor = get_preprocessor()
    return preprocessor.preprocess(video_path, disease, output_path)