import os
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

# Paths
VIDEO_DIR = "data/raw/video"
OUTPUT_DIR = "data/processed/video"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load pretrained ResNet18
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Remove classifier layer
model = torch.nn.Sequential(*list(model.children())[:-1])

model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_features(video_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(fps))  # 1 frame per second

    frame_id = 0
    features = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % frame_interval == 0:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = transform(frame).unsqueeze(0)

            with torch.no_grad():
                feature = model(img)

            feature = feature.squeeze().numpy()
            features.append(feature)

        frame_id += 1

    cap.release()

    if len(features) == 0:
        return None

    # Average all frame embeddings → one video embedding
    return np.mean(features, axis=0)


# Traverse dataset
for disease in os.listdir(VIDEO_DIR):

    disease_path = os.path.join(VIDEO_DIR, disease)

    if not os.path.isdir(disease_path):
        continue

    # Create corresponding processed folder
    output_disease_dir = os.path.join(OUTPUT_DIR, disease)
    os.makedirs(output_disease_dir, exist_ok=True)

    print(f"\nProcessing class: {disease}")

    for video_file in tqdm(os.listdir(disease_path)):

        # Only process video files
        if not video_file.lower().endswith((".mp4", ".mov", ".avi")):
            continue

        video_path = os.path.join(disease_path, video_file)

        name = os.path.splitext(video_file)[0]

        output_path = os.path.join(output_disease_dir, name + ".npy")

        embedding = extract_features(video_path)

        if embedding is not None:
            np.save(output_path, embedding)
            print(f"Saved: {output_path}")