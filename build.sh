#!/usr/bin/env bash
set -e  # Exit on any error

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Downloading text model from Google Drive ==="
MODEL_PATH="models/text_model/best_model.pt"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading best_model.pt..."
    pip install gdown -q
    gdown "1EsroigS-qVfx8sA8IYKCG754wjelAIxa" -O "$MODEL_PATH"
    echo "Download complete: $MODEL_PATH"
else
    echo "best_model.pt already exists, skipping download."
fi

echo "=== Build complete ==="
