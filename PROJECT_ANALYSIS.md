# Mental Health ML API - Project Analysis

## Overview
This project is a comprehensive mental health assessment API that uses multiple ML models (Audio, Video, Text, MCQ) to predict mental health conditions from various input modalities.

## Project Structure

```
Mental_Health_ML_api/
├── main.py                          # FastAPI application entry point
├── scheduler.py                     # Background job scheduler for MongoDB polling
├── requirements.txt                 # Python dependencies
├── models/                          # Trained model artifacts
│   ├── audio_model/                # Audio-based mental health classifier
│   │   ├── best_audio_model.joblib
│   │   ├── scaler.joblib
│   │   └── [other classifier variants]
│   ├── text_model/                 # DistilBERT-based text classifier
│   │   ├── best_model.pt           # Saved PyTorch model
│   │   ├── config.json             # Model configuration
│   │   └── tokenizer/              # DistilBERT tokenizer files
│   └── video_model/                # Video-based facial expression classifiers
│       ├── depression_best_model.pth
│       ├── anxiety_best_model.pth
│       ├── ocd_best_model.pth
│       └── adhd_best_model.pth
└── src/                            # Source code
    ├── audio/                      # Audio processing pipeline
    │   ├── predict.py             # Audio prediction interface
    │   ├── config.py              # Audio model configuration
    │   ├── dataset.py             # Audio dataset handling
    │   ├── features.py            # Audio feature extraction
    │   ├── pipeline.py            # Audio preprocessing pipeline
    │   ├── preprocess.py          # Audio preprocessing utilities
    │   └── train.py               # Audio model training
    ├── text/                       # Text processing pipeline
    │   ├── predict.py             # Text prediction interface (DistilBERT)
    │   ├── video_to_text.py       # Speech-to-text from video
    │   └── README.md              # Text model documentation
    ├── video/                      # Video processing pipeline
    │   ├── video_predictor.py     # Video prediction interface
    │   └── video_preprocessor.py  # Video preprocessing
    └── mcq/                        # MCQ processing
        └── Final_Preprocessing/
            ├── mcq_inference.py   # MCQ prediction interface
            └── mcq_preprocessing.py
```

## Model Details

### 1. Text Model (DistilBERT-based)

**Purpose:** Predict mental health conditions from text input (speech transcripts, written responses)

**Architecture:**
- **Backbone:** DistilBERT-base-uncased (frozen bottom 4 layers)
- **Classification Head:**
  - Dropout(0.3)
  - LayerNorm(768) - normalizes BERT output
  - Linear(768 → 256) - first projection
  - ReLU activation
  - Dropout(0.3)
  - Linear(256 → 4) - final classification

**Classes Detected:** Depression, Anxiety, OCD, ADHD

**Training Details:**
- Training epochs: 7
- Batch size: 16
- Max sequence length: 256
- Hidden dimension: 256
- Frozen layers: 4 (bottom layers of DistilBERT)

**Input Format:** Text string (e.g., speech transcripts, questionnaire responses)

**Output Format:**
```json
{
  "success": true,
  "predictions": {
    "depression": {"probability": 0.72, "has_disorder": true},
    "anxiety": {"probability": 0.45, "has_disorder": false},
    "ocd": {"probability": 0.23, "has_disorder": false},
    "adhd": {"probability": 0.31, "has_disorder": false}
  },
  "input_length": 45
}
```

**Integration Points:**
- `/predict/text` - Direct text prediction API endpoint
- Scheduler - Processes video interviews by extracting speech and predicting
- Used in multi-modal assessment pipeline

### 2. Audio Model

**Purpose:** Predict mental health conditions from audio features

**Features Used:**
- MFCC (Mel-frequency cepstral coefficients)
- Chroma features
- Spectral contrast
- Tonnetz
- Zero crossing rate
- Spectral rolloff
- Tempo and rhythm features

**Models:**
- Primary: ExtraTreesClassifier
- Alternatives: RandomForest, SVM, XGBoost, LightGBM, GradientBoosting

**Classes Detected:** Normal, Depression, Anxiety, OCD, ADHD

**Output Format:**
```json
{
  "probabilities": {
    "Normal": 0.05,
    "Depression": 0.78,
    "Anxiety": 0.12,
    "OCD": 0.03,
    "ADHD": 0.02
  },
  "condition": "Depression",
  "confidence": 0.78
}
```

### 3. Video Model

**Purpose:** Predict mental health conditions from facial expressions in video

**Architecture:** CNN-based models (separate models for each condition)

**Models:**
- depression_best_model.pth
- anxiety_best_model.pth
- ocd_best_model.pth
- adhd_best_model.pth

**Classes Detected:** Depression, Anxiety, OCD, ADHD

**Output Format:**
```json
{
  "predictions": {
    "depression": {"probability": 0.65, "has_disorder": true},
    "anxiety": {"probability": 0.42, "has_disorder": false},
    "ocd": {"probability": 0.18, "has_disorder": false},
    "adhd": {"probability": 0.28, "has_disorder": false}
  }
}
```

### 4. MCQ Model

**Purpose:** Predict mental health conditions from multiple-choice questionnaire answers

**Models:**
- XGBoost classifier
- Logistic Regression
- Random Forest

**Input Format:** List of 15 questionnaire answers

**Output Format:**
```json
{
  "depression": 72.5,
  "anxiety": 18.0,
  "ocd": 5.0,
  "adhd": 4.5
}
```

## API Endpoints

### 1. Text Prediction
```
POST /predict/text
Content-Type: application/json

{
  "text": "I feel very sad and hopeless most of the time"
}

Response:
{
  "success": true,
  "input_length": 45,
  "prediction": {
    "depression": {"probability": 0.72, "has_disorder": true},
    "anxiety": {"probability": 0.45, "has_disorder": false},
    ...
  }
}
```

### 2. Audio Prediction
```
POST /predict/audio
Content-Type: multipart/form-data

file: <audio_file>

Response:
{
  "success": true,
  "filename": "audio.wav",
  "prediction": {
    "probabilities": {...},
    "condition": "Depression",
    "confidence": 0.78
  }
}
```

### 3. Video Prediction (with Text Extraction)
```
POST /predict/video
Content-Type: multipart/form-data

file: <video_file>

Response:
{
  "success": true,
  "filename": "video.webm",
  "content_type": "video/webm",
  "text_extraction": {
    "text_length": 150,
    "word_count": 25,
    "video_size_bytes": 1024000
  },
  "prediction": {
    "depression": {"probability": 0.68, "has_disorder": true},
    ...
  }
}
```

### 4. MCQ Prediction
```
POST /predict/mcq
Content-Type: application/json

{
  "answers": [
    {"questionId": 1, "answer": "A"},
    {"questionId": 2, "answer": "B"},
    ...
  ]
}

Response:
{
  "depression": 72.5,
  "anxiety": 18.0,
  "ocd": 5.0,
  "adhd": 4.5
}
```

## Scheduler Workflow

The scheduler polls MongoDB every 60 seconds for tests with `mlExecutionStatus = 'in_progress'`:

1. **MCQ Analysis** - Processes 15 questionnaire answers once
2. **Video Processing** (up to 8 videos per test):
   - Downloads video from Firebase URL
   - **Audio Analysis** - Extracts audio features and predicts
   - **Text Analysis** - Extracts speech via speech-to-text, then predicts
   - **Video Analysis** - Extracts facial features and predicts
3. **Result Aggregation** - Averages predictions across all videos
4. **Database Update** - Saves results back to MongoDB

**Final Result Structure:**
```json
{
  "mcq": {"depression": 72.5, "anxiety": 18.0, "ocd": 5.0, "adhd": 4.5},
  "audio": {"Normal": 0.05, "Depression": 0.78, "Anxiety": 0.12, "OCD": 0.03, "ADHD": 0.02},
  "text": {
    "depression": {"probability": 0.72, "has_disorder": true},
    "anxiety": {"probability": 0.45, "has_disorder": false},
    "ocd": {"probability": 0.23, "has_disorder": false},
    "adhd": {"probability": 0.31, "has_disorder": false}
  },
  "video": {
    "depression": {"probability": 0.65, "has_disorder": true},
    "anxiety": {"probability": 0.42, "has_disorder": false},
    "ocd": {"probability": 0.18, "has_disorder": false},
    "adhd": {"probability": 0.28, "has_disorder": false}
  },
  "meta": {
    "videoCount": 8,
    "audioSuccess": 6,
    "textSuccess": 7,
    "videoSuccess": 8
  }
}
```

## Technology Stack

### Core Frameworks
- **FastAPI** - Web framework for REST API
- **Uvicorn** - ASGI server
- **APScheduler** - Background job scheduling

### ML/DL Libraries
- **PyTorch** - Deep learning framework (Text model)
- **Transformers (Hugging Face)** - NLP models (DistilBERT)
- **Scikit-learn** - Traditional ML models (Audio, MCQ)
- **TensorFlow** - Video model inference
- **XGBoost/LightGBM** - Gradient boosting models

### Audio/Video Processing
- **Librosa** - Audio feature extraction
- **MoviePy** - Video processing
- **SpeechRecognition** - Speech-to-text
- **OpenCV** - Image processing

### Data Storage
- **MongoDB** - Test results and metadata
- **Firebase Storage** - Video/audio file storage
- **PyMongo** - MongoDB driver

## Recent Fix

### Issue
Text model failed to load due to architecture mismatch:
```
size mismatch for head.1.weight: copying a param with shape torch.Size([768]) from checkpoint, 
the shape in current model is torch.Size([256, 768])
```

### Solution
Updated `MentalHealthClassifier` in `src/text/predict.py` to match saved checkpoint:
- Changed `head.1` from Linear to LayerNorm
- Changed `head.2` from BatchNorm1d to Linear
- Maintained correct tensor shapes

### Verification
✅ Model loads successfully
✅ Predictions work correctly
✅ Integrates with scheduler and API

## Deployment Notes

### Requirements
```bash
pip install -r requirements.txt
```

### Running the API
```bash
python main.py
```
Server runs on `http://0.0.0.0:8000`

### Environment Variables
- `MONGODB_URI` - MongoDB connection string
- `HF_TOKEN` - Hugging Face API token (optional, for higher rate limits)

### Performance Considerations
- Text model uses CPU by default (can use CUDA if available)
- Video processing is resource-intensive
- Scheduler runs every 60 seconds
- Models are loaded lazily on first request

## Future Improvements

1. **Model Optimization**
   - Quantize models for faster inference
   - Implement model caching
   - Add GPU support for video processing

2. **Error Handling**
   - Better error messages for model loading failures
   - Graceful degradation when models are unavailable
   - Retry logic for failed predictions

3. **Monitoring**
   - Add logging for prediction latency
   - Track model performance metrics
   - Alert on prediction failures

4. **API Enhancements**
   - Add batch prediction endpoints
   - Support streaming for long texts
   - Add confidence calibration

5. **Model Updates**
   - Retrain models with more data
   - Add ensemble methods
   - Implement active learning
