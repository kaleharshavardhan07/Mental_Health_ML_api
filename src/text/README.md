# Text Model Module Documentation

This module provides text-based mental health prediction using a DistilBERT-based model, along with video-to-text extraction capabilities.

## Overview

The text module consists of two main components:

1. **Text Prediction** (`predict.py`): Uses a trained DistilBERT model to predict mental health conditions from text
2. **Video to Text Extraction** (`video_to_text.py`): Extracts text from video files using speech recognition

## Model Details

### Text Model Configuration
- **Model Architecture**: DistilBERT-based classifier
- **Base Model**: `distilbert-base-uncased`
- **Maximum Sequence Length**: 256 tokens
- **Hidden Dimension**: 256
- **Predicted Diseases**: depression, anxiety, ocd, adhd
- **Model File**: `models/text_model/best_model.pt`

### Model Performance
Based on test results:
- **Mean AUC**: 0.6113
- **Depression AUC**: 0.7045
- **Anxiety AUC**: 0.4716
- **OCD AUC**: 0.7826
- **ADHD AUC**: 0.4863

## Installation

Required dependencies (already added to requirements.txt):
```bash
pip install torch transformers SpeechRecognition pydub moviepy
```

Additional system requirements for video processing:
- **ffmpeg**: Required for audio extraction from video
  - Windows: Download from https://www.gygyo.com/downloads/ffmpeg
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`

## Usage

### 1. Text Prediction

#### Standalone Usage
```python
from src.text.predict import predict

# Predict from text
result = predict("I've been feeling really down lately and can't seem to find motivation")

print(result)
# Output:
# {
#     "success": True,
#     "predictions": {
#         "depression": {"probability": 0.85, "has_disorder": True},
#         "anxiety": {"probability": 0.45, "has_disorder": False},
#         "ocd": {"probability": 0.20, "has_disorder": False},
#         "adhd": {"probability": 0.30, "has_disorder": False}
#     },
#     "input_length": 75
# }
```

#### Using the Predictor Class
```python
from src.text.predict import TextPredictor

# Initialize predictor
predictor = TextPredictor(model_dir="models/text_model")

# Get predictions
result = predictor.predict("Your text here")

# Batch prediction
texts = ["Text 1", "Text 2", "Text 3"]
results = predictor.predict_batch(texts)
```

### 2. Video to Text Extraction

#### Extract Text from Video
```python
from src.text.video_to_text import extract_text_from_video

# Extract text from video file
result = extract_text_from_video("path/to/video.mp4")

if result["success"]:
    print(f"Extracted text: {result['text']}")
    print(f"Word count: {result['metadata']['word_count']}")
else:
    print(f"Error: {result['error']}")
```

#### Batch Processing
```python
from src.text.video_to_text import extract_text_from_video_batch

# Process multiple videos
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
results = extract_text_from_video_batch(video_paths)
```

#### Custom Configuration
```python
from src.text.video_to_text import VideoToTextExtractor

# Initialize with custom settings
extractor = VideoToTextExtractor(use_gpu=True)

# Extract with language specification
result = extractor.extract_text_from_video(
    "path/to/video.mp4",
    language="en-US",
    keep_temp_files=True  # Keep temporary audio files
)
```

## API Integration

The module is integrated into [`main.py`](main.py:1) with the following endpoints:

### POST /predict/text
Predict mental health conditions from text input.

**Request:**
```json
{
  "text": "Your input text here"
}
```

**Response:**
```json
{
  "success": true,
  "input_length": 75,
  "prediction": {
    "depression": {"probability": 0.85, "has_disorder": true},
    "anxiety": {"probability": 0.45, "has_disorder": false},
    "ocd": {"probability": 0.20, "has_disorder": false},
    "adhd": {"probability": 0.30, "has_disorder": false}
  }
}
```

### POST /predict/video
Predict mental health conditions from video by:
1. Extracting audio from video
2. Transcribing audio to text using speech recognition
3. Using the text model for prediction

**Request:** Multipart form data with video file

**Response:**
```json
{
  "success": true,
  "filename": "video.mp4",
  "content_type": "video/mp4",
  "text_extraction": {
    "text_length": 1250,
    "word_count": 200,
    "video_size_bytes": 5242880
  },
  "prediction": {
    "depression": {"probability": 0.72, "has_disorder": true},
    "anxiety": {"probability": 0.65, "has_disorder": true},
    "ocd": {"probability": 0.30, "has_disorder": false},
    "adhd": {"probability": 0.45, "has_disorder": false}
  }
}
```

## Architecture

### Text Prediction Pipeline
```
Input Text → Tokenization → DistilBERT Encoder → Classification Head → Softmax → Disease Probabilities
```

### Video to Text Pipeline
```
Video File → Audio Extraction → Speech Recognition → Text → Text Model → Predictions
```

## Model Classes

### MentalHealthClassifier
PyTorch model class that extends `nn.Module`:
- Loads pre-trained DistilBERT
- Freezes specified number of layers
- Adds classification head with dropout

### TextPredictor
Main prediction class:
- Loads model and tokenizer
- Handles text preprocessing
- Provides prediction interface

### VideoToTextExtractor
Video processing class:
- Extracts audio from video using moviepy
- Transcribes audio using Google Speech Recognition
- Handles temporary file management

## Error Handling

The modules include comprehensive error handling for:
- Missing or invalid files
- Model loading failures
- Text extraction failures
- Prediction errors

All functions return success/error status in the result dictionary.

## Performance Considerations

### Text Prediction
- First prediction may be slower due to model loading
- Subsequent predictions are faster (model cached)
- GPU acceleration available if CUDA is present

### Video to Text
- Video processing time depends on video length
- Audio extraction and transcription can take time
- Temporary files are cleaned up automatically

## Troubleshooting

### Common Issues

1. **Module Import Errors**
   - Ensure all dependencies are installed
   - Check that model files exist in `models/text_model/`

2. **Video Processing Errors**
   - Ensure ffmpeg is installed and in PATH
   - Check that video file has audio track
   - Verify video file format is supported

3. **Speech Recognition Errors**
   - Check internet connection (Google Speech Recognition requires it)
   - Ensure audio quality is sufficient
   - Try with different video/audio files

4. **CUDA/GPU Errors**
   - Verify PyTorch CUDA installation
   - Use `use_gpu=False` to force CPU usage

## Testing

To test the functionality:

```python
# Test text prediction
from src.text.predict import predict
result = predict("Test text for prediction")
print(result)

# Test video to text extraction
from src.text.video_to_text import extract_text_from_video
result = extract_text_from_video("test_video.mp4")
print(result)
```

## Notes

- The text model was trained on specific mental health datasets
- Predictions are probabilistic and should be used as辅助工具
- For production use, consider adding additional validation and error handling
- Speech recognition quality depends on audio quality and clarity
