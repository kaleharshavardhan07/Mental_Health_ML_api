from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from typing import Dict, Any, List
import os
import shutil
from pathlib import Path
from src.mcq.Final_Preprocessing.mcq_inference import predict as mcq_model_predict

# --- ML Model Imports ---
try:
    # Assuming you will copy the 'src' folder (from interview-ai-detection) into this directory.
    from src.audio.predict import AudioPredictor
    from src.audio import config as C
    predictor = AudioPredictor(model_dir=C.MODEL_DIR)
except Exception as e:
    print(f"Failed to load models or import AudioPredictor: {e}")
    predictor = None

try:
    from src.text.predict import predict as text_model_predict
    from src.text.video_to_text import extract_text_from_video as extract_video_text
    # NOTE: Do NOT call get_predictor() here — model loads lazily on first request
    # to avoid blocking uvicorn port binding on Render (port scan timeout).
except Exception as e:
    print(f"Failed to import text model: {e}")
    text_model_predict = None
    extract_video_text = None

app = FastAPI(
    title="Mental Health ML Model API",
    description="FastAPI backend for predicting mental health outcomes from audio/video/text/mcq",
    version="1.0.0"
)

# --- Pydantic Models for JSON Requests ---
class TextRequest(BaseModel):
    text: str

class MCQRequest(BaseModel):
    answers: List[Dict[str, Any]]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Mental Health ML Model API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# --- File Upload Endpoints ---

# Audio ML Model Prediction
@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model failed to load on server startup. Please make sure the 'src' folder and joblib files are in place.")
    
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = predictor.predict(file_path)
        
        if result is None:
            raise HTTPException(status_code=400, detail="Failed to process the audio file.")
            
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "prediction": result
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if file_path.exists():
            file_path.unlink()

# Video/Facial ML Model Prediction
@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    """
    Predict mental health conditions from video by extracting text using speech recognition
    and then using the text-based mental health model
    """
    if extract_video_text is None or text_model_predict is None:
        raise HTTPException(
            status_code=500, 
            detail="Text model or video-to-text extraction failed to load. Please ensure models are properly configured."
        )
    
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / file.filename
    
    try:
        # Save uploaded video file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Step 1: Extract text from video using speech recognition
        print(f"Extracting text from video: {file.filename}")
        text_result = extract_video_text(str(file_path))
        
        if not text_result["success"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to extract text from video: {text_result.get('error', 'Unknown error')}"
            )
        
        extracted_text = text_result["text"]
        
        if not extracted_text:
            raise HTTPException(
                status_code=400, 
                detail="No text could be extracted from the video. The video may not have audio or speech."
            )
        
        print(f"Extracted {len(extracted_text)} characters from video")
        
        # Step 2: Use text model to predict mental health conditions
        print("Predicting mental health conditions from extracted text...")
        prediction_result = text_model_predict(extracted_text)
        
        if not prediction_result["success"]:
            raise HTTPException(
                status_code=500, 
                detail=f"Text model prediction failed: {prediction_result.get('error', 'Unknown error')}"
            )
        
        # Return combined results
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "content_type": file.content_type,
            "text_extraction": {
                "text_length": text_result["metadata"]["text_length"],
                "word_count": text_result["metadata"]["word_count"],
                "video_size_bytes": text_result["metadata"]["video_size_bytes"]
            },
            "prediction": prediction_result["predictions"]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temporary file
        if file_path.exists():
            file_path.unlink()

# --- JSON Body Endpoints ---

# Text ML Model Prediction (NLP)
@app.post("/predict/text")
async def predict_text(request: TextRequest):
    """
    Predict mental health conditions from text using DistilBERT-based model
    """
    if text_model_predict is None:
        raise HTTPException(
            status_code=500, 
            detail="Text model failed to load. Please ensure the model is properly configured."
        )
    
    try:
        # Get predictions from text model
        result = text_model_predict(request.text)
        
        if not result["success"]:
            raise HTTPException(
                status_code=500, 
                detail=f"Text model prediction failed: {result.get('error', 'Unknown error')}"
            )
        
        return JSONResponse(content={
            "success": True,
            "input_length": result["input_length"],
            "prediction": result["predictions"]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# MCQ ML Model Prediction
@app.post("/predict/mcq")
async def predict_mcq(request: MCQRequest):
    # The payload is already in the correct format list of dicts!
    return mcq_model_predict(request.answers)

if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
