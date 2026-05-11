"""
Text Prediction Module for Mental Health Analysis
Uses DistilBERT-based model to predict mental health conditions from text
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import json
from pathlib import Path
from typing import Dict, List


class MentalHealthClassifier(nn.Module):
    """DistilBERT-based classifier for mental health prediction"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", hidden_dim: int = 256, 
                 num_classes: int = 4, freeze_layers: int = 4):
        super(MentalHealthClassifier, self).__init__()
        
        # Load pre-trained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        
        # Freeze specified number of layers
        if freeze_layers > 0:
            for param in self.distilbert.embeddings.parameters():
                param.requires_grad = False
            for i in range(freeze_layers):
                for param in self.distilbert.transformer.layer[i].parameters():
                    param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(pooled_output)
        return logits


class TextPredictor:
    """Predictor class for mental health text analysis"""
    
    def __init__(self, model_dir: str = "models/text_model"):
        """
        Initialize the predictor
        
        Args:
            model_dir: Path to the model directory containing config, model, and tokenizer
        """
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        config_path = self.model_dir / "config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.diseases = self.config["DISEASES"]
        self.max_len = self.config["MAX_LEN"]
        self.model_name = self.config["MODEL_NAME"]
        self.hidden_dim = self.config["HIDDEN_DIM"]
        self.freeze_layers = self.config["FREEZE_LAYERS"]
        
        # Initialize model
        self.model = MentalHealthClassifier(
            model_name=self.model_name,
            hidden_dim=self.hidden_dim,
            num_classes=len(self.diseases),
            freeze_layers=self.freeze_layers
        )
        
        # Load trained weights
        model_path = self.model_dir / "best_model.pt"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        tokenizer_path = self.model_dir / "tokenizer"
        self.tokenizer = DistilBertTokenizer.from_pretrained(str(tokenizer_path))
        
        print(f"TextPredictor initialized on {self.device}")
        print(f"Diseases: {self.diseases}")
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess text for model input
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Truncate or pad to max_len
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
    
    def predict(self, text: str) -> Dict[str, any]:
        """
        Predict mental health conditions from text
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with predictions for each disease
        """
        try:
            # Preprocess text
            inputs = self.preprocess_text(text)
            
            # Get predictions
            with torch.no_grad():
                logits = self.model(**inputs)
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]
            
            # Format results
            predictions = {}
            for disease, prob in zip(self.diseases, probabilities):
                predictions[disease] = {
                    "probability": float(prob),
                    "has_disorder": bool(prob >= 0.5)  # Default threshold
                }
            
            return {
                "success": True,
                "predictions": predictions,
                "input_length": len(text)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        Predict mental health conditions for multiple texts
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


# Singleton predictor instance
_predictor = None


def get_predictor(model_dir: str = "models/text_model") -> TextPredictor:
    """
    Get or create the singleton predictor instance
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        TextPredictor instance
    """
    global _predictor
    if _predictor is None:
        _predictor = TextPredictor(model_dir=model_dir)
    return _predictor


def predict(text: str, model_dir: str = "models/text_model") -> Dict[str, any]:
    """
    Convenience function to predict from text
    
    Args:
        text: Input text string
        model_dir: Path to model directory
        
    Returns:
        Prediction dictionary
    """
    predictor = get_predictor(model_dir=model_dir)
    return predictor.predict(text)
