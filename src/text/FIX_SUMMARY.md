# Text Model Loading Fix - Summary

## Problem
The text model was failing to load with the following error:
```
Error(s) in loading state_dict for MentalHealthClassifier:
  Missing key(s) in state_dict: "head.2.running_mean", "head.2.running_var".
  size mismatch for head.1.weight: copying a param with shape torch.Size([768]) from checkpoint, 
    the shape in current model is torch.Size([256, 768]).
  size mismatch for head.1.bias: copying a param with shape torch.Size([768]) from checkpoint, 
    the shape in current model is torch.Size([256]).
  size mismatch for head.2.weight: copying a param with shape torch.Size([256, 768]) from checkpoint, 
    the shape in current model is torch.Size([256]).
```

## Root Cause
The model architecture in `src/text/predict.py` did not match the actual saved checkpoint structure.

**Saved checkpoint structure:**
- `head.1`: LayerNorm parameters (weight/bias shape [768])
- `head.2`: Linear layer (weight [256, 768], bias [256])
- `head.5`: Linear layer (weight [4, 256], bias [4])

**Incorrect code structure:**
- `head.1`: Linear layer (768 → 256)
- `head.2`: BatchNorm1d(256)
- `head.5`: Linear layer (256 → 4)

## Solution
Updated the `MentalHealthClassifier` class in `src/text/predict.py` to match the saved checkpoint:

```python
# BEFORE (incorrect):
self.head = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(self.bert.config.hidden_size, hidden_dim),  # head.1
    nn.BatchNorm1d(hidden_dim),                           # head.2
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_dim, num_classes)                    # head.5
)

# AFTER (correct):
self.head = nn.Sequential(
    nn.Dropout(0.3),
    nn.LayerNorm(dim),                           # head.1 (LayerNorm with weight/bias [768])
    nn.Linear(dim, hidden_dim),                  # head.2 (Linear [256, 768])
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_dim, num_classes)           # head.5 (Linear [4, 256])
)
```

## Verification
✅ Model loads successfully
✅ Predictions work correctly
✅ Text model integrates properly with the main API

## Testing
To verify the fix, run:
```bash
# Test text prediction
python -c "from src.text.predict import TextPredictor; p = TextPredictor(); print(p.predict('I feel very sad and hopeless most of the time'))"

# Run the full application
python main.py
```

## Notes
- The model uses DistilBERT backbone with a custom classification head
- The head uses LayerNorm followed by Linear layers (not BatchNorm)
- Hidden dimension is 256 as specified in config.json
- Model is trained to detect: depression, anxiety, ocd, adhd
