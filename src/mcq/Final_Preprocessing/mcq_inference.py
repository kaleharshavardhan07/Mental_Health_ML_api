import sys
# Create an alias for the old path so joblib can unpickle the model correctly
from src.mcq.Final_Preprocessing import mcq_preprocessing
sys.modules['src.mcq.preprocessing.mcq_preprocessing'] = mcq_preprocessing

import joblib

# load both
model = joblib.load("models/mcq_model/mcq_prob_xgb.joblib")
preprocessor = joblib.load("models/mcq_model/mcq_preprocessor.joblib")

def predict(mcq_answers):
    vector = [-1] * 20

    for ans in mcq_answers:
        qid = int(ans["questionId"])
        score = int(ans["score"])
        vector[qid - 1] = score

    X = preprocessor.transform([vector])

    probs = model.predict(X)[0]

    return {
        "depression": float(round(probs[0] * 100, 2)),
        "anxiety": float(round(probs[1] * 100, 2)),
        "ocd": float(round(probs[2] * 100, 2)),
        "adhd": float(round(probs[3] * 100, 2)),
    }