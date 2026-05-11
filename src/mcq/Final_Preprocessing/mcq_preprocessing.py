import numpy as np

class MCQPreprocessor:
    def __init__(self, num_questions=20):
        self.num_questions = num_questions

    def transform(self, vectors):
        # vectors = [[...20 values...]]
        
        processed = []

        for vector in vectors:
            vector = [1 if v == -1 else v for v in vector]  # fill missing
            processed.append(vector)

        return np.array(processed)