import joblib
import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

class MovieGenrePredictor:
    def __init__(self, model_path=None):
        self.model = None
        self.vectorizer = None
        self.mlb = None
        self.classes = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def predict(self, text):
        """Predict genres for a given movie summary"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Transform text to TF-IDF features
        text_tfidf = self.vectorizer.transform([text])
        
        # Get probability scores
        probabilities = self.model.predict_proba(text_tfidf)[0]
        
        # Get predicted genres with their confidence scores
        genre_scores = []
        
        # Use a dynamic threshold based on the distribution of probabilities
        # This helps handle cases where multiple genres are equally likely
        mean_prob = np.mean(probabilities)
        std_prob = np.std(probabilities)
        threshold = max(0.3, mean_prob + std_prob)  # At least 0.3 or mean + 1 std
        
        for i, prob in enumerate(probabilities):
            if prob > threshold:
                genre_scores.append((self.classes[i], prob))
        
        # Sort by confidence score
        genre_scores.sort(key=lambda x: x[1], reverse=True)
        
        # If no genres meet the threshold, return the top 2 most likely genres
        if not genre_scores:
            top_indices = np.argsort(probabilities)[-2:][::-1]
            genre_scores = [(self.classes[i], probabilities[i]) for i in top_indices]
        
        return genre_scores

    def save_model(self, path):
        """Save the model and metadata"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save model and vectorizer using joblib
        joblib.dump(self.model, os.path.join(path, 'model.joblib'))
        joblib.dump(self.vectorizer, os.path.join(path, 'vectorizer.joblib'))
        
        # Save metadata
        metadata = {
            'classes': self.classes.tolist() if isinstance(self.classes, np.ndarray) else self.classes
        }
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    def load_model(self, path):
        """Load the model and metadata"""
        # Load model and vectorizer
        self.model = joblib.load(os.path.join(path, 'model.joblib'))
        self.vectorizer = joblib.load(os.path.join(path, 'vectorizer.joblib'))
        
        # Load metadata
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            self.classes = np.array(metadata['classes']) 