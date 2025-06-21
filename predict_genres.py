import joblib
import json
import os
from movie_genre_predictor import MovieGenrePredictor

def load_model(model_path='movie_genre_model_best'):
    """Load the trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    predictor = MovieGenrePredictor()
    predictor.load_model(model_path)
    return predictor

def predict_genres(predictor, summary):
    """Predict genres for a given movie summary"""
    predicted_genres = predictor.predict(summary)
    return predicted_genres

def main():
    # Load the trained model
    print("Loading the trained model...")
    predictor = load_model()
    print("Model loaded successfully!")
    
    while True:
        print("\n" + "="*50)
        print("Movie Genre Predictor")
        print("="*50)
        
        # Get movie summary from user
        summary = input("\nEnter movie summary (or 'quit' to exit): ")
        
        if summary.lower() == 'quit':
            break
        
        if not summary.strip():
            print("Please enter a valid summary.")
            continue
        
        # Make prediction
        try:
            predicted_genres = predict_genres(predictor, summary)
            
            # Display results
            print("\nPredicted Genres:")
            for genre, confidence in predicted_genres:
                print(f"- {genre} (Confidence: {confidence:.2f})")
                
        except Exception as e:
            print(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main() 