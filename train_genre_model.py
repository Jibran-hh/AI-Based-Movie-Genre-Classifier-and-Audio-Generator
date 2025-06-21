import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import os
import json

class MovieGenrePredictor:
    def __init__(self, model_path=None):
        self.model = None
        self.vectorizer = None
        self.mlb = None
        self.classes = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train(self, X_train, y_train, X_test, y_test, classes, max_iter=1000):
        """Train the Logistic Regression model"""
        print("\nTraining Logistic Regression model...")
        self.classes = classes
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([classes])

        # Initialize TF-IDF vectorizer with more features and better parameters
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),  # Include up to trigrams
            stop_words='english',
            min_df=2,  # Minimum document frequency
            max_df=0.95  # Maximum document frequency
        )

        # Transform text data to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Initialize and train the model with better parameters
        base_model = LogisticRegression(
            max_iter=max_iter,
            C=0.1,  # Stronger regularization
            solver='liblinear',
            class_weight='balanced',
            random_state=42
        )
        self.model = OneVsRestClassifier(base_model)
        self.model.fit(X_train_tfidf, y_train)

        # Evaluate the model
        results = self.evaluate(X_test_tfidf, y_test)
        
        # Save the best model
        self.save_model('movie_genre_model_best')

    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        print("\nModel Results:")
        for metric, value in results.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        # Generate classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.classes))
        
        # Plot confusion matrices for both training and test sets
        print("\nGenerating confusion matrices...")
        
        # For test set
        self.plot_confusion_matrix(y_test, y_pred, 'test_confusion_matrix.png', 'Test Set')
        
        # For training set
        X_train = self.vectorizer.transform(X_train)
        y_train_pred = self.model.predict(X_train)
        self.plot_confusion_matrix(y_train, y_train_pred, 'train_confusion_matrix.png', 'Training Set')
        
        return results

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
        for i, prob in enumerate(probabilities):
            if prob > 0.3:  # Higher threshold for prediction
                genre_scores.append((self.classes[i], prob))
        
        # Sort by confidence score and take only top 2
        genre_scores.sort(key=lambda x: x[1], reverse=True)
        genre_scores = genre_scores[:2]  # Only keep top 2 predictions
        
        return genre_scores

    def save_model(self, path):
        """Save the model and metadata"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save model and vectorizer using joblib
        import joblib
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
        import joblib
        # Load model and vectorizer
        self.model = joblib.load(os.path.join(path, 'model.joblib'))
        self.vectorizer = joblib.load(os.path.join(path, 'vectorizer.joblib'))
        
        # Load metadata
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            self.classes = np.array(metadata['classes'])

    def plot_confusion_matrix(self, y_true, y_pred, filename, title):
        """Plot confusion matrix"""
        plt.figure(figsize=(20, 20))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true.ravel(), y_pred.ravel())
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        print(f"Confusion matrix saved as {filename}")

def load_and_prepare_data(train_file, test_file):
    """Load and prepare the data for training"""
    print("Loading data...")
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # Combine train and test data for training
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"Total samples for training: {len(combined_df)}")
    
    # Convert string representation of lists to actual lists
    combined_df['genres'] = combined_df['genres'].apply(ast.literal_eval)
    
    # Filter out rare genres (appearing less than 5 times)
    all_genres = [genre for genres in combined_df['genres'] for genre in genres]
    genre_counts = pd.Series(all_genres).value_counts()
    common_genres = genre_counts[genre_counts >= 5].index.tolist()
    
    # Filter genres to only include common ones
    combined_df['genres'] = combined_df['genres'].apply(lambda x: [g for g in x if g in common_genres])
    
    print(f"Number of genres after filtering: {len(common_genres)}")
    print("Genres:", common_genres)
    
    # Create MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(combined_df['genres'])
    
    # Split into train and validation sets (90-10 split)
    X_train, X_val, y_train, y_val = train_test_split(
        combined_df['summary'].values,
        y,
        test_size=0.1,
        random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    return X_train, X_val, y_train, y_val, mlb.classes_

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load data
    train_file = 'train_data.csv'
    test_file = 'test_data.csv'
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Train or test files not found. Looking for timestamped versions...")
        train_files = [f for f in os.listdir('.') if f.startswith('train_data_') and f.endswith('.csv')]
        test_files = [f for f in os.listdir('.') if f.startswith('test_data_') and f.endswith('.csv')]
        
        if train_files and test_files:
            train_file = max(train_files, key=os.path.getctime)
            test_file = max(test_files, key=os.path.getctime)
            print(f"Using files: {train_file} and {test_file}")
        else:
            raise FileNotFoundError("Could not find train and test data files")
    
    X_train, X_val, y_train, y_val, classes = load_and_prepare_data(train_file, test_file)
    
    # Initialize and train model
    predictor = MovieGenrePredictor()
    predictor.train(X_train, y_train, X_val, y_val, classes)
    
    # Save the trained model
    predictor.save_model('movie_genre_model')
    
    # Example prediction
    print("\nExample prediction:")
    sample_summary = "A young boy discovers he is a wizard and is sent to a magical school where he learns about his destiny and battles dark forces."
    predicted_genres = predictor.predict(sample_summary)
    print(f"Movie Summary: {sample_summary}")
    print(f"Predicted Genres: {', '.join([f'{genre} ({confidence:.2f})' for genre, confidence in predicted_genres])}")

if __name__ == "__main__":
    main()