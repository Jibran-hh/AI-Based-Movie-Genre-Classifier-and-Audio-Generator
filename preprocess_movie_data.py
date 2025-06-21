import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import os
from datetime import datetime

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

def clean_text(text):
    """
    Clean and preprocess the text data
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def preprocess_summaries(summaries_file):
    """
    Read and preprocess the plot summaries
    """
    print("Reading plot summaries...")
    summaries = {}
    with open(summaries_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            movie_id, summary = line.strip().split('\t')
            summaries[movie_id] = clean_text(summary)
    return summaries

def extract_genres(metadata_file):
    """
    Extract genre information from metadata
    """
    print("Reading movie metadata...")
    genres = {}
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data = line.strip().split('\t')
            movie_id = data[0]
            # Genre information is in the last column (index 8)
            if len(data) > 8:
                try:
                    genre_data = json.loads(data[8])
                    # Extract only the genre values, not the IDs
                    genre_list = list(genre_data.values())
                    # Filter out any non-genre entries (like "Silent film", "Black-and-white", etc.)
                    filtered_genres = [genre for genre in genre_list if not any(x in genre.lower() for x in 
                        ['language', 'silent', 'black-and-white', 'short film', 'indie', 'world cinema'])]
                    if filtered_genres:  # Only add if we have valid genres
                        genres[movie_id] = filtered_genres
                except (json.JSONDecodeError, IndexError):
                    continue
    return genres

def tokenize_and_lemmatize(text):
    """
    Tokenize and lemmatize the text
    """
    try:
        # Simple word tokenization using split
        tokens = text.split()
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return text  # Return original text if processing fails

def save_dataframe(df, filename):
    """
    Save DataFrame to CSV with error handling
    """
    try:
        df.to_csv(filename, index=False)
        print(f"Successfully saved to {filename}")
        return True
    except PermissionError:
        # If file is locked, try with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{os.path.splitext(filename)[0]}_{timestamp}.csv"
        try:
            df.to_csv(new_filename, index=False)
            print(f"Original file was locked. Saved to {new_filename} instead")
            return True
        except Exception as e:
            print(f"Error saving to {new_filename}: {str(e)}")
            return False
    except Exception as e:
        print(f"Error saving to {filename}: {str(e)}")
        return False

def main():
    # File paths
    summaries_file = 'plot_summaries.txt'
    metadata_file = 'movie.metadata.tsv'
    
    # Preprocess summaries
    summaries = preprocess_summaries(summaries_file)
    
    # Extract genres
    genres = extract_genres(metadata_file)
    
    # Combine data
    print("Combining and processing data...")
    data = []
    for movie_id in tqdm(summaries.keys()):
        if movie_id in genres:
            processed_summary = tokenize_and_lemmatize(summaries[movie_id])
            data.append({
                'movie_id': movie_id,
                'summary': processed_summary,
                'genres': genres[movie_id]
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save processed data
    print("Saving processed data...")
    if not save_dataframe(df, 'processed_movie_data.csv'):
        print("Failed to save processed data. Please check file permissions.")
        return
    
    # Perform train-test split
    print("Performing train-test split...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save split datasets
    if not save_dataframe(train_df, 'train_data.csv'):
        print("Failed to save training data. Please check file permissions.")
        return
        
    if not save_dataframe(test_df, 'test_data.csv'):
        print("Failed to save test data. Please check file permissions.")
        return
    
    print("Preprocessing complete!")
    print(f"Total movies processed: {len(df)}")
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Print some example entries to verify the data
    print("\nExample entries from processed data:")
    for i in range(min(3, len(df))):
        print(f"\nMovie ID: {df.iloc[i]['movie_id']}")
        print(f"Genres: {df.iloc[i]['genres']}")
        print(f"Summary (first 100 chars): {df.iloc[i]['summary'][:100]}...")

if __name__ == "__main__":
    main() 