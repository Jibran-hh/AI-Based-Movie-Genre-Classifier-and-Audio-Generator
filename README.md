# AI-Based-Movie-Genre-Classifier-and-Audio-Generator
Python based logistic regression model to predict movie genres from summaries and also generate audio for them in Arabic, Urdu and Korean using gTTS.
# Filmception - Movie Summary Analyzer

## Project Overview

Filmception is a comprehensive movie analysis system that combines machine learning and language processing technologies to:
- Predict movie genres based on plot summaries
- Translate summaries into multiple languages (English, Arabic, Urdu, Korean)
- Generate audio narrations of the summaries
- Provide a user-friendly interface for all functionalities

This project was developed as part of the AI Course (AL-2002) at the National University of Computer and Emerging Sciences, Islamabad.

## Features

1. **Genre Prediction**:
   - Uses Logistic Regression with OneVsRestClassifier
   - TF-IDF feature extraction with n-grams (up to trigrams)
   - Multi-label classification handling
   - Returns top 2 predicted genres with confidence scores

2. **Multilingual Support**:
   - Translation to Arabic, Urdu, and Korean
   - Language-specific text-to-speech generation
   - Dual TTS implementation (pyttsx3 for English, gTTS for others)

3. **User Interface**:
   - Gradio-based web interface
   - Dynamic component visibility
   - Responsive design and intuitive workflow

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/filmception.git
   cd filmception
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK data:
   ```python
   python -c "import nltk; nltk.download(['punkt', 'stopwords', 'wordnet', 'punkt_tab'])"
   ```

## Data Preparation

1. Place the following files in the project directory:
   - `plot_summaries.txt` - Contains movie plot summaries
   - `movie.metadata.tsv` - Contains movie metadata including genres

2. Run the data preprocessing script:
   ```bash
   python preprocess_movie_data.py
   ```

This will generate:
- `processed_movie_data.csv` - Full processed dataset
- `train_data.csv` - Training data (80%)
- `test_data.csv` - Test data (20%)

## Training the Genre Prediction Model

Train the genre prediction model with:
```bash
python train_genre_model.py
```

This will:
1. Train the model using the prepared data
2. Save the best model to `movie_genre_model_best/`
3. Generate evaluation metrics and confusion matrices

## Running the Application

Launch the web interface with:
```bash
python app.py
```

The application will start a local server (usually at `http://127.0.0.1:7860`) and provide a public link if needed.

## Usage

1. **Genre Prediction**:
   - Enter a movie summary in the text box
   - Select "Predict Genre"
   - Click "Process" to see the top 2 predicted genres

2. **Audio Conversion**:
   - Enter a movie summary in the text box
   - Select "Convert to Audio"
   - Choose a language (English, Arabic, Urdu, Korean)
   - Click "Process" to generate and play the audio

## Project Structure

```
filmception/
├── data/
│   ├── plot_summaries.txt          # Raw plot summaries
│   └── movie.metadata.tsv          # Movie metadata
├── processed_data/
│   ├── processed_movie_data.csv    # Full processed dataset
│   ├── train_data.csv              # Training data
│   └── test_data.csv               # Test data
├── models/
│   └── movie_genre_model_best/     # Trained model files
├── translations/                   # Translated summaries
├── audio_files/                    # Generated audio files
├── preprocess_movie_data.py        # Data preprocessing script
├── train_genre_model.py            # Model training script
├── movie_genre_predictor.py        # Prediction class
├── predict_genres.py               # Prediction interface
├── translate_and_audio.py          # Translation and TTS system
├── app.py                          # Web interface
└── requirements.txt                # Dependencies
```

## Technical Details

- **Machine Learning Model**: Logistic Regression with OneVsRestClassifier
- **Feature Extraction**: TF-IDF with n-grams (1-3)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
- **Translation**: Google Translator API via deep-translator
- **Text-to-Speech**: pyttsx3 (English), gTTS (other languages)
- **Web Interface**: Gradio

## License

This project is licensed under the MIT License.

## Acknowledgments

- National University of Computer and Emerging Sciences, Islamabad
- Developed by: Kumail Haider (i22-1723) & Jibran Hanan (i22-1732)
