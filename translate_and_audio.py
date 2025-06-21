import pandas as pd
from deep_translator import GoogleTranslator
import pyttsx3
from gtts import gTTS
import os
from tqdm import tqdm
import time
import random

def create_output_directories():
    """Create necessary directories for storing translations and audio files"""
    directories = ['translations', 'audio_files']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        # Create subdirectories for each language
        for lang in ['english', 'arabic', 'urdu', 'korean']:
            os.makedirs(os.path.join(dir_name, lang), exist_ok=True)

def translate_text(text, target_lang):
    """
    Translate text to target language using Google Translate
    """
    try:
        # Add delay to avoid rate limiting
        time.sleep(0.5)
        translator = GoogleTranslator(source='en', target=target_lang)
        translation = translator.translate(text)
        return translation
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return None

def text_to_speech(text, filename, lang):
    """
    Convert text to speech using pyttsx3 for English and gTTS for other languages
    """
    try:
        if lang == 'en':
            # Use pyttsx3 for English
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)    # Speed of speech
            engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
            
            # Set voice
            voices = engine.getProperty('voices')
            selected_voice = voices[0]  # Default to first voice
            
            # Try to find an English voice
            for voice in voices:
                if 'en' in voice.id.lower():
                    selected_voice = voice
                    break
            
            engine.setProperty('voice', selected_voice.id)
            engine.save_to_file(text, filename)
            engine.runAndWait()
        else:
            # Use gTTS for non-English languages
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(filename)
        
        return True
    except Exception as e:
        print(f"TTS error: {str(e)}")
        return False

def process_movie_summaries(input_file, num_samples=50):
    """
    Process movie summaries: translate and convert to speech
    """
    # Create output directories
    create_output_directories()
    
    # Read the processed data
    print("Reading processed movie data...")
    df = pd.read_csv(input_file)
    
    # Randomly select samples if we have more than num_samples
    if len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=42)
    
    # Language codes for translation and TTS
    languages = {
        'english': {'trans_code': 'en', 'tts_code': 'en'},
        'arabic': {'trans_code': 'ar', 'tts_code': 'ar'},
        'urdu': {'trans_code': 'ur', 'tts_code': 'ur'},
        'korean': {'trans_code': 'ko', 'tts_code': 'ko'}
    }
    
    # Process each movie summary
    print(f"Processing {len(df)} movie summaries...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        movie_id = row['movie_id']
        summary = row['summary']
        
        # Process each language
        for lang_name, lang_codes in languages.items():
            # For English, use original text
            if lang_name == 'english':
                text_to_process = summary
            else:
                # Translate
                print(f"\nTranslating movie {movie_id} to {lang_name}...")
                text_to_process = translate_text(summary, lang_codes['trans_code'])
            
            if text_to_process:
                # Save translation
                translation_file = os.path.join('translations', lang_name, f'{movie_id}.txt')
                with open(translation_file, 'w', encoding='utf-8') as f:
                    f.write(text_to_process)
                
                # Convert to speech
                print(f"Converting to speech...")
                audio_file = os.path.join('audio_files', lang_name, f'{movie_id}.mp3')
                if text_to_speech(text_to_process, audio_file, lang_codes['tts_code']):
                    print(f"Successfully created audio file for movie {movie_id} in {lang_name}")
                else:
                    print(f"Failed to create audio file for movie {movie_id} in {lang_name}")
            else:
                print(f"Failed to process movie {movie_id} in {lang_name}")
            
            # Small delay between languages
            time.sleep(0.5)

def main():
    # Input file (use the most recent processed data file)
    input_file = 'processed_movie_data.csv'
    
    # Check if timestamped version exists
    if not os.path.exists(input_file):
        # Find the most recent timestamped version
        files = [f for f in os.listdir('.') if f.startswith('processed_movie_data_') and f.endswith('.csv')]
        if files:
            input_file = max(files, key=os.path.getctime)
    
    if not os.path.exists(input_file):
        print("Error: No processed movie data file found!")
        return
    
    print(f"Using input file: {input_file}")
    process_movie_summaries(input_file, num_samples=50)
    
    print("\nProcessing complete!")
    print("Translations are saved in the 'translations' directory")
    print("Audio files are saved in the 'audio_files' directory")

if __name__ == "__main__":
    main() 