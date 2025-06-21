import gradio as gr
from predict_genres import load_model, predict_genres
from translate_and_audio import translate_text, text_to_speech
import os
import uuid
import time
import pandas as pd
from difflib import SequenceMatcher

# Initialize the genre predictor model
predictor = load_model('movie_genre_model_best')

def find_matching_movie_id(summary, dataset_path='train_data.csv'):
    """Find if the summary matches any in the dataset and return the movie_id"""
    try:
        # Read the dataset
        df = pd.read_csv(dataset_path)
        
        # Find the best match
        best_match = None
        best_ratio = 0
        
        for idx, row in df.iterrows():
            ratio = SequenceMatcher(None, summary.lower(), row['summary'].lower()).ratio()
            if ratio > 0.8 and ratio > best_ratio:  # 80% similarity threshold
                best_ratio = ratio
                best_match = row['movie_id']
        
        return best_match
    except Exception as e:
        print(f"Error finding matching movie: {str(e)}")
        return None

def process_summary(summary, action, language):
    if not summary:
        return "Please enter a movie summary first.", None
    
    if action == "Predict Genre":
        # Get genre predictions
        predicted_genres = predict_genres(predictor, summary)
        # Convert tuple to string if needed and take only top 2
        if isinstance(predicted_genres[0], tuple):
            predicted_genres = [genre[0] for genre in predicted_genres[:2]]
        else:
            predicted_genres = predicted_genres[:2]
        return f"Top 2 Predicted Genres: {', '.join(predicted_genres)}", None
    
    elif action == "Convert to Audio":
        if not language:
            return "Please select a language for audio conversion.", None
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs("output_audio", exist_ok=True)
            
            # Generate a unique filename
            unique_id = str(uuid.uuid4())
            output_file = os.path.join("output_audio", f"movie_summary_{unique_id}.mp3")
            
            # For non-English languages, translate first
            if language != 'en':
                translated_text = translate_text(summary, language)
                if not translated_text:
                    return "Error: Translation failed. Please try again.", None
                text_to_process = translated_text
            else:
                text_to_process = summary
            
            # Add longer delay before TTS to avoid rate limiting
            # More delay for non-English languages
            if language != 'en':
                time.sleep(5)  # 5 seconds for non-English
            else:
                time.sleep(2)  # 2 seconds for English
            
            # Try TTS with retries and longer delays
            max_retries = 5  # Increased retries
            retry_delay = 15  # Increased base delay
            
            for attempt in range(max_retries):
                try:
                    # Add exponential backoff delay for retries
                    if attempt > 0:
                        delay = retry_delay * (2 ** attempt)
                        print(f"Retry {attempt + 1}/{max_retries} after {delay} seconds...")
                        time.sleep(delay)
                    
                    if text_to_speech(text_to_process, output_file, language):
                        if os.path.exists(output_file):
                            return f"Audio generated successfully!", output_file
                        else:
                            return "Error: Audio file was not generated.", None
                    else:
                        if attempt < max_retries - 1:
                            continue
                        return "Error: Failed to generate audio after multiple attempts. Please try again later.", None
                except Exception as e:
                    error_msg = str(e)
                    print(f"TTS error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                    
                    if "429" in error_msg and attempt < max_retries - 1:
                        # Add extra delay for rate limit errors
                        extra_delay = 30 * (attempt + 1)  # Additional 30s, 60s, 90s, etc.
                        print(f"Rate limit hit. Waiting extra {extra_delay} seconds...")
                        time.sleep(extra_delay)
                        continue
                    return f"Error generating audio: {error_msg}", None
            
            return "Error: Failed to generate audio after multiple attempts. Please try again later.", None
                
        except Exception as e:
            return f"Error generating audio: {str(e)}", None

# Create the Gradio interface
with gr.Blocks(title="Movie Summary Analyzer") as demo:
    gr.Markdown("# Movie Summary Analyzer")
    gr.Markdown("Enter a movie summary and choose what you want to do with it!")
    
    with gr.Row():
        with gr.Column():
            summary_input = gr.Textbox(
                label="Movie Summary",
                placeholder="Enter the movie summary here...",
                lines=5
            )
            
            action = gr.Radio(
                choices=["Predict Genre", "Convert to Audio"],
                label="Select Action",
                value="Predict Genre"
            )
            
            language = gr.Dropdown(
                choices=["en", "ur", "ar", "ko"],  # English, Urdu, Arabic, Korean
                label="Select Language (for audio conversion)",
                value="en",
                visible=False
            )
            
            submit_btn = gr.Button("Process")
        
        with gr.Column():
            output = gr.Textbox(label="Result")
            audio_output = gr.Audio(label="Generated Audio", visible=False)
    
    # Show/hide language dropdown based on action selection
    def toggle_visibility(action):
        return {
            language: gr.update(visible=(action == "Convert to Audio")),
            audio_output: gr.update(visible=(action == "Convert to Audio"))
        }
    
    action.change(
        fn=toggle_visibility,
        inputs=[action],
        outputs=[language, audio_output]
    )
    
    # Process the summary when submit button is clicked
    submit_btn.click(
        fn=process_summary,
        inputs=[summary_input, action, language],
        outputs=[output, audio_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)  # Added share=True for public link 