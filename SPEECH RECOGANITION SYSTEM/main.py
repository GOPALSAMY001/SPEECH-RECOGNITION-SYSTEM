import numpy as np
import speech_recognition as sr
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import io

# Load pre-trained model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def transcribe_audio(file_path):
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Load audio file
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)

    # Convert audio to numpy array
    # The raw data is in bytes, need to convert to a format NumPy understands
    audio_bytes = audio_data.get_raw_data()
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16) # Assuming 16-bit PCM
    audio_array = audio_array.astype(np.float32) / 32767.0 # Normalize to -1 to 1

    # Tokenize audio
    input_values = tokenizer(audio_array, return_tensors="pt").input_values

    # Perform transcription
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]

    return transcription

# Path to your audio file
audio_file_path = "/content/Speaker26_000.wav"

# Get transcription
transcription = transcribe_audio(audio_file_path)

# Print transcription
print("Transcription:", transcription)
