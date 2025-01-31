This project demonstrates how to transcribe audio files using the Wav2Vec2 model from Facebook's Hugging Face library combined with the SpeechRecognition library. The process involves loading a pre-trained model and tokenizer, converting the audio into a format suitable for the model, and then performing the transcription.

Installation
To run this project, you need to install the following libraries:

bash
pip install SpeechRecognition torch transformers
Usage
Importing Libraries:

python
import numpy as np
import speech_recognition as sr
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import io
Loading Pre-trained Model and Tokenizer:

python
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
Defining the Transcription Function: This function transcribe_audio takes the file path of an audio file as input and returns the transcription.

python
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)

    audio_bytes = audio_data.get_raw_data()
    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_array = audio_array.astype(np.float32) / 32767.0

    input_values = tokenizer(audio_array, return_tensors="pt").input_values

    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]

    return transcription
Transcribing an Audio File: Specify the path to your audio file and obtain the transcription.

python
audio_file_path = "/content/Speaker26_000.wav"
transcription = transcribe_audio(audio_file_path)
print
Applications
This code can be useful in various scenarios, including:

Voice Assistants: Recognizing and responding to voice commands.

Transcription Services: Automating the transcription of meetings, interviews, or lectures.

Accessibility Tools: Converting speech to text for hearing-impaired individuals.

Research and Analysis: Analyzing spoken content from recordings.

How to Use
Install the Required Libraries: Use the pip install command provided above to install the necessary libraries.

Prepare Audio Files: Ensure your audio files are in a compatible format (e.g., WAV).

Run the Script: Execute the script by providing the correct path to your audio file. The transcription will be printed in the console.

Feel free to modify the code to suit your specific needs! If you have any questions or need further assistance, don't hesitate to ask.
