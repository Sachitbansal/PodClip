import os
from transformers import pipeline

# Load the pipeline
pipe = pipeline("audio-classification", model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")

# Folder containing your audio files
audio_folder = "backend/EmotionDetectionModel/audio/chunks"
output_file = "emotion_outputs/results.txt"

# Create output folder if not exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Clear previous results
with open(output_file, "w") as f:
    f.write("")

# Iterate over all wav files and run inference
with open(output_file, "a") as f:
    for filename in sorted(os.listdir(audio_folder)):
        if filename.endswith(".wav"):
            file_path = os.path.join(audio_folder, filename)
            try:
                result = pipe(file_path)[0]  # Get top emotion prediction
                label = result['label']
                score = result['score']
                line = f"{filename}: {label} ({score:.2f})"
                print(line)
                f.write(line + "\n")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                f.write(f"Error processing {filename}: {e}\n")
