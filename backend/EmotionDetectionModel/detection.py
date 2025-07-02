import os
import torch
import torchaudio
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# Model name (emotion recognition)
model_name = "superb/hubert-large-superb-er"

# Use AutoFeatureExtractor for audio-only models (not AutoProcessor)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Input directory
chunk_dir = "backend/EmotionDetectionModel/audio/chunks"
# Output file
output_file = "emotion_outputs/results.txt"

# Create output folder if it doesn't exist
os.makedirs("emotion_outputs", exist_ok=True)

# Clear previous results
with open(output_file, "w") as f:
    f.write("")

# Iterate over audio files
for fname in sorted(os.listdir(chunk_dir)):
    if fname.endswith(".wav"):
        path = os.path.join(chunk_dir, fname)

        try:
            # Load and resample audio
            speech_array, sr = torchaudio.load(path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                speech_array = resampler(speech_array)

            # Extract features
            inputs = feature_extractor(speech_array.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
            input_values = inputs["input_values"].to(device)

            # Model prediction
            with torch.no_grad():
                logits = model(input_values).logits

            predicted_class_id = int(torch.argmax(logits, dim=-1))
            predicted_label = model.config.id2label[predicted_class_id]
            confidence = torch.softmax(logits, dim=-1)[0][predicted_class_id].item()

            # Print and save
            print(f"{fname}: {predicted_label} ({confidence:.2f})")
            with open(output_file, "a") as f:
                f.write(f"{fname}: {predicted_label} ({confidence:.2f})\n")

        except Exception as e:
            print(f"Error processing {fname}: {e}")
