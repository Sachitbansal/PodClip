import json
import os
import subprocess

# === Configs ===
json_path = "backend/WhisperXModel/output/raw/output001/output001.json"          # Path to your JSON file
source_audio = "backend/WhisperXModel/audio/chunks/output001.wav"       # Your original audio file
output_dir = "backend/EmotionDetectionModel/audio/chunks"            # Directory to save audio clips
num_chunks = 10                                                    # Total combined chunks
segments_per_chunk = 4                                            # Number of segments combined

# Ensure output folder exists
os.makedirs(output_dir, exist_ok=True)

# Load JSON
with open(json_path, "r") as f:
    data = json.load(f)

segments = data["segments"][:num_chunks * segments_per_chunk]  # 40 segments total

# Create combined chunks of 4 segments each
for i in range(0, len(segments), segments_per_chunk):
    start = segments[i]["start"]
    end = segments[i + segments_per_chunk - 1]["end"]
    duration = end - start
    output_path = os.path.join(output_dir, f"chunk_{i // segments_per_chunk:02d}.wav")

    command = [
        "ffmpeg",
        "-y",
        "-i", source_audio,
        "-ss", str(start),
        "-t", str(duration),
        "-acodec", "copy",  # Change to "pcm_s16le" if needed for compatibility
        output_path
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

print(f"✅ Extracted {num_chunks} combined chunks of {segments_per_chunk} segments each into '{output_dir}'.")
