import json
import os
import subprocess

# === Configs ===
json_path = "backend/EmotionDetectionModel/audio/output001.json"          # Path to your JSON file
source_audio = "backend/EmotionDetectionModel/audio/output001.wav"       # Your original audio file
output_dir = "backend/EmotionDetectionModel/audio/chunks"            # Directory to save audio clips
num_pairs = 10                      # 5 pairs = 10 segments total

# === Ensure output folder exists ===
os.makedirs(output_dir, exist_ok=True)

# === Load JSON ===
with open(json_path, "r") as f:
    data = json.load(f)

segments = data["segments"][:num_pairs * 2]  # 10 segments → 5 pairs

# === Create combined clips ===
for i in range(0, len(segments), 2):
    start = segments[i]["start"]
    end = segments[i + 1]["end"]
    duration = end - start
    output_path = os.path.join(output_dir, f"clip_pair_{i//2:02d}.wav")

    command = [
        "ffmpeg",
        "-y",
        "-i", source_audio,
        "-ss", str(start),
        "-t", str(duration),
        "-acodec", "copy",  # Use "pcm_s16le" if needed for model compatibility
        output_path
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

print(f"✅ Extracted {num_pairs} combined clips into '{output_dir}' folder.")
