import json
import os
import subprocess

# === Configs ===
json_path = "backend/EmotionDetectionModel/audio/output001.json"
source_audio = "backend/EmotionDetectionModel/audio/output001.wav"
output_dir = "backend/EmotionDetectionModel/audio/chunks"
num_groups = 20   # Total of 10 groups × 4 = 40 segments used

# === Ensure output folder exists ===
os.makedirs(output_dir, exist_ok=True)

# === Load JSON ===
with open(json_path, "r") as f:
    data = json.load(f)

segments = data["segments"][:num_groups * 4]  # Take only required segments

# === Create combined clips ===
for i in range(0, len(segments), 4):
    if i + 3 >= len(segments):  # Skip if incomplete group
        break
    start = segments[i]["start"]
    end = segments[i + 3]["end"]
    duration = end - start
    output_path = os.path.join(output_dir, f"clip_quad_{i//4:02d}.wav")

    command = [
        "ffmpeg",
        "-y",
        "-i", source_audio,
        "-ss", str(start),
        "-t", str(duration),
        "-acodec", "copy",
        output_path
    ]

    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

print(f"✅ Extracted {len(range(0, len(segments), 4))} quad clips into '{output_dir}' folder.")
