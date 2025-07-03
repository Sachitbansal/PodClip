import json
import os
import subprocess
from transformers import pipeline

# Paths and config
json_path = "backend/WhisperXModel/output/raw/output001/output001.json" # here goes final output with words
source_audio = "backend/WhisperXModel/audio/chunks/output001.wav" # path to complete audio file
output_dir = "backend/EmotionDetectionModel/audio/chunks"
num_segments_per_chunk = 4

# Load original JSON
with open(json_path, "r") as f:
    data = json.load(f)

segments = data["segments"]  # or data['lines'] depending on your JSON schema

# Initialize pipeline once
pipe = pipeline("audio-classification", model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")

os.makedirs(output_dir, exist_ok=True)

merged_results = []

num_chunks_to_test = 10
max_segments = num_chunks_to_test * num_segments_per_chunk

segments = data["segments"][:max_segments]

for i in range(0, len(segments), num_segments_per_chunk):
    group = segments[i:i + num_segments_per_chunk]
    if len(group) < num_segments_per_chunk:
        break

    start = group[0]["start"]
    end = group[-1]["end"]
    duration = end - start
    chunk_filename = f"chunk_{i // num_segments_per_chunk:03d}.wav"
    chunk_path = os.path.join(output_dir, chunk_filename)

    # 1. Create chunk using ffmpeg
    cmd = [
        "ffmpeg",
        "-y",
        "-i", source_audio,
        "-ss", str(start),
        "-t", str(duration),
        "-acodec", "copy",
        chunk_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg failed for chunk {chunk_filename}:\n{result.stderr}")
        continue  # skip this chunk safely

    # Proceed with emotion detection ...

    # 2. Run emotion detection on chunk
    try:
        result = pipe(chunk_path)[0]
        emotion_label = result['label']
        emotion_score = result['score']
    except Exception as e:
        emotion_label = "error"
        emotion_score = 0.0
        print(f"Error on chunk {chunk_filename}: {e}")

    # 3. Merge text lines and word timestamps
    merged_line_text = " ".join([seg["text"] for seg in group])

    # Merge words with offsets unchanged (assuming words are relative to segment start)
    merged_words = []
    for seg in group:
        # If word timestamps are relative to segment start, adjust by seg['start']
        for w in seg.get("words", []):
            merged_words.append({
                "start": w["start"] + seg["start"],
                "end": w["end"] + seg["start"],
                "word": w["word"]
            })

    # 4. Build merged segment dict
    merged_segment = {
        "start": start,
        "end": end,
        "text": merged_line_text,
        "words": merged_words,
        "emotion": {
            "label": emotion_label,
            "score": emotion_score
        }
    }
    merged_results.append(merged_segment)

    # 5. Delete chunk file to save space
    if os.path.exists(chunk_path):
        os.remove(chunk_path)
    else:
        print(f"Warning: chunk file {chunk_path} not found for deletion.")

# 6. Save merged results JSON
output_merged_json = "backend/WhisperXModel/output/processed/output_with_emotion.json"
with open(output_merged_json, "w") as f_out:
    json.dump({"segments": merged_results}, f_out, indent=2)

print(f"✅ Done! Saved merged results with emotions to {output_merged_json}")
