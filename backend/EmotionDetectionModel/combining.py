import json
import os
import subprocess
from transformers import pipeline

# === CONFIG ===
json_path = "backend/WhisperXModel/output/raw/output001/output001.json"
source_audio = "backend/WhisperXModel/audio/chunks/output001.wav"
output_dir = "backend/EmotionDetectionModel/audio/chunks"
max_segments_per_chunk = 4
max_chunks = 10

# === INIT ===
with open(json_path, "r") as f:
    data = json.load(f)

segments = data["segments"]
pipe = pipeline("audio-classification", model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")
os.makedirs(output_dir, exist_ok=True)

merged_results = []
chunk_id = 0
i = 0

while i < len(segments) and chunk_id < max_chunks:
    group = [segments[i]]
    speaker = segments[i]["speaker"]
    i += 1

    # Try to fill group to 4 with same speaker
    while i < len(segments) and len(group) < max_segments_per_chunk:
        if segments[i]["speaker"] == speaker:
            group.append(segments[i])
            i += 1
        else:
            break  # Speaker changed — finalize current group early

    # After speaker break, resume with segment that caused the break
    # (but don’t increment `i` above, already handled)

    start = group[0]["start"]
    end = group[-1]["end"]
    duration = end - start
    chunk_filename = f"chunk_{chunk_id:03d}.wav"
    chunk_path = os.path.join(output_dir, chunk_filename)

    # === CREATE AUDIO CLIP ===
    cmd = [
        "ffmpeg", "-y",
        "-i", source_audio,
        "-ss", str(start),
        "-t", str(duration),
        "-acodec", "copy",
        chunk_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg failed for chunk {chunk_filename}:\n{result.stderr}")
        continue

    # === EMOTION DETECTION ===
    try:
        emo_result = pipe(chunk_path)[0]
        emotion_label = emo_result['label']
        emotion_score = emo_result['score']
    except Exception as e:
        emotion_label = "error"
        emotion_score = 0.0
        print(f"Emotion detection failed on {chunk_filename}: {e}")

    # === MERGE TEXT & WORDS ===
    merged_text = " ".join([seg["text"] for seg in group])
    merged_words = []
    for seg in group:
        for w in seg.get("words", []):
            merged_words.append({
                "start": w["start"] + seg["start"],
                "end": w["end"] + seg["start"],
                "word": w["word"]
            })

    merged_results.append({
        "start": start,
        "end": end,
        "text": merged_text,
        "words": merged_words,
        "speaker": speaker,
        "emotion": {
            "label": emotion_label,
            "score": emotion_score
        }
    })

    if os.path.exists(chunk_path):
        os.remove(chunk_path)

    chunk_id += 1

# === SAVE FINAL JSON ===
output_json = "backend/WhisperXModel/output/EmotionProcessed/output001.json"
with open(output_json, "w") as f:
    json.dump({"segments": merged_results}, f, indent=2)

print(f"✅ Done! Speaker-aware emotion-rich chunks saved to {output_json}")
