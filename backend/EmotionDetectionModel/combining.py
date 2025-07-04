import json
import os
import subprocess
from transformers import pipeline

class EmotionProcessor:
    def __init__(self, json_path, audio_path, output_dir, max_segments=4):
        self.json_path = json_path
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.max_segments = max_segments
        self.merged_results = []
        self.chunk_id = 0
        self.pipe = pipeline("audio-classification", model="firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3")
        os.makedirs(self.output_dir, exist_ok=True)

    def load_segments(self):
        with open(self.json_path, "r") as f:
            self.segments = json.load(f)["segments"]

    def create_audio_chunk(self, start, duration, chunk_filename):
        chunk_path = os.path.join(self.output_dir, chunk_filename)
        cmd = [
            "ffmpeg", "-y", "-i", self.audio_path,
            "-ss", str(start), "-t", str(duration),
            "-acodec", "copy", chunk_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return chunk_path if result.returncode == 0 else None

    def detect_emotion(self, chunk_path):
        try:
            result = self.pipe(chunk_path)[0]
            return result['label'], result['score']
        except Exception as e:
            print(f"Emotion detection failed on {chunk_path}: {e}")
            return "error", 0.0

    def process(self):
        i = 0
        while i < len(self.segments):
            if "speaker" not in self.segments[i]:
                i += 1
                continue

            group = [self.segments[i]]
            speaker = self.segments[i]["speaker"]
            i += 1

            while i < len(self.segments) and len(group) < self.max_segments:
                if "speaker" in self.segments[i] and self.segments[i]["speaker"] == speaker:
                    group.append(self.segments[i])
                    i += 1
                else:
                    break

            start = group[0]["start"]
            end = group[-1]["end"]
            duration = end - start
            chunk_filename = f"chunk_{self.chunk_id:03d}.wav"

            chunk_path = self.create_audio_chunk(start, duration, chunk_filename)
            if not chunk_path:
                print(f"FFmpeg failed for chunk {chunk_filename}")
                continue

            emotion_label, emotion_score = self.detect_emotion(chunk_path)

            merged_text = " ".join([seg["text"] for seg in group])
            merged_words = []
            for seg in group:
                for w in seg.get("words", []):
                    merged_words.append({
                        "start": w["start"] + seg["start"],
                        "end": w["end"] + seg["start"],
                        "word": w["word"]
                    })

            self.merged_results.append({
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

            os.remove(chunk_path)
            self.chunk_id += 1

    def save_results(self, output_json):
        with open(output_json, "w") as f:
            json.dump({"segments": self.merged_results}, f, indent=2)
        print(f"✅ Done! Speaker-aware emotion-rich chunks saved to {output_json}")




if __name__ == "__main__":
    processor = EmotionProcessor(
        json_path="backend/WhisperXModel/output/merged_raw/full_audio_raw_transcription_with_absolute_timestamps.json",
        audio_path="backend/WhisperXModel/audio/audio.wav",
        output_dir="backend/EmotionDetectionModel/audio/chunks"
    )

    processor.load_segments()
    processor.process()
    processor.save_results("backend/WhisperXModel/output/EmotionProcessed/complete.json")

