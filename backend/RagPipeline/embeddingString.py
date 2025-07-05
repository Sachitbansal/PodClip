import json
import os
def generate_embedding_strings_from_segments(json_path: str) -> list[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    embedding_strings = []

    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_UNKNOWN")
        emotion = seg.get("emotion", {}).get("label", "EMOTION_UNKNOWN")
        text = seg.get("text", "").strip()

        if not text:
            continue

        line = f"[{speaker}] [EMOTION_{emotion}] {text}"
        embedding_strings.append(line)

    return embedding_strings

def save_embedding_strings_to_txt(strings: list[str], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create parent folders if needed
    with open(output_path, "w", encoding="utf-8") as f:
        for line in strings:
            f.write(line + "\n")

if __name__ == "__main__":
    input_json = "backend/WhisperXModel/output/EmotionProcessed/complete.json"
    output_txt = "backend/RagPipeline/outputs/embedding_input.txt"

    strings = generate_embedding_strings_from_segments(input_json)
    save_embedding_strings_to_txt(strings, output_txt)

    print(f"✅ Saved {len(strings)} embedding strings to {output_txt}")
