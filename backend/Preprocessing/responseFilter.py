import re
import json
from backend.Preprocessing.sceneDetection import analyze_podcast_segment
from backend.Preprocessing.chunking import extract_youtube_transcript_chunks

file = extract_youtube_transcript_chunks("https://www.youtube.com/watch?v=9EqrUK7ghho")

# Replace this with your actual LLM response (as string)
raw_response = analyze_podcast_segment(24, 36, file)

# Use regex to extract the first JSON object
match = re.search(r'\{[\s\S]*?\}', raw_response)

if match:
    json_str = match.group(0)

    # Parse and save to a file
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(json_str), f, indent=2, ensure_ascii=False)

    print("✅ JSON extracted and saved to output.json")
else:
    print("❌ No valid JSON found in the response.")

