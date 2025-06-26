import os
import re
import json
from langchain_community.document_loaders.youtube import TranscriptFormat, YoutubeLoader

def extract_youtube_transcript_chunks(
    video_url: str,
    chunk_size: int = 10
) -> str:
    # Extract video ID for unique filename
    video_id_match = re.search(r"(?:v=|youtu\.be/)([\w\-]+)", video_url)
    video_id = video_id_match.group(1) if video_id_match else "output"
    output_file = f"youtube_chunks_{video_id}.json"

    # Check if file already exists
    if os.path.exists(output_file):
        print(f"✅ File already exists: {output_file}")
        return output_file

    # Otherwise, process and save transcript
    loader = YoutubeLoader.from_youtube_url(
        video_url,
        transcript_format=TranscriptFormat.CHUNKS,
        chunk_size_seconds=chunk_size,
    )

    docs = loader.load()

    data = [
        {
            "transcript": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in docs
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Transcript saved: {output_file}")
    return output_file
