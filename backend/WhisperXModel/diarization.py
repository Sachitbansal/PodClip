# modules required are whisperx and ffmpeg

import subprocess
import os

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

chunks_dir = "audio/chunks"
output_base = "output"  # base directory to store results

# Ensure output base directory exists
os.makedirs(output_base, exist_ok=True)

for file in sorted(os.listdir(chunks_dir)):
    if file.endswith(".wav"):
        input_path = os.path.join(chunks_dir, file)
        filename_without_ext = os.path.splitext(file)[0]
        output_dir = os.path.join(output_base, filename_without_ext)

        os.makedirs(output_dir, exist_ok=True)

        command = [
            "whisperx",
            "--model", "medium",
            "--chunk_size", "4",
            "--diarize",
            "--hf_token", os.getenv("HUGGING_FACE_TOKEN"),
            "--output_dir", output_dir,
            input_path
        ]

        subprocess.run(command, check=True)
