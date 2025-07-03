import os
from transformers import pipeline

def run_emotion_pipeline(
    input_folder="backend/EmotionDetectionModel/audio/chunks",
    output_file="emotion_outputs/results.txt",
    model_name="superb/hubert-large-superb-er",
    device=None,
):
    # Automatically detect GPU
    if device is None:
        try:
            import torch
            device = 0 if torch.cuda.is_available() else -1
        except ImportError:
            device = -1  # fallback

    # Load emotion classification pipeline
    pipe = pipeline("audio-classification", model=model_name, device=device)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Clear previous results
    with open(output_file, "w") as f:
        f.write("")

    # Process all .wav files in input folder
    for fname in sorted(os.listdir(input_folder)):
        if fname.endswith(".wav"):
            file_path = os.path.join(input_folder, fname)

            try:
                result = pipe(file_path)[0]  # Top prediction
                label = result['label']
                score = result['score']

                print(f"{fname}: {label} ({score:.2f})")

                with open(output_file, "a") as f:
                    f.write(f"{fname}: {label} ({score:.2f})\n")

            except Exception as e:
                print(f"Error processing {fname}: {e}")

