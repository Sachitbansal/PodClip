import os
from sentence_transformers import SentenceTransformer

def load_texts(input_path):
    """Load each line as a separate input string."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def embed_texts(texts, model_name="intfloat/e5-large"):
    """Generate embeddings using local SentenceTransformer."""
    print("📦 Loading model:", model_name)
    model = SentenceTransformer(model_name)

    # E5 models require "query:" or "passage:" prefix
    texts = [f"passage: {text}" for text in texts]
    embeddings = model.encode(texts, convert_to_tensor=False)
    return embeddings

def save_embeddings(embeddings, output_path):
    """Save embeddings as a list of vectors."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for vec in embeddings:
            f.write(','.join(map(str, vec)) + '\n')
    print(f"✅ Saved {len(embeddings)} embeddings to {output_path}")

if __name__ == "__main__":
    input_path = "backend/RagPipeline/outputs/embedding_input.txt"
    output_path = "backend/RagPipeline/outputs/local_text_embeddings.csv"

    print("📄 Loading input strings...")
    texts = load_texts(input_path)

    print("🧠 Generating embeddings locally...")
    embeddings = embed_texts(texts)

    print("💾 Saving embeddings...")
    save_embeddings(embeddings, output_path)
