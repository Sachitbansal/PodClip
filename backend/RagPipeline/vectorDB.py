import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# === CONFIG ===
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TEXT_FILE_PATH = "backend/RagPipeline/outputs/embedding_input.txt"
INDEX_PATH = "backend/RagPipeline/outputs/faiss_index.index"
MAPPING_PATH = "backend/RagPipeline/outputs/id_to_text.pkl"

# === FUNCTION: Load texts ===
def load_texts_from_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Text file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# === FUNCTION: Generate Embeddings ===
def get_text_embeddings(texts, model_name=EMBEDDING_MODEL_NAME):
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True)

# === FUNCTION: Create and Save FAISS Index ===
def save_faiss_index(embeddings, texts, index_path, mapping_path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(mapping_path, "wb") as f:
        pickle.dump(texts, f)
    print(f"✅ FAISS index and mapping saved.")

# === MAIN PIPELINE ===
def main():
    print("📥 Loading texts...")
    texts = load_texts_from_file(TEXT_FILE_PATH)
    
    print("📐 Generating embeddings...")
    embeddings = get_text_embeddings(texts)

    print("💾 Saving FAISS index and ID-to-text mapping...")
    save_faiss_index(embeddings, texts, INDEX_PATH, MAPPING_PATH)

    print("✅ All done!")

if __name__ == "__main__":
    main()
