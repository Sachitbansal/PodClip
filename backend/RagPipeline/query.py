import os
import json
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Updated imports for LangChain community modules
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# === Load Embedding Model and FAISS VectorStore ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
faiss_index = FAISS.load_local("backend/RagPipeline/outputs/vectorstore", embedding_model, allow_dangerous_deserialization=True)
retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# === Load transcript segments and convert to Document list ===
def load_transcript_segments(json_path: str) -> List[Document]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for segment in data["segments"]:
        text = segment["text"].strip()
        meta = {
            "start": segment["start"],
            "end": segment["end"],
            "speaker": segment.get("speaker", "unknown"),
            "emotion": segment.get("emotion", {}),
        }
        documents.append(Document(page_content=text, metadata=meta))
    return documents

# === Chunk documents with overlap for better context ===
def chunk_documents(documents: List[Document], chunk_size=500, chunk_overlap=50) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            all_chunks.append(Document(page_content=chunk, metadata=doc.metadata))
    return all_chunks

# === Create FAISS vectorstore from chunks with embeddings ===
def create_faiss_vectorstore(docs: List[Document], embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# === Save and load FAISS vectorstore ===
def save_faiss_index(vectorstore: FAISS, path: str):
    vectorstore.save_local(path)
    print(f"✅ FAISS index saved to {path}")    

def load_faiss_index(path: str, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return FAISS.load_local(path, embeddings)

# === Example: Query vectorstore ===
def query_vectorstore(vectorstore: FAISS, query: str, top_k=5):
    results = vectorstore.similarity_search(query, k=top_k)
    return results

def save_chunks_to_json(output_path="backend/RagPipeline/outputs/vector_chunks.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    chunks = []
    for i, doc in enumerate(faiss_index.docstore._dict.values()):
        chunks.append({
            "chunk_id": i + 1,
            "content": doc.page_content,
            "metadata": doc.metadata
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(chunks)} chunks to {output_path}")

# === MAIN workflow ===
if __name__ == "__main__":
    json_path = "backend/WhisperXModel/output/EmotionProcessed/complete.json"
    faiss_index_path = "backend/RagPipeline/outputs/vectorstore"

    print("🔍 Loading transcript segments...")
    documents = load_transcript_segments(json_path)

    print("✂️ Chunking documents with overlap...")
    chunked_docs = chunk_documents(documents, chunk_size=500, chunk_overlap=50)

    print("📦 Creating FAISS vectorstore...")
    vectorstore = create_faiss_vectorstore(chunked_docs)

    print("💾 Saving FAISS index locally...")
    save_faiss_index(vectorstore, faiss_index_path)
    save_chunks_to_json()

    print("✅ Done! Vectorstore ready.")

    # Example query:
    query = "What are the best emotional moments?"
    results = query_vectorstore(vectorstore, query)
    print("\nTop results:")
    for r in results:
        print(f"- {r.page_content} (metadata: {r.metadata})")
