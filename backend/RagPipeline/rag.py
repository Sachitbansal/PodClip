# SWITCHING ON TO SYMANTIC SEGMENTATION MODEL USING CONSIGN SIMILARITY SEARCH
# WITH SOME HARD CODED RULES AND KEEPING LLM USE TO A SEPARATE



# import os
# import json
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document

# # === Load Environment ===
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("Missing GOOGLE_API_KEY in .env")

# # === Load Embedding Model and FAISS VectorStore ===
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# faiss_index = FAISS.load_local("backend/RagPipeline/vectorstore", embedding_model, allow_dangerous_deserialization=True)
# retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# # === Gemini Chat LLM ===
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# # === Prompt Template ===
# prompt_template = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
#         You are a viral content editor working for a podcast highlights channel.
#         Your task is to read the podcast transcript context and generate a short, hook-worthy highlight line.

#         Instructions:
#         - Make it catchy, emotional, and attention-grabbing
#         - Informal tone is preferred
#         - Max 20 words

#         Context:
#         {context}

#         Question:
#         {question}
#         """
#     )

# # === RAG Chain Setup ===
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     chain_type="stuff",
#     chain_type_kwargs={"prompt": prompt_template},
#     return_source_documents=True
# )

# # === Ask LLM using Retrieval ===
# def generate_rag_hook(question: str):
#     result = qa_chain({"query": question})
#     return result["result"], result["source_documents"]

# # === MAIN ===
# if __name__ == "__main__":
#     question = "What are the most hook-worthy moments in this podcast?"
#     print("🚀 Running RAG with Gemini...")
#     answer, sources = generate_rag_hook(question)
#     print("\n🎯 Hook Line:", answer)
#     print("\n📄 Top Source Chunks:")
#     for i, doc in enumerate(sources):
#         print(f"\n--- Chunk {i+1} ---\n{doc.page_content}")
