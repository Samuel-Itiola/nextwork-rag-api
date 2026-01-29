import os
from fastapi import FastAPI
import chromadb

# Mock LLM mode for CI testing
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "0") == "1"

# Ollama host for Docker/Kubernetes
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

if not USE_MOCK_LLM:
    import ollama

app = FastAPI()
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")

@app.post("/query")
def query(q: str):
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results.get("documents") else ""

    if USE_MOCK_LLM:
        # In mock mode, return retrieved context directly (deterministic)
        return {"answer": context}

    # In production mode, use Ollama (pointed at your K8s Service or host)
    answer = ollama.generate(
        model="tinyllama",
        prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely:",
        host=OLLAMA_HOST,
    )
    return {"answer": answer["response"]}

@app.post("/add")
def add_knowledge(text: str):
    """Add new content to the knowledge base dynamically."""
    try:
        import uuid
        doc_id = str(uuid.uuid4())
        collection.add(documents=[text], ids=[doc_id])
        return {"status": "success", "message": "Content added to knowledge base", "id": doc_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}
