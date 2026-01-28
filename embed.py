import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB client with explicit path
client = chromadb.PersistentClient(path="./db")

# Create collection with default embedding function (sentence-transformers)
# You can also use: embedding_functions.DefaultEmbeddingFunction()
# or specify a custom one: embedding_functions.SentenceTransformerEmbeddingFunction(model_name="...")
default_ef = embedding_functions.DefaultEmbeddingFunction()
collection = client.get_or_create_collection(
    name="docs",
    embedding_function=default_ef
)

# Read the text file (now in the same directory)
with open("k8s.txt", "r") as f:
    text = f.read()

# Add document to collection (ChromaDB will automatically generate embeddings)
collection.add(documents=[text], ids=["k8s"])

print("Embedding stored in Chroma")

