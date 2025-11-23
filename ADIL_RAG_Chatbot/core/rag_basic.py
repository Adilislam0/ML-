from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load text corpus
with open("corpus.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f.readlines()]

# Convert each document to embeddings
embeddings = model.encode(documents)

# Convert to float32 for FAISS
embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

print("Index built with", index.ntotal, "documents")

# Simple query
query = "What is gradient descent?"
query_emb = model.encode([query]).astype("float32")

# Search top-3 results
distances, indices = index.search(query_emb, k=3)

print("\nTop 3 relevant documents:")
for idx in indices[0]:
    print("-", documents[idx])
