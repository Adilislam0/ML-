# rag_chatbot_openai.py
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

# CONFIG
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
OPENAI_MODEL = "gpt-4o-mini"   # use a model you have access to
MAX_TOKENS = 300
TEMPERATURE = 0.0

# API KEY
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("Set OPENAI_API_KEY environment variable before running")

# Load embedding model and corpus
embed_model = SentenceTransformer(EMBED_MODEL)
with open("corpus.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

# Build embeddings & FAISS index (for small corpora, rebuild each run; for large, save index)
embeddings = embed_model.encode(documents, show_progress_bar=False)
embeddings = np.array(embeddings).astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print("Index loaded with", index.ntotal, "documents")

# Retrieval helper
def retrieve(query, k=TOP_K):
    q_emb = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(q_emb, k)
    docs = []
    for idx in indices[0]:
        docs.append(documents[int(idx)])
    return docs, distances[0]

# Prompt builder — strict: use only context
def build_system_user(query, docs):
    context = "\n\n".join(f"Doc {i+1}: {d}" for i,d in enumerate(docs))
    system = (
        "You are a helpful assistant. Use ONLY the provided context to answer. "
        "If the answer cannot be found in the context, respond with: 'I don't know'. "
        "Cite sources as Doc 1, Doc 2, etc. Be concise."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a concise answer and cite the Docs you used."
    return system, user

# Call OpenAI chat
def call_openai_chat(system, user):
    resp = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )
    return resp["choices"][0]["message"]["content"].strip()

# End-to-end
def answer(query):
    docs, distances = retrieve(query)
    system, user = build_system_user(query, docs)
    reply = call_openai_chat(system, user)
    return reply, docs

if __name__ == "__main__":
    print("RAG chatbot (OpenAI) ready. Type 'exit' to quit.")
    while True:
        q = input("\nQuery: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        ans, docs = answer(q)
        print("\n--- Answer ---")
        print(ans)
        print("\n--- Sources (retrieved) ---")
        for i,d in enumerate(docs):
            print(f"Doc {i+1}:", d)
