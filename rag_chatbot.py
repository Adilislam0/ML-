# rag_chatbot.py  (Local LLM variant)
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# ---------- Config ----------
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-small"   # small and CPU-friendly for demo
TOP_K = 3

# ---------- Load embedding model and index ----------
embed_model = SentenceTransformer(EMBED_MODEL)

# Load documents (same corpus.txt)
with open("corpus.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f.readlines()]

# Build/load index - for simplicity rebuild (fast for small corpora)
embeddings = embed_model.encode(documents).astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print("Index loaded with", index.ntotal, "documents")

# ---------- Load local LLM pipeline ----------
gen = pipeline("text2text-generation", model=LLM_MODEL, device=-1)  # CPU

def retrieve(query, k=TOP_K):
    q_emb = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(q_emb, k)
    return [documents[i] for i in indices[0]], distances[0]

def build_prompt(query, retrieved_docs):
    context = "\n\n".join(f"Doc {i+1}: {d}" for i,d in enumerate(retrieved_docs))
    prompt = (
        f"You are an assistant. Use the following context (do not hallucinate). "
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer concisely and cite Docs as Doc 1, Doc 2, etc. If the answer "
        f"is not in the context, say 'I don't know'."
    )
    return prompt

def answer(query):
    docs, distances = retrieve(query)
    prompt = build_prompt(query, docs)
    # local LLM generation
    out = gen(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    return out, docs

if __name__ == "__main__":
    while True:
        q = input("\nQuery (type 'exit' to quit): ").strip()
        if q.lower() in ("exit","quit"):
            break
        ans, docs = answer(q)
        print("\n--- Answer ---")
        print(ans)
        print("\n--- Sources (retrieved) ---")
        for i,d in enumerate(docs):
            print(f"Doc {i+1}:", d)
