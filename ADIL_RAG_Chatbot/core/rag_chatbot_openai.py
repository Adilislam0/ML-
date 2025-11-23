# core/rag_chatbot_openai.py
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3
OPENAI_MODEL = "gpt-4o-mini"   # change if you don't have access
MAX_TOKENS = 300
TEMPERATURE = 0.0

def load_corpus(corpus_path="core/corpus.txt"):
    with open(corpus_path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    return docs

class RAG:
    def __init__(self, corpus_path="core/corpus.txt", embed_model=EMBED_MODEL):
        self.embed_model = SentenceTransformer(embed_model)
        self.documents = load_corpus(corpus_path)
        self.embeddings = self.embed_model.encode(self.documents, show_progress_bar=False).astype("float32")
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)

    def retrieve(self, query, k=TOP_K):
        q_emb = self.embed_model.encode([query]).astype("float32")
        distances, indices = self.index.search(q_emb, k)
        docs = [self.documents[int(idx)] for idx in indices[0]]
        return docs, distances[0]

    def build_messages(self, query, docs):
        context = "\n\n".join(f"Doc {i+1}: {d}" for i, d in enumerate(docs))
        system = (
            "You are a helpful assistant. Use ONLY the provided context. "
            "If the answer is not present, respond with: 'I don't know'. "
            "Cite sources as Doc 1, Doc 2, etc."
        )
        user = f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a concise answer with citations."
        return system, user

    def call_openai(self, system, user):
        resp = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return resp.choices[0].message.content.strip()

    def answer(self, query):
        docs, _ = self.retrieve(query)
        system, user = self.build_messages(query, docs)
        ans = self.call_openai(system, user)
        return ans, docs
