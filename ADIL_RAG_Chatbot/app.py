# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.rag_chatbot_openai import RAG

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

# initialize RAG instance once
try:
    rag = RAG(corpus_path=os.path.join(os.path.dirname(__file__), "..", "core", "corpus.txt"))
except Exception as e:
    rag = None
    print("RAG initialization error:", e)

@app.post("/ask")
async def ask(payload: Query):
    if not rag:
        raise HTTPException(status_code=500, detail="RAG backend not initialized.")
    q = payload.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    ans, docs = rag.answer(q)
    return {"answer": ans, "sources": docs}
