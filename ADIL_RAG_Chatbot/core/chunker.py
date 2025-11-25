# core/chunker.py

import re

def clean_text(text):
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def create_chunks(text, chunk_size=300, overlap=50):
    """
    Splits long text into overlapping chunks.
    chunk_size = number of words per chunk.
    overlap = repeated words between chunks.
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start = end - overlap  # move window backward for overlap
        if start < 0:
            start = 0

    return chunks


def load_text_file(path, chunk_size=300, overlap=50):
    with open(path, "r", encoding="utf-8") as f:
        text = clean_text(f.read())
    return create_chunks(text, chunk_size, overlap)
