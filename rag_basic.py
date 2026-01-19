# rag_basic.py
import os
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = os.path.join("data", "knowledge.txt")

def load_knowledge_text() -> str:
    if not os.path.exists(DATA_PATH):
        return ""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()

def simple_chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = max(0, end - overlap)

    return chunks

def build_retriever(k: int = 3):
    text = load_knowledge_text()
    if not text.strip():
        text = "This is an empty knowledge base. No domain info available."

    chunks = simple_chunk_text(text, chunk_size=500, overlap=100)

    embeddings = OpenAIEmbeddings()
    metadatas = [{"source": "knowledge.txt", "chunk_id": i} for i in range(len(chunks))]

    vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
    return vectorstore.as_retriever(search_kwargs={"k": k})
