# rag_basic.py
import os
from typing import List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Make path reliable regardless of where you run uvicorn from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "knowledge.txt")


def load_knowledge_text() -> str:
    if not os.path.exists(DATA_PATH):
        return ""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


def simple_chunk_text(text: str, chunk_size: int = 500, overlap: int = 120) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def build_retriever(k: int = 3):
    text = load_knowledge_text()
    if not text.strip():
        text = "Knowledge base is empty. Add content to data/knowledge.txt."

    chunks = simple_chunk_text(text)

    embeddings = OpenAIEmbeddings()
    metadatas = [{"source": "knowledge.txt", "chunk_id": i} for i in range(len(chunks))]

    vectorstore = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)
    return vectorstore.as_retriever(search_kwargs={"k": k})