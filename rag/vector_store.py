import os
from langchain_community.vectorstores import FAISS
from rag.embeddings import embeddings

VECTOR_DIR = "./vector_db"

def save_vectorstore(db):
    db.save_local(VECTOR_DIR)

def load_vectorstore():
    if os.path.exists(VECTOR_DIR):
        return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    return None
