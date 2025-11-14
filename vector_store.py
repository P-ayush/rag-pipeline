from langchain_community.vectorstores import FAISS
from embeddings import embeddings

PERSIST_DIR = "./faiss_db"

def create_vector_store(chunks):
    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )
    vector_store.save_local(PERSIST_DIR)
    return vector_store

def load_vector_store():
    return FAISS.load_local(
        PERSIST_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
