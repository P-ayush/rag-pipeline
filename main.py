from pdf_loader import load_pdf
from splitter import split_text
from rag import rag_query
from vector_store import load_vector_store, create_vector_store
import os


text = load_pdf("./raw/ayushResumeUpdated.pdf")
chunks = split_text(text)

if not os.path.exists("./faiss_db"):
    print("Creating FAISS DB for the first time...")
    create_vector_store(chunks)
else:
        print("FAISS DB already exists.")
while True:
    question = input("\nAsk something about the uploaded pdf: ")
    print("\nAnswer:", rag_query(question))