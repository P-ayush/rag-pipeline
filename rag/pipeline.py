from utils.file_utils import load_any
from rag.splitter import split_docs
from rag.vector_store import load_vectorstore, save_vectorstore
from rag.embeddings import embeddings
from langchain_community.vectorstores import FAISS
from rag.llm import chat
from langchain_core.prompts import ChatPromptTemplate

def ingest(path):
    docs = load_any(path)    
    print("[DEBUG] Docs returned by loader:", len(docs))
    print("[DEBUG] Length of page_content:", len(docs[0].page_content))
               
    chunks = split_docs(docs)            
    print("[DEBUG] Chunks after splitting:", len(chunks))

    db = load_vectorstore()                 
    if db:
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)

    save_vectorstore(db)
    return len(chunks)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer only using the provided context. "
     "If you cannot find the answer, say 'I don't know'."),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
])

def rag_query(question: str):
    db = load_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    messages = prompt.invoke({"context": context, "question": question})
    response = chat.invoke(messages)
    return response.content
