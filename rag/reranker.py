import numpy as np
from rag.embeddings import embeddings  

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def rerank(question, docs):
    """Re-rank documents using cosine similarity between embeddings."""
    
    q_emb = embeddings.embed_query(question)

    ranked = []
    for d in docs:
        doc_emb = embeddings.embed_query(d.page_content) 
        score = cosine_similarity(q_emb, doc_emb)
        ranked.append((d, score))

    ranked.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in ranked]
