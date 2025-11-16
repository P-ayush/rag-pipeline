from langchain_huggingface import  HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
)
# text1 = "That’s a sample file."
# text2 = "That’s a sample file."

# Generate embeddings
# v1 = embeddings.embed_query(text1)
# v2 = embeddings.embed_query(text2)
# similarity = cosine_similarity([v1], [v2])[0][0]
# print(f"Cosine similarity: {similarity:.4f}")
