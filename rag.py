from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from vector_store import load_vector_store

from dotenv import load_dotenv


load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature=0,
    max_new_tokens=300,
)
chat = ChatHuggingFace(llm=llm)
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You MUST answer ONLY using the provided context.\n"
     "If the answer is not in the context, say ONLY: \"I don't know\".\n"
     "Do NOT use outside knowledge.\n"),
    ("human",
     "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
])

vector_store = load_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
def rag_query(question: str):
   
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    final_prompt = prompt.format(context=context, question=question)
    response = chat.invoke(final_prompt)
    return response.content