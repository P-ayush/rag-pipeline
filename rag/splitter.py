from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
def split_text(text):
    return text_splitter.split_text(text)
def split_docs(docs):
    return text_splitter.split_documents(docs)