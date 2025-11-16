from langchain_core.documents import Document

def load_txt(path):
    text = open(path, "r", encoding="utf-8").read()
    return [Document(page_content=text, metadata={"source": path})]
