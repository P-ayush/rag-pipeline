from langchain_community.document_loaders import WebBaseLoader

def load_webpage(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    for d in docs:
        d.metadata["source"] = url

    return docs
