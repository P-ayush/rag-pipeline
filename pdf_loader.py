from langchain_community.document_loaders import PyPDFLoader

file_path = "./raw/ayushResumeUpdated.pdf"
loader = PyPDFLoader(file_path)

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return "\n\n".join([d.page_content for d in docs])


