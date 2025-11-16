import pdfplumber
from langchain_core.documents import Document

def load_pdf(path):
    docs = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={"source": path, "page": i+1}
                ))
    return docs
