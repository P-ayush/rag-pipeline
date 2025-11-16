from loaders.pdf_loader import load_pdf
from loaders.docx_loader import load_docx
from loaders.csv_loader import load_csv
from loaders.txt_loader import load_txt

def load_any(path):
    ext = path.split(".")[-1].lower()

    if ext == "pdf":
        return load_pdf(path)

    if ext == "docx":
        return load_docx(path)

    if ext == "csv":
        return load_csv(path)

    if ext == "txt":
        return load_txt(path)

    raise Exception(f"Unsupported file type: {ext}")
