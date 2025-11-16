import pandas as pd
from langchain_core.documents import Document
def load_csv(path):
    df = pd.read_csv(path)
    return [Document(page_content=df.to_string(), metadata={"source": path})]
