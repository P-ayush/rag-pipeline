import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

def load_webpage(url):
    html = requests.get(url).text
    # print(html)
    soup = BeautifulSoup(html, "html.parser")
    # print(soup)
    text = soup.get_text(separator="\n")
    # print(text)
    return [Document(page_content=text, metadata={"source": url})]
