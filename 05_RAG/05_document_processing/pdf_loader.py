# Каждая страница исходного pdf-документа превращается в отдельный Document с метаданными об источнике и номере страницы.
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("./05_RAG/05_document_processing/paper.pdf")
pages = loader.load()
print(len(pages))
print(pages[0].page_content)
print(pages[0].metadata)