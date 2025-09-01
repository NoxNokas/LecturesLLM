# В библиотеке LangChain предусмотрен специальный компонент TextSplitter,
# который реализует логику разбиения текста на фрагменты или чанки.
# Например, класс RecursiveCharacterTextSplitter рекурсивно разделяет документ,
# используя такие разделители, как перенос строки, до тех пор, пока каждый фрагмент не достигнет заданного размера.
# Начинать работать с текстом всегда лучше именно с этого разделителя.
# Результатом разбиения является список из элементов Document, которые далее можно передавать в VectorStore.
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = PyPDFLoader("./05_RAG/05_document_processing/paper.pdf")
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
chunks = text_splitter.split_documents(pages)
print(chunks)