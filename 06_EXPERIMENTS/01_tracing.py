from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableConfig
from langchain_core.output_parsers.string import StrOutputParser

# Реализуем трейсинг для RAG-приложения. Начнем с подключения трейсинга к компонентам LangChain
tracer_provider = register(project_name="text_tracing", endpoint="http://localhost:6006/v1/traces")
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# Реализуем пайплайн получения данных и создания ретривера над ними
loader = PyPDFLoader("06_EXPERIMENTS/paper.pdf")
pages = loader.load()[:10]
full_fext = '\n'.join(page.page_content for page in pages)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200, add_start_index=True)
text_chunks = text_splitter.split_text(full_fext)
documents = [Document(page_content=text, metadata={"source": "paper.pdf"}) for text in text_chunks]

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
vectorstore = InMemoryVectorStore(embedding=embeddings)
vectorstore.add_documents(documents=documents)
retriever = vectorstore.as_retriever(search_kwargs={'k': 6})

# Инициализируем LLM и промпт для вопросно-ответной системы на базе RAG
llm = ChatOllama(model="llama3.1:8b")
prompt = ChatPromptTemplate([
    ("system", "You are an assistant for QA. Use the following pieces of retrieved context to answer the question. "
               "If you don't know the answer, just say that you don't know. Answer as short as possible. "
               "Context: {context}"),
    ("human", "Question: {question}")
])

# Создадим функцию format_docs_runnable  для того, чтобы в контекст попадал только текст из переданных фрагментов без метаданных.
# Метод with_config используется для явного задания имени компонента, что будет удобно в дальнейшей визуализации
format_docs_runnable = RunnableLambda(func=lambda docs: '\n\n'.join(doc.page_content for doc in docs))\
                                             .with_config(RunnableConfig(run_name="format documents"))

chain = RunnableParallel(context=retriever | format_docs_runnable, question=RunnablePassthrough()) | prompt | llm | StrOutputParser()

result = chain.invoke("What is attention?")
print(result)