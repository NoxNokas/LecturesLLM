# LangChain также позаботился о том, чтобы нам не приходилось реализовывать векторный поиск вручную.
# Для этого есть специальный компонент VectorStore, который отвечает за создание векторов на основе переданного Embeddings,
# их хранения и создания объекта Retriever для поиска по документам.
# В примере ниже используется InMemoryVectorStore, который хранит документы в оперативной памяти во время работы программы,
# а поиск осуществляет без оптимизационных механизмов.
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


embeddings = OllamaEmbeddings(model="llama3.2:3b")

relevant_doc = Document(page_content="Большая языковая модель это языковая модель, состоящая из нейронной сети со множеством параметров (обычно миллиарды весовых коэффициентов и более), обученной на большом количестве неразмеченного текста с использованием обучения без учителя.")
irrelevant_doc = Document(page_content="Задачи сокращения размерности. Исходная информация представляется в виде признаковых описаний, причём число признаков может быть достаточно большим. Задача состоит в том, чтобы представить эти данные в пространстве меньшей размерности, по возможности, минимизировав потери информации..")

vectorstore = InMemoryVectorStore.from_documents(documents=[relevant_doc, irrelevant_doc], embedding=embeddings)

# Вернуть только один наиболее похожий документ
retriever = vectorstore.as_retriever(search_kwargs={'k': 1})

result = retriever.invoke("Что такое большая языковая модель?")
print(result)


# Мы можем уточнить метод поиска с помощью параметра search_kwargs.

# # Вернуть только один наиболее похожий документ
# retriever = vectorstore.as_retriever(search_kwargs={'k': 1})

# # Вернуть 6 наиболее разнообразных документов по метрике MRR
# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={'k': 6, 'lambda_mult': 0.25}
# )

# # Вернуть только те документы, у которых значение похожести больше или равно 0.8
# retriever = vectorstore.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={'score_threshold': 0.8}
# )

# # Использовать только документы у которых в metadata ключ source соответствует Course MVP AI Service
# retriever = vectorstore.as_retriever(
#     search_kwargs={'filter': {'source':'Course MVP AI Service'}}
# )