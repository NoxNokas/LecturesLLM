from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# Простой пример использования Document и Retriever
# documents = [
#     Document(page_content="foo"),
#     Document(page_content="bar"),
#     Document(page_content="hello foo"),
#     Document(page_content="hello bar"),
# ]

# retriever = BM25Retriever.from_documents(documents=documents)
# result = retriever.invoke("Что такое большая языковая модель?")
# print(*result, sep='\n\n\n')
# print('_'*30)

# Цепочка для самого простого RAG
knowledge_store = [
    Document(page_content="Большая языковая модель это языковая модель, состоящая из нейронной сети со множеством параметров (обычно миллиарды весовых коэффициентов и более), обученной на большом количестве неразмеченного текста с использованием обучения без учителя."),
    Document(page_content="Большие языковые модели появились примерно в 2018 году и хорошо справляются с широким спектром задач. Это сместило фокус исследований обработки естественного языка с предыдущей парадигмы обучения специализированных контролируемых моделей для конкретных задач."),
    Document(page_content="Тонкая настройка — это практика модификации существующей предварительно обученной языковой модели путём её обучения (под наблюдением) конкретной задаче (например, анализ настроений, распознавание именованных объектов или маркировка частей речи). Это форма передаточного обучения. Обычно это включает введение нового набора весов, связывающих последний слой языковой модели с выходными данными последующей задачи."),
    Document(page_content="Обучение без учителя — один из способов машинного обучения, при котором испытуемая система спонтанно обучается выполнять поставленную задачу без вмешательства со стороны экспериментатора. С точки зрения кибернетики, это является одним из видов кибернетического эксперимента. Как правило, это пригодно только для задач, в которых известны описания множества объектов (обучающей выборки), и требуется обнаружить внутренние взаимосвязи, зависимости, закономерности, существующие между объектами."),
    Document(page_content="Задачи сокращения размерности. Исходная информация представляется в виде признаковых описаний, причём число признаков может быть достаточно большим. Задача состоит в том, чтобы представить эти данные в пространстве меньшей размерности, по возможности, минимизировав потери информации.."),
    Document(page_content="При этом в эксперименте по «чистому обобщению» от модели мозга или перцептрона требуется перейти от избирательной реакции на один стимул (допустим, квадрат, находящийся в левой части сетчатки) к подобному ему стимулу, который не активизирует ни одного из тех же сенсорных окончаний (квадрат в правой части сетчатки)."),
]

retriever = BM25Retriever.from_documents(knowledge_store)

def format_documents(documents: list[Document]) -> str:
    return '\n\n'.join(doc.page_content for doc in documents)

prompt = ChatPromptTemplate(messages=[
    ("system", "You are an assistant for QA. Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. Answer as short as possible. "),
    ("user", "Context: {context} \nQuestion: {question}")
])

llm = ChatOllama(model="llama3.1:8b")

# В RunnableParallel в ключ "question" можно вместо лямбды использовать RunnablePassthrough(),
# чтобы пробросить входной параметр
chain = RunnableParallel({"context": retriever | format_documents, "question": lambda data: data}) \
    | prompt | llm | StrOutputParser()

result = chain.invoke("Что такое большая языковая модель?")

# способ № 2, где входным параметром является не стока, а словарь
# from operator import itemgetter
# chain = RunnableParallel(
#     {
#         "context": itemgetter("question") | retriever | format_documents,
#         "question": lambda data: data["question"] # или можно itemgetter("question") | RunnablePassthrough()
#     }) | prompt | llm | StrOutputParser()

# result = chain.invoke({"question": "Что такое большая языковая модель?"})
print(result)