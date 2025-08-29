# Попробуем создать вектора документов и сравнить их с вектором запроса.
# В этом нам поможет косинусная близость, которая в примере ниже вычисляется с помощью функции similarity_score.

import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


def similarity_score(vector1: np.array, vector2: np.array) -> float:
    return np.sum(vector1 * vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

relevant_doc = Document(page_content="Большая языковая модель это языковая модель, состоящая из нейронной сети со множеством параметров (обычно миллиарды весовых коэффициентов и более), обученной на большом количестве неразмеченного текста с использованием обучения без учителя.")
irrelevant_doc = Document(page_content="Задачи сокращения размерности. Исходная информация представляется в виде признаковых описаний, причём число признаков может быть достаточно большим. Задача состоит в том, чтобы представить эти данные в пространстве меньшей размерности, по возможности, минимизировав потери информации..")

embeddings = OllamaEmbeddings(model="llama3.1:8b")

query_vector = embeddings.embed_query("Что такое большая языковая модель?") # получаем эмбеддинг запроса

document_vectors = embeddings.embed_documents([relevant_doc.page_content, irrelevant_doc.page_content]) # получаем эмбеддинги документов

# Результаты показывают, что более релевантный документ имеет большее значение similarity_score.
print("Relevant document score:", similarity_score(np.array(query_vector), np.array(document_vectors[0])))
print("Irrelevant document score:", similarity_score(np.array(query_vector), np.array(document_vectors[1])))