# Современные модели могут поддерживать структурированный вывод из коробки.
# Это когда мы задаем схему в параметрах вызова модели, а не в описании промпта.
# Обычно это позволяет добиться более надежного следования схеме ответа.
#
# Со списком моделей, поддерживающих данную функциональность, можно ознакомиться по ссылке: 
# https://python.langchain.com/docs/integrations/chat/.
#
# У ChatModel в LangChain есть метод with_structured_output с помощью которого мы можем задавать
# схему для ответа с помощью словаря в виде JSON-схемы или pydantic-класса.
# Результатом вызова модели в этом случае является не сообщение AIMessage,
# а словарь или объект класса указанного pydantic-класса.

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama


class Person(BaseModel):
    firstname: str = Field(description="firstname of hero")
    lastname: str = Field(description="lastname of hero")
    age: int = Field(description="age of hero")

llm = ChatOllama(model="llama3.1:8b", ).with_structured_output(Person)

messages = [
    ("system", "Handle the user query"),
    ("human", "Генрих Смит был восемнацдцателетним юношей, мечтающим уехать в город")
]

answer = llm.invoke(messages)
print(answer)