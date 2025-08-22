from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import pydantic
from langchain_ollama import ChatOllama

# StrOuputParser - возвращает атрибут content у AIMessage с помощью метода invoke. 
message = AIMessage(content="Самолёты могут летать.")
output_parser = StrOutputParser()
print(output_parser.invoke(message))

# PydanticOutputParser (Работает как json.loads только ещё и с пользовательскими классами)
class Person(pydantic.BaseModel):
    firstname: str = pydantic.Field(description="firstname of hero")
    lastname: str = pydantic.Field(description="lastname of hero")
    age: int = pydantic.Field(description="age of hero")


output_parser = PydanticOutputParser(pydantic_object=Person)
answer = AIMessage(content='{"firstname": "John", "lastname": "Smith", "age": 45}')
print(output_parser.invoke(answer))


# PydanticOutputParser: в связке с LLM
llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

output_parser = PydanticOutputParser(pydantic_object=Person)

# В таком формате чёт не работает. Пришлось в другом сообщения оформлять.
# messages = [
#     SystemMessage("Handle the user query.\n{format_instructions}"),
#     HumanMessage("{user_query}")
# ]

messages = [
    ("system", "Handle the user query.\n{format_instructions}"),
    ("human", "{user_query}")
]

prompt_template = ChatPromptTemplate(messages=messages)
prompt_value = prompt_template.invoke({
    "format_instructions": output_parser.get_format_instructions(), 
    "user_query": "Генрих Смит был сорокавосьми летним вымешленным роботом"})

answer = llm.invoke(prompt_value.to_messages())
print(output_parser.invoke(answer))


# Можно ещё и опциональные поля делать или по умолчанию.
# from typing import Optional
# class Person(pydantic.BaseModel):
#     firstname: str = pydantic.Field(description="firstname of hero")
#     lastname: Optional[str] = pydantic.Field(description="lastname of hero")
#     age: int = pydantic.Field(description="age of hero", default=0)