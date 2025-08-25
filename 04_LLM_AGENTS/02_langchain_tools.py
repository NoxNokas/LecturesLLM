# tool-декоратор
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import Field
import json

ORDERS_STATUSES_DATA = {
    "a42": "Доставляется",
    "b61": "Выполнен",
    "k37": "Отменен",
}

@tool
def get_order_status(order_id: str = Field(description="Identifier of order")) -> str:
    """Get status of order by order identifier"""
    return ORDERS_STATUSES_DATA.get(order_id, f"Не существует заказа с order_id={order_id}")

# Посмотрим на отдельные части генерируемой схемы
print(get_order_status.name)
print(get_order_status.description)
print(get_order_status.args)
# Теперь посмотрим на всю схему функции целиком с помощью атрибута args_schema
print("Вся схема:")
print(json.dumps(get_order_status.args_schema.model_json_schema(), indent=4))

# Декорированная функция является Runnable-объектом, поэтому её можно вызвать с помощью invoke
print("декорированную функцию можно вызвать с помощью invoke (аргументы передаются обёрнутыми в словарь):")
print(get_order_status.invoke({"order_id": "a42"}))


# Использование Tool Calling с LLM
llm = ChatOllama(model="llama3.1:8b", temperature=0.1)
llm_with_tools = llm.bind_tools([get_order_status])

messages = [
    HumanMessage(content="What about my order k37?")
]
answer = llm_with_tools.invoke(messages)
messages.append(answer) # Хоть и content пустой, но всё равно добавляем в историю
print(answer.content)
print(answer.tool_calls)

for tool_call in answer.tool_calls:
    if tool_call["name"] == get_order_status.name:
        tool_message = get_order_status.invoke(tool_call)
        messages.append(tool_message)


answer = llm_with_tools.invoke(messages)
messages.append(answer)
print(answer.content)