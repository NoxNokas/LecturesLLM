import json, openai


ORDERS_STATUSES_DATA = {
    "a42": "Доставляется",
    "b61": "Выполнен",
    "k37": "Отменен",
}

def get_order_status(order_id: str) -> str:
    return ORDERS_STATUSES_DATA.get(order_id, f"Не существует заказа с order_id={order_id}")

def cancel_order(order_id: str) -> str:
    if order_id not in ORDERS_STATUSES_DATA:
        return f"Не существует заказа с order_id={order_id}"
    if ORDERS_STATUSES_DATA[order_id] != "Отменен":
        ORDERS_STATUSES_DATA[order_id] = "Отменен"
        return "Заказ успешно отменен"
    return "Заказ уже отменен"


NAMES_OF_FUNCTION = {
    "get_order_status": get_order_status,
    "cancel_order": cancel_order
}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get status of order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order identifier"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel the order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order identifier",
                    }
                },
                "required": ["order_id"],
            }
        }
    }
]

client = openai.OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

messages = [
    {
        "role": "user",
        "content": "Отмени заказ a42"
    }
]

print("You: ", messages[0]["content"])

# Получаем ответ модели
chat_completion = client.chat.completions.create(
    messages=messages,
    model="llama3.1:8b",
    tools=TOOLS,
    tool_choice="auto"
)

# Когда модель решает, что нужно вызвать функцию, то она поле 'content' оставляет пустым
print("AI:", chat_completion.choices[0].message)

tool_call = chat_completion.choices[0].message.tool_calls[0]
function_name = tool_call.function.name
function_kwargs = json.loads(tool_call.function.arguments)
function_result = NAMES_OF_FUNCTION[function_name](**function_kwargs)

messages.append(chat_completion.choices[0].message)
messages.append({"role": "tool", "name": function_name, "content": function_result, "tool_call_id": tool_call.id})

print("Function calling:", tool_call)
print("function_name: ", function_name)
print("function_params: ", function_kwargs)
print("function_result: ", function_result)
print('_' * 15)

chat_completion = client.chat.completions.create(messages=messages, model="llama3.1:8b")
print("Final answer:", chat_completion.choices[0].message.content)