# Агент без истории

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


ORDERS_STATUSES_DATA = {
    "a42": "Доставляется",
    "b61": "Выполнен",
    "k37": "Отменен",
}

@tool
def get_order_status(order_id: str) -> str:
    """Get status of order by order identifier"""
    return ORDERS_STATUSES_DATA.get(order_id, f"Не существует заказа с order_id={order_id}")

@tool
def cancel_order(order_id: str) -> str:
    """Cancel the order by order identifier"""
    if order_id not in ORDERS_STATUSES_DATA:
        return f"Такой заказ не существует"
    if ORDERS_STATUSES_DATA[order_id] != "Отменен":
        ORDERS_STATUSES_DATA[order_id] = "Отменен"
        return "Заказ успешно отменен"
    return "Заказ уже отменен"

tools = [get_order_status, cancel_order]

llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

prompt = ChatPromptTemplate([
    ("system", "Твоя задача отвечать на вопросы клиентов об их заказах, используя вызов инструментов. Отвечай пользователю подробно и вежливо."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose/return_intermediate_steps - вывод в консоль/в атрибуте ответа
answer = agent_executor.invoke({"input": "Отмени заказ k37"})
print(answer["output"])