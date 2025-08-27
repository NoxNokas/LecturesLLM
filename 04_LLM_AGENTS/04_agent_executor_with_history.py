# Агент с историей в оперативке
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor


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

llm = ChatOllama(model="llama3.1:8b", temperature=0)

prompt = ChatPromptTemplate([
    ("system", "Твоя задача отвечать на вопросы клиентов об их заказах, используя вызов инструментов. "
               "Отвечай пользователю подробно и вежливо."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

memory = InMemoryChatMessageHistory()
agent_executor_with_history = RunnableWithMessageHistory(
    runnable=agent_executor,
    get_session_history=lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output" # т.к. в конце пайплайна вернётся словарь с сообщением-ответом под ключом "output"
)

config = {"configurable": {"session_id": "by_big_boss"}}
while (user_question:=input("You: ")) != r"\bye":
    answer = agent_executor_with_history.invoke({"input": user_question}, config=config)
    print("Bot:", answer["output"])