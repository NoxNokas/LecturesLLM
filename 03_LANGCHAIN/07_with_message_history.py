# Пример работы функции trim_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, trim_messages


messages = [
    SystemMessage("Ты добрый дворецкий"),
    HumanMessage("Доброе утро!"),
    AIMessage("Здравствуйте!"),
    HumanMessage("Как Ваши дела?"),
    AIMessage("Неплохо! А у вас как?"),
    HumanMessage("Тоже неплохо! Как вы поживаете, как здоровье?"),
    AIMessage("Отлично! Спасибо, что поинтересовались."),
    HumanMessage("Ну, право. До свидания, рад был повидаться!"),
    AIMessage("Взаимно, взаимно. До встречи."),
]

trimmer = trim_messages(
    strategy="last",
    token_counter=len,
    max_tokens=6,
    start_on="human",
    end_on=["human", "ai"],
    include_system=True,
    allow_partial=False
)

new_messages = trimmer.invoke(messages)
print(*new_messages, sep='\n\n')
print('_' * 10)


# Пример работы функции trim_messages совместно с llm
print("Пример работы функции trim_messages совместно с llm")
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.config import RunnableConfig


DEFAULT_SESSION_ID = "default322"
chat_history = InMemoryChatMessageHistory()

llm = ChatOllama(model="llama3.1:8b")

trimmer = trim_messages(
    strategy="last",
    token_counter=len,
    max_tokens=6,
    start_on="human",
    end_on=["human", "ai"],
    include_system=True,
    allow_partial=False
)

chain = trimmer | llm
chain_with_history = RunnableWithMessageHistory(runnable=chain, get_session_history=lambda session_id: chat_history)

chain_with_history.invoke([HumanMessage("Hi, my name is Bob!")], config={"configurable": {"session_id": DEFAULT_SESSION_ID}})

answer = chain_with_history.invoke([HumanMessage("What is my name?")], 
                                   config=RunnableConfig(configurable={"session_id": DEFAULT_SESSION_ID}))
print(answer.content)