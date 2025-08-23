from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


DEFAULT_SESSION_ID = "default"
chat_history = InMemoryChatMessageHistory()

messages = [
    ("system", "You are an expert in {domain}. Your task is answer the question as short as possible"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
]

prompt = ChatPromptTemplate(messages=messages)

trimmer = trim_messages(
    strategy="last",
    token_counter=len,
    max_tokens=10,
    start_on="human",
    end_on="human",
    include_system=True,
    allow_partial=False
)

llm = ChatOllama(model="llama3.1:8b")

chain = prompt | trimmer | llm
chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=lambda session_id: chat_history,
    input_messages_key="question",
    history_messages_key="history")
final_chain = chain_with_history | StrOutputParser()

domain = input("Choice domain area: ")
while (user_question := input("You: ")) != r"\bye":
    print("AI: ", end='')
    for chunk in final_chain.stream(
        {"domain": domain, "question": user_question},
        config={"configurable": {"session_id": DEFAULT_SESSION_ID}}):
        print(chunk, end='')
    print()