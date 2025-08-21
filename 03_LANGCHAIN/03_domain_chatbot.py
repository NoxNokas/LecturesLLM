from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

messages = [
    SystemMessage("You are an expert in {domain}. Your task is answer the question as short as possible"),
    MessagesPlaceholder("history")
]
prompt_template = ChatPromptTemplate(messages=messages)

domain = input("Choice domain area: ")
print("____________________")
history = []
while (user_content:=input("You: ")) != r"\bye":
    history.append(HumanMessage(user_content))
    prompt_value = prompt_template.invoke({"domain": domain, "history": history})

    full_ai_answer = ""
    print("AI: ", end='')
    for chunk in llm.stream(input=prompt_value):
        full_ai_answer += chunk.text()
        print(chunk.content, end='')
    history.append(AIMessage(content=full_ai_answer))
    print("\n____________________")