from typing import TypedDict, Literal
import random
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph


# --------------------------------- State ---------------------------------------
class State(TypedDict):
    query: str
    resolver: str
    answer: str


# --------------------------------- Nodes ---------------------------------------
def choice_resolver(state: State) -> State:
    resolver = "support" if random.random() < 0.5 else "llm"
    state["resolver"] = resolver
    return state

def send_to_support(state: State) -> State:
    print(f"New message for Support: {state["query"]}")
    state["answer"] = "Your request is very important to us! Please wait for the agent to contact you. (Some logic...)"
    return state

def llm(state: State) -> State:
    messages = [
        ("system", "You are a friendly chatbot. Your task is answer the question as short as possible"),
        ("human", "{question}")
    ]
    prompt = ChatPromptTemplate(messages=messages)
    llama = ChatOllama(model="llama3.1:8b")
    chain = prompt | llama | StrOutputParser()
    answer = chain.invoke({"question": state["query"]})
    state["answer"] = answer
    return state

def send_to_user(state: State) -> State:
    print(f"New message to User: {state['answer']}")
    return state

# --------------------------------- Edges ---------------------------------------
## Ребро графа (Edge) это способ связать узлы графа друг с другом. Они могут быть двух видов:

## Direct: соединяет один узел с другим напрямую (в примере выше это llm -> send_to_user)
## Conditional: содержит логику по определению следующего узла для обработки
def route_by_resolver(state: State) -> Literal["llm", "send_to_support"]:
    if state["resolver"] == "support":
        # Заметим, что возвращаемым значением является название узла.
        return "send_to_support"
    else:
        return "llm"

# ---------------------------- Graph Building -----------------------------------
builder = StateGraph(state_schema=State)
builder.add_node("choice_resolver", choice_resolver)
builder.add_node("send_to_support", send_to_support)
builder.add_node("llm", llm)
builder.add_node("send_to_user", send_to_user)

builder.add_edge(START, "choice_resolver")
builder.add_conditional_edges("choice_resolver", route_by_resolver)
builder.add_edge("send_to_support", END)
builder.add_edge("llm", "send_to_user")
builder.add_edge("send_to_user", END)

graph = builder.compile()
# ------------------------ Graph Visualization ---------------------------------
with open("./04_LLM_AGENTS/07_simple_graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


# ------------------------ Graph Invoke ---------------------------------
result = graph.invoke({"query": "Hi, my computer is not working!"})
print(result) # Возвращается dict с полями указанными в классе State