import datetime
from typing import Annotated, Sequence, TypedDict
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END


# --------------------------------- State ---------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int


# --------------------------------- Tools ---------------------------------------
@tool(return_direct=True)
def get_this_year_tool() -> int:
    """Получить текущий год"""
    return datetime.datetime.now().year

class WikiInput(BaseModel):
    query: str = Field(description="Запрос для поиска в Википедии")

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="ru"))

@tool(return_direct=True, args_schema=WikiInput)
def search_using_wikipedia(query: str) -> str:
    """Позволяет искать что-то в Википедии"""
    return wikipedia.run({"query": query})

tools = [get_this_year_tool, search_using_wikipedia]
tools_by_name = {tool.name: tool for tool in tools}

# --------------------------------- Nodes ---------------------------------------
def call_tool(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(ToolMessage(content=tool_result, name=tool_call["name"], tool_call_id=tool_call["id"]))
    return AgentState(messages=outputs, number_of_steps=state["number_of_steps"] + 1)

def call_model(state: AgentState, config: RunnableConfig):
    model = ChatOllama(model="llama3.1:8b", temperature=0)
    model = model.bind_tools(tools)
    response = model.invoke(input=state["messages"], config=config)
    return AgentState(messages=[response], number_of_steps=state["number_of_steps"] + 1)

# --------------------------------- Edge ---------------------------------------
def should_continue(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "continue"
    return "end"


# ---------------------------- Graph Building -----------------------------------
builder = StateGraph(AgentState)

builder.add_node("llm", call_model)
builder.add_node("tools", call_tool)

builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", should_continue, {"continue": "tools", "end": END})
builder.add_edge("tools", "llm")
graph = builder.compile()


# ------------------------ Graph Visualization ---------------------------------
with open("./04_LLM_AGENTS/08_langgraph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


# ------------------------ Graph Invoke ---------------------------------
inputs = {
    "messages": [("user", "Сколько лет прошло с появления передачи Поле чудес в эфире? Кто её ведущий сегодня?")],
    "number_of_steps": 0
    }
state = graph.invoke(inputs)

for message in state["messages"]:
    message.pretty_print()