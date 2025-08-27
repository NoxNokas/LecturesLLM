# ReAct Agent
from langchain import hub
import datetime

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain.agents import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent


@tool(name_or_callable="current_date_tool")
def get_current_date_tool() -> str:
    """Get the current date"""
    
    return str(datetime.datetime.now().date())

class WikiInputs(BaseModel):
    """Inputs to the wikipedia tool"""
    
    query: str = Field(description="query to look up in Wikipedia")


wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="ru"))
wikipedia_tool = Tool(
    name="wikipedia-tool",
    description="Look up things in Wikipedia",
    args_schema=WikiInputs,
    func=wikipedia.run
)

TOOLS = [wikipedia_tool, get_current_date_tool]

llm = ChatOllama(model="llama3.1:8b", temperature=0.1)

prompt = hub.pull("sanchezzz/russian_react_chat")

agent = create_react_agent(llm=llm, tools=TOOLS, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)

result = agent_executor.invoke({
    "input": "Сколько точно лет прошло с появления телепередачи Поле чудес в эфире? Кто сегодня ведущий телепередачи Поле чудес?",
    "chat_history": []
})

print(result["output"])