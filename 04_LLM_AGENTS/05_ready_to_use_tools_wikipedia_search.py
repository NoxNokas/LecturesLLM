# Для доступа к wikipedia можно использовать тул, предоставляющий информацию о кратком содержании страницы по запросу.
# Доступ полностью бесплатный
from langchain.agents import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="ru"))
wikipedia_tool = Tool(name="wikipedia", func=wikipedia.run, description="Search in Wikipedia knowledge database.")
result = wikipedia_tool.invoke("Большие языковые модели")
print(result)

# Объект wikipedia не является тулом и не является Runnable, но его легко можно превратить в тул
# с помощью класса Tool, где можно задать название, описание и вызов тула.