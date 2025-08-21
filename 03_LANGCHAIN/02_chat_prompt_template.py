import datetime
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


now_datetime = datetime.datetime.now(datetime.timezone.utc)


# PromptTemplate - это как f-строки
prompt_template = PromptTemplate.from_template("Ты полезный ассистент. Текущая дата: {current_date} и время {current_time}")
prompt_value = prompt_template.invoke({
    "current_date": now_datetime.date(),
    "current_time": now_datetime.time()
})
print(prompt_value.text)
print("\n______________________\n")


# ChatPromptTemplate - это как массив f-строк
messages = [
    ("system", "Ты полезный ассистент. Текущая дата: {current_date} и время {current_time}"),
    ("human", "Какое сейчас время?")
]
prompt_template = ChatPromptTemplate(messages=messages)
prompt_value = prompt_template.invoke({
    "current_date": now_datetime.date(),
    "current_time": now_datetime.time()
})
print(prompt_value.to_messages())
print("\n______________________\n")

# MessagesPlaceholder - заглушка при создании массива сообщений,
# чтобы на её место вставить одно или несколько сообщений по ключу загрушки в методе invoke
messages = [
    ("system", "Ты полезный ассистент. Текущая дата: {current_date} и время {current_time}"),
    MessagesPlaceholder(variable_name="sexy_name") # Ну или для истории можно использовать -_-
]
prompt_template = ChatPromptTemplate(messages)
prompt_value = prompt_template.invoke(input={
    "current_date": now_datetime.date(),
    "current_time": now_datetime.time(),
    "sexy_name": [HumanMessage("Какое сейчас время?")]
})
print(prompt_value.to_messages())
