# RunnableLambda - превращение вызываемых объектов в Runnable-объекты
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableParallel, RunnablePassthrough
import math


print("RunnableLambda")
square_runnable = RunnableLambda(lambda x: x**2)
result = square_runnable.invoke(10)
print(result)
print("_" * 10)

# RunnableSequence - объединение Runnable-объектов
print("RunnableSequence")
square_runnable = RunnableLambda(lambda x: x**2)
add_10_runnable = RunnableLambda(lambda x: x + 10)
log_runnable = RunnableLambda(lambda x: math.log(x))
pipeline = RunnableSequence(square_runnable, add_10_runnable, log_runnable)
result = pipeline.invoke(10)
print(result)
## способ через "|"
pipeline = square_runnable | add_10_runnable | log_runnable
result = pipeline.invoke(10)
print(result)
print("_" * 10)

# RunnableParallel - параллельное выполнение Runnable-объектов: наподобие батча
print("RunnableParallel")
square_runnable = RunnableLambda(lambda x: x ** 2)
add_10_runnable = RunnableLambda(lambda x: x + 10)

chain = RunnableParallel(square_result=square_runnable, add_10_result=add_10_runnable)
result = chain.invoke(10)
print(result)
## способ через "|" с шагом, где выполняется RunnableParallel, результат которого можно дальше проталкивать
chain = square_runnable | {"some": add_10_runnable, "other": log_runnable}
result = chain.invoke(10)
print(result)
print("_" * 10)

# RunnablePassthrough - возвращает входные параметры вместе с результатом
print("RunnablePassthrough")
square_runnable = RunnableLambda(lambda x: x["initial_value"] ** 2)
chain = RunnablePassthrough.assign(square_result=square_runnable)
result = chain.invoke({"initial_value": 2})
print(result)