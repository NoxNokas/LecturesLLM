# Решить квадратичное уравнение с помощью Runnable-объектов
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


a = float(input("Input a: "))
b = float(input("Input b: "))
c = float(input("Input c: "))

discriminant = RunnableLambda(lambda data: data['b'] ** 2 - (4 * data['a'] * data['c']))
discriminant = RunnablePassthrough.assign(discriminant=discriminant)
result = discriminant | RunnableLambda(
    lambda data: "Нет корней" if data['discriminant'] < 0 else
                 -data['b']/(2*data['a']) if data['discriminant'] == 0 else
                 ((-data['b'] - data['discriminant']**0.5)/(2*data['a']),
                  (-data['b'] + data['discriminant']**0.5)/(2*data['a'])
                 )
)
print(result.invoke({'a':a, 'b':b, 'c':c}))