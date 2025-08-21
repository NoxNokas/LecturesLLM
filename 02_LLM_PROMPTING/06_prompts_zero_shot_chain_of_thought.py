# В этом случае достаточно будет одного системного сообщения.
# Используется волшебная фраза “Think step by step“, которая призывает модель рассуждать по шагам и в итоге приходить к ответу. 

import openai

API_KEY = "ollama"
BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "llama3.1:8b"


client = openai.OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

system_message = """\
Solve the task. Think step by step and give answer in format "Answer is True or False"
"""

user_message = "Task: Check if the odd numbers in this group add up to an even number: 17,  10, 19, 4, 8, 12, 24"


chat_comletion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_message
        }
    ],
    model=MODEL_NAME,
    temperature=0.1
)


print(chat_comletion.choices[0].message.content)