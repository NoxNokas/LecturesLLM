#  Запуск скачанной модели через ollama с помощью библиотеки  openai


import openai


API_KEY = "ollama"
BASE_URL = "http://localhost:11434/v1/"
MODEL_NAME = "llama3:latest"

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

message = "Привет! Когда ждать появления AGI?"

chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Ты чокнутый профессор, который думает, что AGI сидит у него в подвале",
            },
            {
                "role": "user",
                "content": message
            }
        ],
    model=MODEL_NAME,
    temperature=0.1
)

print(chat_completion.choices[0].message.content)