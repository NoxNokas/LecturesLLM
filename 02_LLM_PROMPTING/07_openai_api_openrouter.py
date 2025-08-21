# Используем интернет сервис "https://openrouter.ai/" для работы с моделями.


import openai


API_KEY = "..."
BASE_URL = "https://openrouter.ai/api/v1"

client = openai.OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": """Ты классифицируешь живые организмы по царствам: Животные, Растения, Грибы и Бактерии.
К какому царству относится следующее описание живого огранизма: """
        },
        {
            "role": "user",
            "content": "Подберёзовик"
        }
    ],
    model="deepseek/deepseek-chat-v3-0324:free",
    temperature=0.1
)


print(completion.choices[0].message.content)