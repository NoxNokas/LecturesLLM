# По сравнению с техникой "Zero-shot prompting" в этой добавляется один или несколько примеров,
# чтобы показать модели, как нужно отвечать или решать задачу.


import openai

API_KEY = "ollama"
BASE_URL = "http://localhost:11434/v1"
MODEL_NAME = "llama3.1:8b"


client = openai.OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


system_message = """\
Classify the text into neutral, negative or positive
Examples:
1) Text: Wow that movie was rad!
AI: positive
2) Text: He is so bad!
AI: negative

Text: 
"""

user_message = "I think the vacation is okay."

chat_completion = client.chat.completions.create(
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

print(chat_completion.choices[0].message.content)