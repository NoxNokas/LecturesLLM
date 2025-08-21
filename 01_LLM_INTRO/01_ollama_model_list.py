# Получение списка установленных моделей в Ollama

import json
import requests

OLLAMA_HOST = "localhost"
OLLAMA_PORT = "11434"
GET_MODELS_ENDPOINT = "api/tags"

URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/{GET_MODELS_ENDPOINT}"

with open("model_list.json", "wt", encoding="UTF-8", newline="") as f:
    response = requests.get(URL)
    data = response.json()
    json.dump(data, f, indent=2)

