
# Sample request file

import requests
import json

url = "http://35.208.196.115:9092/generate"


content = {
    "chat_id": "i dont exist",
    "prompt": "Sure, in 10 min?",
    "temperature": 0.5,
    "max_new_tokens": 100,
    "repetition_penalty": 1.15,
    "custom_stop_tokens": "<|eot_id|>"
}

response = requests.post(url, json=content)
obj = json.loads(response.text)
print(obj["generated_text"])