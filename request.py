
# Sample request file

import requests
import json

url = "https://{IP_ADDRESS}:{PORT}/generate"


content = {
    "chat_id": "i dont exist",
    "prompt": "Sure, in 10 min?",
    "temperature": 0.5,
    "max_new_tokens": 100,
    "repetition_penalty": 1.15,
    "custom_stop_tokens": "<|eot_id|>"
}

response = requests.post(url.format(IP_ADDRESS="34.16.184.1", PORT="443"), json=content)
obj = json.loads(response.text)
print(obj["generated_text"])