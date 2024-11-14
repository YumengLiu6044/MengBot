from transformers import (
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    AutoTokenizer,
    pipeline
    )
import torch
from peft import PeftModel
from fire import Fire



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import Dict, List
import os


# load model
base_model_id = os.getenv("base_model_id")
lora_weights = os.getenv("lora_weights")


bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)


base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=base_model_id,
    quantization_config=bnb_config,  
    trust_remote_code=True,
    token=True
)

eval_tokenizer = AutoTokenizer.from_pretrained(
    base_model_id, 
    add_bos_token=True, 
    trust_remote_code=True, 
    use_fast=True
    )
# unk token was used as pad token during finetuning, must set the same here
eval_tokenizer.pad_token = eval_tokenizer.unk_token
ft_model = PeftModel.from_pretrained(base_model, lora_weights)


device = torch.device("cuda")
print(device)
ft_model.to(device)
ft_model.eval()

# end load model

generator = pipeline(
    "text-generation",
    model=ft_model,
    tokenizer=eval_tokenizer
)

def generate_with_model(eval_prompt, temperature, repetition_penalty, custom_stop_tokens, max_new_tokens):        
    output = generator(
        eval_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        return_full_text=False  # Ensures only new tokens are returned
    )
    # Extract the generated text
    text_output = output[0]["generated_text"]
    print(text_output)
    return text_output

# start the fast api app

app = FastAPI()

history = {}

class CompletionQuery(BaseModel):
    chat_id: str = "oiwehfibv89b98h3"
    prompt: str = "Hello how are you?"
    temperature: float = 0.5
    max_new_tokens: int = 100
    repetition_penalty: float = 1.15
    custom_stop_tokens: str = "<|eot_id|>"


def append_to_history(paired, history: dict, chat_id: str):
    if chat_id not in history:
        history[chat_id] = [paired]

    else:
        history[chat_id].append(paired)
    
    if len(history[chat_id]) > 10:
        history[chat_id].pop(0)


def convert_input(query: CompletionQuery, history: dict) -> str:
    chat_id = query.chat_id
    prompt = query.prompt

    # Append message history by key
    paired = ("user", prompt)
    append_to_history(paired, history, chat_id)

    # Append to template
    template = ""
    for pair in history[chat_id]:
        role = "user" if pair[0] != "system" else "system"
        template += f"<start_header_id>{role}<end_header_id>{pair[1]}<|eot_id|>\n"

    template += "<start_header_id>system<end_header_id>"
    return template


def extract_output(response: str, history: dict, chat_id: str) -> str:
    output = response.split("<start_header_id>user<end_header_id>")[0]
    output = output.split("<|eot_id|>")[0].strip()
    output = output.split("<start_header_id>user<end_header_id>")
    output = [x.strip() for x in output]
    output = list(filter(lambda a: a, output))
    output = "\n".join(output)

    # Append to history
    paired = ("system", output)
    append_to_history(paired, history, chat_id)

    return output


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/generate/")
async def generate(query: CompletionQuery):    
    global history

    collected_prompt = convert_input(query, history)
    output = generate_with_model(
        eval_prompt=collected_prompt, 
        temperature=query.temperature, 
        repetition_penalty=query.repetition_penalty, 
        custom_stop_tokens=query.custom_stop_tokens, 
        max_new_tokens=query.max_new_tokens
    )
    output = extract_output(output, history, query.chat_id)
    return {"generated_text": output}
