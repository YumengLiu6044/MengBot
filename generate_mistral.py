from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
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
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=base_model_id,
    quantization_config=bnb_config,  
    trust_remote_code=True,
    token=True
)

eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True, use_fast=True)
# unk token was used as pad token during finetuning, must set the same here
eval_tokenizer.pad_token = eval_tokenizer.unk_token
ft_model = PeftModel.from_pretrained(base_model, lora_weights)


device = torch.device("cuda")
print(device)
ft_model.to(device)
ft_model.eval()

print(ft_model)

# end load model

def generate_with_model(eval_prompt, temperature, repetition_penalty, custom_stop_tokens, max_new_tokens):        
    model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        if custom_stop_tokens is None:
            model_output = ft_model.generate(**model_input, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, temperature=temperature)[0]
        else:
            model_output = ft_model.generate(**model_input, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, stop_strings=custom_stop_tokens.split(","), tokenizer=eval_tokenizer, temperature=temperature)[0]

        text_output = eval_tokenizer.decode(model_output, skip_special_tokens=True)
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
    
    if len(history[chat_id]) > 100:
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
    output = response.split("<start_header_id>system<end_header_id>")
    if len(output) < 1:
        return ""

    output = output[1]

    output = output.split("<|eot_id|>")
    if len(output) == 0:
        return ""

    output = output[0].strip()

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
    output = generate_with_model(eval_prompt=collected_prompt, temperature=query.temperature, repetition_penalty=query.repetition_penalty, custom_stop_tokens=query.custom_stop_tokens, max_new_tokens=query.max_new_tokens)
    output = extract_output(output, history, query.chat_id)
    return {"generated_text": output}

# if __name__ == "__main__":
#     query = CompletionQuery()
#     history = {}
#     print(convert_input(query, history))
#     print()
#     print(f"History: {history}")
#     sampleOutput = "<start_header_id>user<end_header_id>Hello how are you?<|eot_id|><start_header_id>system<end_header_id>Not bad<|eot_id|>"
#     print(f"Output: {extract_output(sampleOutput, history, query.chat_id)}")
#     print(f"History: {history}")
#     query.prompt = "I went to starbucks today"
#     print(convert_input(query, history))
#     print()
#     print(f"History: {history}")
#     sampleOutput = "<start_header_id>user<end_header_id>Hello how are you?<|eot_id|>\
#     <start_header_id>system<end_header_id>Not bad<|eot_id|> \
#     <start_header_id>user<end_header_id>I went to star bucks today?<|eot_id|>\
#     <<start_header_id>system<end_header_id>That's cool<|eot_id|>>"
#     print(f"Output: {extract_output(sampleOutput, history, query.chat_id)}")
#     print(f"History: {history}")
