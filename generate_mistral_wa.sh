base_model_id="alpindale/Mistral-7B-v0.2-hf" lora_weights="../model/output/" CUDA_VISIBLE_DEVICES=0 uvicorn generate_mistral:app --port 443 --host 0.0.0.0 --reload
