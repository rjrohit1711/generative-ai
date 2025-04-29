from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "google/gemma-1.1-1b-it"   # Correct model id for 1B IT version

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Simple inference
prompt = "Write a short story about a dragon and a cat."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
