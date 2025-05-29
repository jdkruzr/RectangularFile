# Create a script called download_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

model_name = "Qwen/Qwen2-VL-7B"
print(f"Downloading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

print("Download complete!")