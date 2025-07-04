# Utility script for downloading the model we use
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor
from transformers.models.qwen2.5_vl.modeling_qwen2.5_vl import Qwen2.5VLForConditionalGeneration

# Set cache directory explicitly
cache_dir = "/mnt/rectangularfile/qwencache"
os.environ['TRANSFORMERS_CACHE'] = cache_dir

# Create cache directory if it doesn't exist
Path(cache_dir).mkdir(parents=True, exist_ok=True)
print(f"Using cache directory: {cache_dir}")

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
print(f"Downloading {model_name}...")

# Download tokenizer and processor
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True,
    cache_dir=cache_dir
)

print("Downloading processor...")
processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=cache_dir
)

# Download the model using the specific model class
print("Downloading model (this will take some time)...")
model = Qwen2.5VLForConditionalGeneration.from_pretrained(
    model_name, 
    trust_remote_code=True,
    cache_dir=cache_dir,
    low_cpu_mem_usage=True
)

# Also create an offload directory for when the model runs
offload_dir = "/mnt/rectangularfile/qwencache/offload"
Path(offload_dir).mkdir(parents=True, exist_ok=True)
print(f"Created offload directory: {offload_dir}")

print(f"Download complete! Files saved to {cache_dir}")
print(f"To verify, check the directory size: du -sh {cache_dir}")
