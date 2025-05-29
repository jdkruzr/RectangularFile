# Utility script for downloading the model we use

from transformers import AutoTokenizer, AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

model_name = "Qwen/Qwen2-VL-7B"
print(f"Downloading {model_name}...")

# Download tokenizer and processor
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Download the model using the specific model class
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, 
    trust_remote_code=True
)

print("Download complete!")