import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Paths
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # Your original base model
ADAPTER_PATH = "/workspace/FINETUNING/llama-finetuned"
MERGED_MODEL_PATH = "/workspace/FINETUNING/llama-merged"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter and merge
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model = model.merge_and_unload()  # Merge LoRA weights into the base model

# Save the fully merged model
model.save_pretrained(MERGED_MODEL_PATH)
tokenizer.save_pretrained(MERGED_MODEL_PATH)

print(f"Model merged and saved successfully at {MERGED_MODEL_PATH}")
