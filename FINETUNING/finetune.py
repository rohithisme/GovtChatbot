import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, 
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Ensure single GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Step 3: Load and Preprocess Dataset
DATASET_PATH = "/workspace/FINETUNING/dataset/ml_dataset.json"
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Load tokenizer
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

def preprocess(example):
    prompt = f"{example['instruction']}\nInput: {example['input']}\nResponse:"
    response = example["output"]
    
    # Ensure consistent max length and better tokenization
    max_length = 512
    input_encoding = tokenizer(
        prompt, 
        truncation=True, 
        padding="max_length", 
        max_length=max_length, 
        return_tensors="pt"
    )
    output_encoding = tokenizer(
        response, 
        truncation=True, 
        padding="max_length", 
        max_length=max_length, 
        return_tensors="pt"
    )
    
    return {
        "input_ids": input_encoding["input_ids"].squeeze(),
        "attention_mask": input_encoding["attention_mask"].squeeze(),
        "labels": output_encoding["input_ids"].squeeze()
    }

# Create dataset
dataset = Dataset.from_list(data).map(
    preprocess, 
    remove_columns=["instruction", "input", "output"],
    batched=False
)

# Extreme memory optimization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load model with explicit device mapping
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    quantization_config=bnb_config,
    device_map={"": 0},  # Force to first GPU
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Prepare model for k-bit training - CRITICAL ADDITION
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,  # Reduced rank
    lora_alpha=8,  # Reduced alpha
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model with LoRA
model = get_peft_model(model, lora_config)

# Enable gradient checkpointing with a more robust method
model.enable_input_require_grads()
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# Print trainable parameters
model.print_trainable_parameters()

# Training arguments with memory-efficient settings
training_args = TrainingArguments(
    output_dir="./llama-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    warmup_steps=50,
    max_steps=500,
    learning_rate=5e-5,
    fp16=True,
    save_strategy="steps",
    save_steps=250,
    logging_dir="./logs",
    logging_steps=10,
    remove_unused_columns=True,
    dataloader_num_workers=1,
    dataloader_pin_memory=False,
    optim="adamw_torch_fused",
    local_rank=-1,
    no_cuda=False
)

# Data collator with more robust configuration
data_collator = DataCollatorForSeq2Seq(
    tokenizer, 
    model=model, 
    padding=True,
    max_length=512,
    return_tensors="pt"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training with comprehensive error handling
try:
    print("Starting training...")
    trainer.train()
except Exception as e:
    print(f"Training interrupted: {e}")
    import traceback
    traceback.print_exc()
    torch.cuda.empty_cache()

# Save Fine-Tuned Model
model.save_pretrained("./llama-finetuned")
tokenizer.save_pretrained("./llama-finetuned")

print("Training completed successfully!")