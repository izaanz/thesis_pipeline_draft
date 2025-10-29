import torch
import gc
import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from huggingface_hub import login

# Setup 
# Set this env var for memory management (dont set it to colab directrory again (note to self)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Log in to Hugging Face
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("HF_TOKEN environment variable not set. Please set it.")
else:
    login(token=hf_token)

# This is the advanced model, CodeLlama-Instruct
base_model_name = "codellama/CodeLlama-7b-Instruct-hf"
sft_output_dir = "output/sft_model_final"
data_dir = "data/formatted_dataset"
checkpoints_dir = "output/sft_model_checkpoints"

os.makedirs(sft_output_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)


# Load Preprocessed Data 
print(f"Loading preprocessed SFT data from {data_dir}...")
train_dataset = load_from_disk(data_dir)['train']

# Load Tokenizer & Model Configs 

# bnb_config is for 4-bit quantization to save memory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# lora_config is for PEFT (LoRA)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
# SFTTrainer seems to like right-padding
tokenizer.padding_side = "right" 

#  3. Format Dataset with Chat Template
# This function formats the prompt and completion into the
# chat template that the Instruct model expects.
def apply_chat_template(example):
    messages = [
        {"role": "user", "content": f"Generate a Python unit test for the following function:\n\n```python\n{example['prompt']}\n```"},
        {"role": "assistant", "content": example['completion']}
    ]
    
    # SFTTrainer wants the full formatted string, including prompt and response
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False) + tokenizer.eos_token}

print("Applying correct chat template to SFT dataset...")
sft_train_dataset = train_dataset.map(apply_chat_template)
print(f"Sample formatted SFT prompt:\n{sft_train_dataset[0]['text']}")

# Load SFT Model 
print("Loading SFT base model (CodeLlama-Instruct)...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Configure Training
training_args = TrainingArguments(
    output_dir=checkpoints_dir,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=1000, 
    save_steps=100,
    fp16=True, 
    report_to="none" 
)

# Create & Run SFT Trainer 
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=sft_train_dataset,
    peft_config=lora_config,
    dataset_text_field="text", # We created this field in step 3
    max_seq_length=1024,
    args=training_args,
)

print("Starting SFT training (with correct format)...")
trainer.train()
print("SFT training complete.")

#  Save Final Model & Clean Up 
trainer.save_model(sft_output_dir)
print(f"✅ SFT model saved to {sft_output_dir}")

del model, trainer, sft_train_dataset, train_dataset
gc.collect()
torch.cuda.empty_cache()
