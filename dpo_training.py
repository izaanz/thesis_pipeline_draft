import torch
import gc
import os
from datasets import load_from_disk
from trl import DPOTrainer
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, PeftModel
from huggingface_hub import login

# . Set Env Var
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("HF_TOKEN environment variable not set. Please set it.")
else:
    login(token=hf_token)

base_model_name = "codellama/CodeLlama-7b-Instruct-hf"
sft_model_path = "output/sft_model_final"
dpo_output_dir = "output/dpo_model_final"
dpo_dataset_dir = "data/my_dpo_dataset"
checkpoints_dir = "output/dpo_model_checkpoints"

os.makedirs(dpo_output_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

#  Load DPO Dataset from Disk
print(f"Loading DPO preference dataset from {dpo_dataset_dir}...")
dpo_dataset = load_from_disk(dpo_dataset_dir)
print(f"✅ Loaded {len(dpo_dataset)} preference pairs.")

# Define Configs 
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

#  Load SFT Model (as base for DPO)
print("Loading base model and SFT adapters for DPO...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, sft_model_path)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
# DPO likes left-padding
tokenizer.padding_side = 'left'
print("✅ Models and tokenizer loaded.")

# Format Dataset for DPOTrainer
# This function formats the (prompt, chosen, rejected) into
# the chat template the model expects.
def format_dpo_dataset(example):
    # This is the prompt (same as PPO)
    prompt_messages = [
        {"role": "user", "content": f"Generate a few high-quality Python unit tests (e.g., 5-7) for the following function:\n\n```python\n{example['prompt']}\n```"},
    ]
    prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    # This is the "chosen" (good) completion
    chosen_messages = prompt_messages + [{"role": "assistant", "content": example['chosen']}]
    chosen_str = tokenizer.apply_chat_template(chosen_messages, tokenize=False) + tokenizer.eos_token

    # This is the "rejected" (bad) completion
    rejected_messages = prompt_messages + [{"role": "assistant", "content": example['rejected']}]
    rejected_str = tokenizer.apply_chat_template(rejected_messages, tokenize=False) + tokenizer.eos_token

    return {
        "prompt": prompt_str,
        "chosen": chosen_str,
        "rejected": rejected_str
    }

print("Applying DPO chat template formatting...")
formatted_dpo_dataset = dpo_dataset.map(format_dpo_dataset, remove_columns=dpo_dataset.column_names)

# Configure Training Arguments 
dpo_training_args = TrainingArguments(
    output_dir=checkpoints_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5, # Lower LR for DPO
    logging_steps=1, 
    max_steps=10,        # Forcing 10 steps just to test the pipeline
    save_strategy="no",
    bf16=True, # Use bf16 if available, fp16 otherwise
    report_to="none",
    remove_unused_columns=False,
    optim="paged_adamw_8bit", # Memory saving optimizer
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False}
)

# Create DPO Trainer 
print("Initializing DPOTrainer...")
model.enable_input_require_grads()

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None, # SFT model is the implicit reference
    args=dpo_training_args,
    train_dataset=formatted_dpo_dataset,
    tokenizer=tokenizer,
    peft_config=lora_config,
    beta=0.1, # DPO hyperparameter
    max_prompt_length=512,
    max_length=512
)
print("✅ DPOTrainer initialized.")

# -Start Training
print(f"\nStarting DPO training (forcing {dpo_training_args.max_steps} steps)...")
dpo_trainer.train()
print("DPO training complete.")

#  Save Final Model & Clean Up
dpo_trainer.save_model(dpo_output_dir)
print(f"✅ DPO model adapters saved to {dpo_output_dir}")

del model, base_model, dpo_trainer, dpo_dataset, formatted_dpo_dataset
gc.collect()
torch.cuda.empty_cache()
