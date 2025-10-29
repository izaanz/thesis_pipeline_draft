import torch
import gc
import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig, 
    AutoModelForCausalLM, 
    DataCollatorWithPadding
)
from peft import PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from huggingface_hub import login

# Import our custom reward function
from reward_model import execute_and_reward

# Setup 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("HF_TOKEN environment variable not set. Please set it.")
else:
    login(token=hf_token)

base_model_name = "codellama/CodeLlama-7b-Instruct-hf"
sft_model_path = "output/sft_model_final"
ppo_output_dir = "output/ppo_model_final"
data_dir = "data/formatted_dataset"

os.makedirs(ppo_output_dir, exist_ok=True)


# Load Tokenizer & PPO Eval Data
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

# We use the 'test' split for PPO training
eval_dataset = load_from_disk(data_dir)['test']

# Format and TOKENIZE Prompts for PPO 
def format_ppo_prompt_and_tokenize(sample):
    
    # This is the prompt we'll show the model
    prompt_content = f"Generate a few high-quality Python unit tests (e.g., 5-7) for the following function:\n\n```python\n{sample['prompt']}\n```"
    
    messages = [{"role": "user", "content": prompt_content}]
    # 'add_generation_prompt=True' adds the '[/INST]' token
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # We tokenize just the prompt
    tokenized_prompt = tokenizer(prompt_str, truncation=True, max_length=512, padding=False) 
    
    return {
        "input_ids": tokenized_prompt['input_ids'], 
        "prompt_text": prompt_str, 
        "prompt": sample['prompt'], # This is the raw code
        "entry_point": sample['entry_point'] 
    }

print("Tokenizing PPO dataset...")
ppo_dataset = eval_dataset.map(
    format_ppo_prompt_and_tokenize,
    remove_columns=[col for col in eval_dataset.column_names if col not in ['prompt', 'entry_point']]
)
print(f"Example PPO prompt: {ppo_dataset[0]['prompt_text']}")


#  Load PPO Model (SFT model + ValueHead) 
print("Loading PPO model (SFT + ValueHead)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

sft_model = PeftModel.from_pretrained(base_model, sft_model_path)
# This adds the ValueHead layer needed for PPO
model = AutoModelForCausalLMWithValueHead(sft_model)
model.is_peft_model = True
print("✅ PPO model loaded.")

# -Configure PPO 
# These settings are important for low-VRAM training (took me a day to undertstand how to manage it on T4x2 GPU!)
ppo_config = PPOConfig(
    batch_size=4,
    learning_rate=5e-6,      # Lower LR for RL
    mini_batch_size=1,       # Must set this
    gradient_accumulation_steps=4, # (1 * 4 = 4, matches batch_size)
    kl_penalty="kl",
    target_kl=0.1,
    adap_kl_ctrl=True
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("✅ Data collator for padding initialized.")

# Create PPO Trainer 
print("Initializing PPOTrainer...")
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None, # We use the SFT model as the reference
    tokenizer=tokenizer,
    dataset=ppo_dataset,
    data_collator=data_collator
)
print("✅ PPOTrainer initialized.")

device = ppo_trainer.accelerator.device
print(f"Trainer is forcing all tensors to device: {device}")
model.v_head = model.v_head.to(device)
print(f"✅ Manually moved v_head to {device}.")


# The PPO Training Loop 
print("\nStarting PPO training...")

generation_kwargs = {
    "max_new_tokens": 512,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "do_sample": True,
    "temperature": 0.5,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3
}

# We'll train on 3 optimization steps for this demo.
for i, batch in enumerate(ppo_trainer.dataloader):
    if i >= 3: break # 3 optimization steps

    query_tensors_2d = batch["input_ids"].to(device)
    attention_mask_2d = batch["attention_mask"].to(device)

    # Need to convert 2D padded tensors back to a list of 1D tensors
    queries = []
    for j in range(query_tensors_2d.shape[0]):
        prompt_len = attention_mask_2d[j].sum()
        queries.append(query_tensors_2d[j, :prompt_len])

    # 1. GENERATE
    print(f"--- PPO Step {i+1}/3: Generating responses ---")
    response_tensors_cpu = ppo_trainer.generate(queries, **generation_kwargs)
    responses = [r.to(device) for r in response_tensors_cpu]

    batch_responses_text = []
    for j in range(len(queries)):
        query_len = queries[j].shape[0]
        response_part = responses[j][query_len:]
        decoded_text = tokenizer.decode(response_part, skip_special_tokens=True)
        batch_responses_text.append(decoded_text)
        print(f"\n--- GENERATED RESPONSE {j} ---\n{decoded_text}\n----------------------------")


    # 2. REWARD
    rewards_list = []
    start_idx = i * ppo_config.batch_size
    for j, response_text in enumerate(batch_responses_text):
        sample_idx = start_idx + j
        original_sample = ppo_dataset[sample_idx]

        original_code = original_sample['prompt']
        entry_point = original_sample['entry_point']

        # Get the score from our reward function
        reward_val = execute_and_reward(original_code, response_text, entry_point)
        print(f"--- 💰 REWARD ASSIGNED (Response {j}): {reward_val} ---")
        rewards_list.append(reward_val)

    # Normalize rewards (good for PPO stability)
    r_tensor = torch.tensor(rewards_list, device=device, dtype=torch.float)
    r_norm = (r_tensor - r_tensor.mean()) / (r_tensor.std(unbiased=False) + 1e-8)
    rewards = [r_norm[k].unsqueeze(0) for k in range(len(rewards_list))]

    print(f"--- 📊 REWARDS (Original): {r_tensor.cpu().numpy()} ---")
    print(f"--- 📊 REWARDS (Normalized): {r_norm.cpu().numpy()} ---")

    # 3. Manually create response_masks
    response_masks = []
    for query, response in zip(queries, responses):
        query_len = query.shape[0]
        response_len = response.shape[0] - query_len
        
        query_mask = torch.zeros(query_len, dtype=torch.long, device=device)
        response_mask = torch.ones(response_len, dtype=torch.long, device=device)
        
        response_masks.append(torch.cat((query_mask, response_mask), dim=0))

    # 4. RUN UPDATE
    print(f"\n--- Running PPO optimization step ---")
    stats = ppo_trainer.step(queries, responses, rewards, response_masks)
    ppo_trainer.log_stats(stats, {}, rewards)
    print(f"--- PPO Optimization Step complete. ---")

    # Memory cleanup
    del stats, queries, responses, rewards, response_masks, query_tensors_2d, attention_mask_2d, response_tensors_cpu, r_tensor, r_norm
    gc.collect()
    torch.cuda.empty_cache()

print("\n🎉 PPO training complete.")

# Save Model & Clean Up 
model.pretrained_model.save_pretrained(ppo_output_dir)
print(f"✅ PPO model saved to {ppo_output_dir}")

del model, base_model, sft_model, ppo_trainer, ppo_dataset
gc.collect()
torch.cuda.empty_cache()
