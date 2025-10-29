import torch
import gc
import os
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
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
data_dir = "data/formatted_dataset"
dpo_dataset_dir = "data/my_dpo_dataset"

os.makedirs(dpo_dataset_dir, exist_ok=True)

# Load SFT Model (to generate candidates)
print("Loading SFT model for DPO dataset creation...")
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
sft_model.eval()
print("✅ SFT model loaded.")

#  Load Tokenizer & Train Data 
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Important for generation

# Use the train split to generate preferences
train_dataset = load_from_disk(data_dir)['train']

# Generation Config & Helper
generation_kwargs = {
    "max_new_tokens": 512, # Use the same length as PPO
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "do_sample": True,
    "temperature": 0.5, # Use same temp as PPO
    "top_k": 50,
    "repetition_penalty": 1.1, # Lowered this
}

#  Generate & Label Preference Pairs
preference_data = []
num_candidates_per_prompt = 4 # Generate 4 options per prompt
num_prompts_to_process = 50   # Use 50 prompts for a decent demo dataset

print(f"Generating preference pairs from {num_prompts_to_process} prompts (k={num_candidates_per_prompt})...")

for i, sample in enumerate(tqdm(train_dataset.select(range(num_prompts_to_process)))):

    original_code = sample['prompt']
    entry_point = sample['entry_point']

    # Format the prompt using the simple PPO format
    prompt_content = f"Generate a few high-quality Python unit tests (e.g., 5-7) for the following function:\n\n```python\n{original_code}\n```"
    messages = [{"role": "user", "content": prompt_content}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(sft_model.device)

    # 1. Generate N candidates
    with torch.no_grad():
        generated_outputs = sft_model.generate(
            **inputs,
            num_return_sequences=num_candidates_per_prompt,
            **generation_kwargs
        )

    # 2. Decode and get rewards using the function
    candidates = []
    for output in generated_outputs:
        decoded_text = tokenizer.decode(output[inputs.input_ids.shape[1]:], skip_special_tokens=True)
        reward = execute_and_reward(original_code, decoded_text, entry_point)
        candidates.append({"text": decoded_text, "reward": reward})

    # 3. Create preference pairs based on reward
    #    We want pairs where (chosen_reward > rejected_reward)
    ranked_candidates = sorted(candidates, key=lambda x: x['reward'], reverse=True)

    chosen, rejected = None, None
    if len(ranked_candidates) >= 2 and ranked_candidates[0]['reward'] > ranked_candidates[-1]['reward']:
        chosen = ranked_candidates[0]['text']   # Highest reward
        rejected = ranked_candidates[-1]['text'] # Lowest reward

    # 4. Add to dataset
    if chosen and rejected:
        preference_data.append({
            "prompt": original_code, # Store the raw function code
            "chosen": chosen,        # Store the raw generated test
            "rejected": rejected
        })

    del inputs, generated_outputs, candidates, ranked_candidates
    gc.collect()
    torch.cuda.empty_cache()

# 5. Save Dataset to Disk & Clean Up 
dpo_dataset = Dataset.from_list(preference_data)
dpo_dataset.save_to_disk(dpo_dataset_dir)
print(f"\n✅ DPO preference dataset with {len(dpo_dataset)} pairs saved to {dpo_dataset_dir}.")

del base_model, sft_model, dpo_dataset, train_dataset
gc.collect()
torch.cuda.empty_cache()
