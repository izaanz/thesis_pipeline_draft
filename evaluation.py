import torch
import subprocess
import tempfile
import os
import gc
import pandas as pd
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import uuid 
import ast 

# Import our custom reward function
# We need it here to score the generations
from reward_model import execute_and_reward

# 0. Setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
base_model_name = "codellama/CodeLlama-7b-Instruct-hf"
data_dir = "data/formatted_dataset"

#  Define the pass@k Estimator 
def unbiased_pass_at_k(n, c, k):
    """Calculates the unbiased pass@k estimator."""
    if n - c < k: return 1.0
    if n <= 0 or k <= 0 or n < k: return 0.0
    if n - c + 1 > n + 1: return 1.0

    term = 1.0
    for i in range(k):
      if n - i == 0:
        term = 0.0
        break
      term *= (n - c - i) / (n - i)
      if term == 0.0: break

    return 1.0 - term


#  Define the Evaluation Harness (WITH BATCHED GENERATION)
def evaluate_model_pass_at_k(model, tokenizer, test_dataset, n_samples=20, k_values=[1, 5, 10], eval_batch_size=5):
    """Evaluates a model on a test dataset and computes pass@k scores."""
    results = {f"pass@{k}": [] for k in k_values}
    # Use left-padding for batch generation
    tokenizer.padding_side = 'left' 

    print(f"Evaluating model on {len(test_dataset)} prompts (n_samples={n_samples}, batch_size={eval_batch_size})...")
    
    for sample_idx, sample in enumerate(test_dataset):

        original_code = sample['prompt']
        entry_point = sample['entry_point']

        # Format prompt (same as PPO/DPO)
        prompt_content = f"Generate a few high-quality Python unit tests (e.g., 5-7) for the following function:\n\n```python\n{original_code}\n```"
        messages = [{"role": "user", "content": prompt_content}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)

        # Generation kwargs for evaluation (low temp for less randomness)
        eval_gen_kwargs = {
             "max_new_tokens": 512,
             "pad_token_id": tokenizer.eos_token_id,
             "eos_token_id": tokenizer.eos_token_id,
             "do_sample": True,
             "temperature": 0.2,
             "top_k": 50,
             "repetition_penalty": 1.1
        }

        num_correct = 0

        # Generate n_samples in smaller batches to avoid OOM
        for i in range(0, n_samples, eval_batch_size):
            current_batch_size = min(eval_batch_size, n_samples - i)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    num_return_sequences=current_batch_size,
                    **eval_gen_kwargs
                )

            batch_candidates = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Score each candidate
            num_correct += sum(1 for c in batch_candidates if execute_and_reward(original_code, c, entry_point) == 1.0)

            del outputs
            gc.collect()
            torch.cuda.empty_cache()

        # Logging: Print pass/fail for this prompt
        passed = num_correct > 0
        print(f"  Prompt {sample_idx + 1}/{len(test_dataset)}: {'✅ Passed' if passed else '❌ Failed'} ({num_correct}/{n_samples} correct)")

        # Calculate pass@k using total n_samples and num_correct
        for k in k_values:
            score = unbiased_pass_at_k(n_samples, num_correct, k)
            results[f"pass@{k}"].append(score)

    tokenizer.padding_side = 'right' # Set it back just in case
    final_scores = {key: np.mean(val) for key, val in results.items()}
    return final_scores

# -Setup Evaluation 
num_test_prompts = 5 # Reduced to 5 for a quick test
k_values = [1, 5, 10]
n_samples_per_prompt = 10
evaluation_batch_size = 5

models_to_evaluate = {
    "SFT (Baseline)": "output/sft_model_final",
    "PPO": "output/ppo_model_final",
    "DPO": "output/dpo_model_final"
}

# Load the held-out test set
test_dataset = load_from_disk(data_dir)['test']
test_sample = test_dataset.select(range(num_test_prompts)) # Select 5
results = {}

# Define Model Loading Configs
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# The Evaluation Loop 
for model_name, adapter_path in models_to_evaluate.items():
    print(f"\n--- Evaluating {model_name} ---")

    # Check if adapter path exists
    if not os.path.exists(adapter_path):
        print(f"⚠️ Warning: Adapter path not found: {adapter_path}. Skipping evaluation.")
        results[model_name] = {f"pass@{k}": 0.0 for k in k_values}
        continue

    # 1. Load the base model in 4-bit
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 2. Load the PEFT adapter
    try:
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
        )
        model.eval()
        print(f"✅ Successfully loaded {model_name} adapters from {adapter_path}.")
    except Exception as e:
        print(f"❌ Error loading adapters for {model_name} from {adapter_path}: {e}")
        results[model_name] = {f"pass@{k}": 0.0 for k in k_values}
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
        continue

    # 3. Run evaluation
    try:
        scores = evaluate_model_pass_at_k(
            model,
            tokenizer,
            test_sample,
            n_samples=n_samples_per_prompt,
            k_values=k_values,
            eval_batch_size=evaluation_batch_size
        )
        results[model_name] = scores
    except Exception as e:
        print(f"❌ Error during evaluation for {model_name}: {e}")
        results[model_name] = {f"pass@{k}": 0.0 for k in k_values}

    # 4. Clean up memory
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()

# Display Final Results 
print("\n\n--- 🏆 Final Evaluation Results 🏆 ---")
df_results = pd.DataFrame(results).T
df_results.columns = [f"pass@{k}" for k in k_values]

# Format as percentages, handle potential NaN
for col in df_results.columns:
    df_results[col] = df_results[col].apply(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "N/A")

print(df_results)
print("\n✅ Evaluation complete.")
