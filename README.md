# LLM Finetuning Pipeline for Unit Test Generation

This repository contains a basic, end-to-end pipeline for finetuning a large language model (**CodeLlama-7b-Instruct**) to generate Python unit tests from function code. This was built as a proof-of-concept for a thesis to ensure the full pipeline (data processing, SFT, preference-based RL, and evaluation) works on consumer-grade hardware before scaling up.

---

## Project Idea

The core idea is to see if we can improve a base code model's ability to generate correct unit tests.

### Supervised Finetuning (SFT)
First, we teach the model the format of a unit test by showing it examples of functions and their corresponding tests. This is our baseline model.

### Reinforcement Learning (PPO / DPO)
Next, we try to teach the model what a good test is. Instead of just showing it examples, we let it generate tests and then give it a "reward" based on whether the test:
- Is valid Python syntax.  
- Compiles and runs.  
- Correctly passes.  

We use two different methods for this: **PPO (Proximal Policy Optimization)** and **DPO (Direct Preference Optimization)**.

### Evaluation
Finally, we compare all three models (SFT, PPO, DPO) to see which one got better at the task using the **pass@k** metric (i.e., "what's the probability the model generates at least one correct test in *k* attempts?").

---

## Pipeline

The code is broken into sequential scripts:

- **`data_preparation.py`**: Loads the HumanEval and MBPP datasets and formats them into (prompt, completion) pairs.  
- **`sft_training.py`**: Trains the baseline SFT model on the formatted data.  
- **`reward_model.py`**: A utility script that defines our reward function. It runs the generated code in a sandbox and returns a score.  
- **`ppo_training.py`**: Uses the SFT model and the reward function to conduct PPO training.  
- **`dpo_dataset_creation.py`**: Generates a preference dataset (chosen vs. rejected tests) by scoring multiple SFT model outputs with the reward function.  
- **`dpo_training.py`**: Uses the preference dataset to conduct DPO training.  
- **`evaluation.py`**: Runs the pass@k evaluation harness on all three finetuned models.  

---

## Current Results (Proof-of-Concept)

> **Note:** These results are based on a tiny amount of data and minimal training steps, just to prove the pipeline works. They are not representative of a fully trained model.

In this initial run, the **SFT (baseline)** model performed the best, and the **DPO** model performed the worst.

| Model | pass@1 | pass@5 | pass@10 |
|--------|--------|--------|---------|
| SFT (Baseline) | 16.00% | 52.22% | 60.00% |
| PPO | 6.00% | 30.00% | 60.00% |
| DPO | 0.00% | 0.00% | 0.00% |

This suggests that with very limited data, the RLHF steps (PPO/DPO) can actually hurt performance (a "mode collapse" or "catastrophic forgetting"). The next step in the thesis would be to scale up the data and training time significantly to see if PPO/DPO eventually overtake the SFT baseline, as expected.

---

## How to Run

### 1. Setup

First, create a Python environment and install the required libraries.

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate
```
#### Install requirements
```bash
pip install -r requirements.txt
```

You will also need to set your Hugging Face token as an environment variable or secret.

```bash
export HF_TOKEN="your_huggingface_token_here"
```

### 2. Run the Pipeline

The scripts are numbered and intended to be run in order.
```bash
# 1. Process the data
python data_preparation.py

# 2. Run Supervised Finetuning
python sft_training.py

# 3. (Optional) Run PPO Training
# Note: PPO is compute-intensive
python ppo_training.py

# 4. (Optional) Create the DPO dataset
python dpo_dataset_creation.py

# 5. (Optional) Run DPO Training
python dpo_training.py

# 6. Evaluate all trained models
python evaluation.py
```


pip install -r requirements.txt
