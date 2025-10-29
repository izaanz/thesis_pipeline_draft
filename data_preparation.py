import os
import re
from datasets import load_dataset, concatenate_datasets

# Define the Data Formatting Function 
def format_prompt_for_unittesting(example, dataset_name):
    """
    Formats an example from HumanEval or MBPP to create a prompt
    for unit test generation.
    """
    if dataset_name == "humaneval":
        prompt_code = example['prompt'] + example['canonical_solution']
        entry_point = example['entry_point']
        raw_tests = example['test']
        
        # remove the "def check(candidate):" line
        cleaned_tests = raw_tests.replace("def check(candidate):", "", 1)
        
        # fix indentation of all lines
        lines = [line.strip() for line in cleaned_tests.splitlines() if line.strip()]
        indented_lines = "\n        ".join(lines) # 8 spaces (4 for class, 4 for def)
        
        # fix the assertions (on the newly indented lines)
        fixed_asserts = re.sub(r"assert (.*)", r"self.assertTrue(\1)", indented_lines, flags=re.MULTILINE)

        target_tests = (
            f"import unittest\n"
            f"# from {entry_point} import {entry_point}\n\n"
            f"class TestSolution(unittest.TestCase):\n"
            f"    def test_function(self):\n"
            f"        candidate = {entry_point}\n"
            f"        {fixed_asserts}" # this is now clean and correctly indented
        )
        
    elif dataset_name == "mbpp":
        prompt_code = example['text'] + "\n" + example['code']
        
        match = re.search(r"def\s+(\w+)\s*\(", example['code'])
        entry_point = match.group(1) if match else "solution_function"
        
        # fix the assertions for MBPP
        fixed_asserts_list = [
            re.sub(r"assert (.*)", r"self.assertTrue(\1)", s) for s in example['test_list']
        ]
        test_assertions = "\n        ".join(fixed_asserts_list)
        
        target_tests = (
            f"import unittest\n"
            f"# from {entry_point} import {entry_point}\n\n"
            f"class TestSolution(unittest.TestCase):\n"
            f"    def test_function(self):\n"
            f"        candidate = {entry_point}\n"
            f"        {test_assertions}" # use the fixed asserts
        )
        
    return {"prompt": prompt_code, "completion": target_tests, "entry_point": entry_point}

def main():
    # Load datasets and prepocess
    print("Loading and processing datasets...")
    humaneval_dataset = load_dataset("openai_humaneval", split="test")
    mbpp_dataset = load_dataset("RLAIF/mbpp", split="test") 

    humaneval_formatted = humaneval_dataset.map(format_prompt_for_unittesting, 
                                                fn_kwargs={"dataset_name": "humaneval"})
    mbpp_formatted = mbpp_dataset.map(format_prompt_for_unittesting, 
                                      fn_kwargs={"dataset_name": "mbpp"})

    # Combine the datasets
    columns_to_keep = ["prompt", "completion", "entry_point"]
    humaneval_formatted = humaneval_formatted.remove_columns(
        [c for c in humaneval_formatted.column_names if c not in columns_to_keep]
    )
    mbpp_formatted = mbpp_formatted.remove_columns(
        [c for c in mbpp_formatted.column_names if c not in columns_to_keep]
    )
    combined_dataset = concatenate_datasets([humaneval_formatted, mbpp_formatted])

    #  Shuffle and Spli
    print("Splitting dataset into train/test...")
    shuffled_dataset = combined_dataset.shuffle(seed=42)
    split_dataset = shuffled_dataset.train_test_split(test_size=0.1)

    # Save to Disk for Later Phases 
    output_dir = "data/formatted_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Full training set size: {len(split_dataset['train'])}")
    print(f"Full test set size: {len(split_dataset['test'])}")
    
    split_dataset.save_to_disk(output_dir)

    print(f"\n✅ Data ingestion and processing complete. Saved to {output_dir}")
    print(f"--- Example Processed Sample ---")
    print("PROMPT (Function + Solution):\n" + split_dataset['train'][0]['prompt'])
    print("\nCOMPLETION (Target Unit Test):\n" + split_dataset['train'][0]['completion'])

if __name__ == "__main__":
    main()
