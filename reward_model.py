import subprocess
import tempfile
import os
import uuid
import ast

# This is the reward function. It runs the code and gives a score.
# This is the new reward logic: -1.0 (bad syntax), +0.1 (runtime error), +0.2 (failed test), +1.0 (passed)
def execute_and_reward(original_code: str, generated_test_str: str, entry_point: str, timeout_seconds: int = 10) -> float:
    
    # create unique filenames for this run
    file_id = str(uuid.uuid4())
    code_file_path = os.path.join(tempfile.gettempdir(), f"{entry_point}.py")
    test_file_path = os.path.join(tempfile.gettempdir(), f"test_{file_id}.py")
    
    generated_test_str = generated_test_str.strip()

    # Penalty for invalid syntax (AST check)
    # ast.parse checks if the code is valid Python
    try:
        ast.parse(generated_test_str)
    except (SyntaxError, IndentationError) as e:
        print(f"❌ AST Syntax/Indentation Error: -1.0 Reward. ({e})")
        return -1.0 # Large negative for garbage
    except Exception as e:
        print(f"❌ Unknown AST Error: -1.0 Reward. ({e})")
        return -1.0 # Also garbage

    # Graded reward for execution (Pass/Fail)
    try:
        # Write the function-to-be-tested to a file
        with open(code_file_path, "w") as f: f.write(original_code)
        # Write the generated test to a file
        with open(test_file_path, "w") as f: f.write(generated_test_str)
        
        # Try to compile the test file
        compile_process = subprocess.run(
            ["python", "-m", "py_compile", test_file_path],
            capture_output=True, text=True, timeout=timeout_seconds
        )
        if compile_process.returncode != 0:
            print(f"❌ Compile Error: +0.1 Reward. ({compile_process.stderr})")
            return 0.1 # Valid AST, but fails to compile

        # Try to run the test file using unittest
        run_process = subprocess.run(
            ["python", "-m", "unittest", "discover", "-s", tempfile.gettempdir(), "-p", f"test_{file_id}.py"],
            capture_output=True, text=True, timeout=timeout_seconds
        )
        output = run_process.stderr
        
        # Check the output of the unittest runner
        if run_process.returncode == 0 and "OK" in output:
            print("✅ Test Passed: +1.0 Reward.")
            return 1.0 # This is what we want!
        elif "FAILED" in output:
            print("⚠️ Test Failed (Logic Error): +0.2 Reward.")
            return 0.2 # Valid test, but it failed (logic error)
        else: # Any other runtime error
            print(f"❌ Runtime Error: +0.1 Reward. ({output})")
            return 0.1
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout Error: +0.1 Reward.") # Valid syntax, but it loops
        return 0.1
    except Exception as e:
        print(f"❌ Unknown Execution Error: +0.1 Reward. ({e})")
        return 0.1
    finally:
        # Clean up the temp files
        if os.path.exists(code_file_path): os.remove(code_file_path)
        if os.path.exists(test_file_path): os.remove(test_file_path)

if __name__ == "__main__":
    # Example usage for testing the reward function itself
    print("Testing reward function...")
    
    # 1. Test "Pass"
    code = "def add(a, b):\n    return a + b"
    test_pass = """
import unittest
# from add import add
class TestSolution(unittest.TestCase):
    def test_function(self):
        candidate = add
        self.assertTrue(candidate(1, 2) == 3)
        self.assertTrue(candidate(-1, 1) == 0)
"""
    reward_pass = execute_and_reward(code, test_pass, "add")
    print(f"Test Pass Reward: {reward_pass}") # Should be 1.0

    # 2. Test "Fail"
    test_fail = """
import unittest
# from add import add
class TestSolution(unittest.TestCase):
    def test_function(self):
        candidate = add
        self.assertTrue(candidate(1, 2) == 4) # This is wrong
"""
    reward_fail = execute_and_reward(code, test_fail, "add")
    print(f"Test Fail Reward: {reward_fail}") # Should be 0.2

    # 3. Test "Syntax Error"
    test_syntax = "import unittest\nclass"
    reward_syntax = execute_and_reward(code, test_syntax, "add")
    print(f"Test Syntax Error Reward: {reward_syntax}") # Should be -1.0
    
    # 4. Test "Runtime Error"
    test_runtime = """
import unittest
# from add import add
class TestSolution(unittest.TestCase):
    def test_function(self):
        candidate = add
        self.assertTrue(candidate(1, 'a') == 3) # TypeError
"""
    reward_runtime = execute_and_reward(code, test_runtime, "add")
    print(f"Test Runtime Error Reward: {reward_runtime}") # Should be 0.1
