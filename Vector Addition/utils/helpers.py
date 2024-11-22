import numpy as np
import json
import os

# Create a folder for test cases
TEST_CASES_DIR = "test_cases"
os.makedirs(TEST_CASES_DIR, exist_ok=True)

def generate_vector(size):
    """Generates a random vector of given size."""
    return np.random.rand(size).tolist()

def save_test_case(file_name, inputs, expected_output):
    """Saves test cases to a JSON file."""
    file_path = os.path.join(TEST_CASES_DIR, file_name)
    with open(file_path, 'w') as f:
        json.dump({"inputs": inputs, "expected_output": expected_output}, f, indent=4)

def load_test_case(file_name):
    """Loads test cases from a JSON file."""
    file_path = os.path.join(TEST_CASES_DIR, file_name)
    with open(file_path, 'r') as f:
        return json.load(f)

def validate_result(computed, expected, tolerance=1e-6):
    """
    Compares computed and expected results.
    Returns True if within the specified tolerance.
    """
    return np.allclose(computed, expected, atol=tolerance)
