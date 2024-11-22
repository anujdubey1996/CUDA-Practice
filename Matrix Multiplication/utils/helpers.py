import numpy as np
import json
import os

# Directory to store test cases
TEST_CASES_DIR = "test_cases"
os.makedirs(TEST_CASES_DIR, exist_ok=True)

def generate_vector(size):
    """
    Generates a random vector of the given size.
    """
    return np.random.rand(size).tolist()

def generate_matrix(rows, cols):
    """
    Generates a random matrix with the given dimensions (rows x cols).
    """
    return np.random.rand(rows, cols).tolist()

def save_test_case(file_name, inputs, expected_output):
    """
    Saves test cases to a JSON file.
    
    Args:
        file_name (str): Name of the JSON file to save.
        inputs (dict): Dictionary containing the inputs.
        expected_output (list): The expected output of the test case.
    """
    file_path = os.path.join(TEST_CASES_DIR, file_name)
    with open(file_path, 'w') as f:
        json.dump({"inputs": inputs, "expected_output": expected_output}, f, indent=4)

def load_test_case(file_name):
    """
    Loads test cases from a JSON file.
    
    Args:
        file_name (str): Name of the JSON file to load.
        
    Returns:
        dict: A dictionary with inputs and expected_output.
    """
    file_path = os.path.join(TEST_CASES_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test case file not found: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def validate_result(computed, expected, tolerance=1e-6):
    """
    Validates the computed result against the expected output.
    
    Args:
        computed (array-like): The computed result.
        expected (array-like): The expected result.
        tolerance (float): Allowed tolerance for numerical differences.
        
    Returns:
        bool: True if the results match within the given tolerance, False otherwise.
    """
    return np.allclose(computed, expected, atol=tolerance)
