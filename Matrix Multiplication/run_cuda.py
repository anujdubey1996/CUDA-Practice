import numpy as np
from ctypes import cdll, POINTER, c_float, c_int
from utils.helpers import load_test_case, validate_result

# Load the compiled CUDA shared library
cuda_lib = cdll.LoadLibrary("./matrix_multiplication.so")

# Define the function signature for matrixMultiply
matrix_multiply = cuda_lib.matrixMultiply
matrix_multiply.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int]

def test_matrix_multiplication():
    # Load test case
    test_case = load_test_case("matrix_multiplication_test.json")
    mat_a = np.array(test_case["inputs"]["matrix_a"], dtype=np.float32)
    mat_b = np.array(test_case["inputs"]["matrix_b"], dtype=np.float32)
    expected_output = np.array(test_case["expected_output"], dtype=np.float32)

    # Get dimensions
    M, N = mat_a.shape
    _, K = mat_b.shape

    # Allocate memory for output
    mat_c = np.zeros((M, K), dtype=np.float32)

    # Call the C function
    matrix_multiply(mat_a.ctypes.data_as(POINTER(c_float)),
                    mat_b.ctypes.data_as(POINTER(c_float)),
                    mat_c.ctypes.data_as(POINTER(c_float)),
                    c_int(M), c_int(N), c_int(K))

    # Validate the result
    if validate_result(mat_c, expected_output):
        print("CUDA Test Passed!")
    else:
        print("CUDA Test Failed!")

if __name__ == "__main__":
    test_matrix_multiplication()
