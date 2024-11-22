import numpy as np
from ctypes import cdll, POINTER, c_float, c_int
from utils.helpers import load_test_case, validate_result

# Load the compiled CUDA shared library
cuda_lib = cdll.LoadLibrary("./vector_addition.so")

# Define the function signature
vector_add = cuda_lib.vectorAdd
vector_add.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int]

def test_vector_addition():
    # Load test case
    test_case = load_test_case("vector_addition_test.json")
    vec_a = np.array(test_case["inputs"]["vector_a"], dtype=np.float32)
    vec_b = np.array(test_case["inputs"]["vector_b"], dtype=np.float32)
    expected_output = np.array(test_case["expected_output"], dtype=np.float32)

    # Allocate output memory
    vec_c = np.zeros_like(vec_a, dtype=np.float32)

    # Call the C function
    vector_add(vec_a.ctypes.data_as(POINTER(c_float)),
               vec_b.ctypes.data_as(POINTER(c_float)),
               vec_c.ctypes.data_as(POINTER(c_float)),
               c_int(len(vec_a)))

    # Validate the result
    if validate_result(vec_c, expected_output):
        print("CUDA Test Passed!")
    else:
        print("CUDA Test Failed!")
        #print(vec_c)
        #print(expected_output)

if __name__ == "__main__":
    test_vector_addition()
