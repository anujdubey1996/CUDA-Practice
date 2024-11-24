from utils.helpers import generate_matrix, save_test_case
import numpy as np

# Parameters
M, N, K = 4, 4, 4  # Matrix dimensions: A[MxN], B[NxK], C[MxK]
file_name = "matrix_multiplication_test.json"

# Generate test case
matrix_a = generate_matrix(M, N)
matrix_b = generate_matrix(N, K)
expected_output = np.dot(np.array(matrix_a), np.array(matrix_b)).tolist()

# Save test case
save_test_case(file_name, {"matrix_a": matrix_a, "matrix_b": matrix_b}, expected_output)

print(f"Test case saved to {file_name}")
