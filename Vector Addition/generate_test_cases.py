from utils.helpers import generate_vector, save_test_case

# Parameters
VECTOR_SIZE = 1000
file_name = "vector_addition_test.json"

# Generate test case
vec_a = generate_vector(VECTOR_SIZE)
vec_b = generate_vector(VECTOR_SIZE)
expected_output = [a + b for a, b in zip(vec_a, vec_b)]

# Save test case
save_test_case(file_name, {"vector_a": vec_a, "vector_b": vec_b}, expected_output)

print(f"Test case saved to {file_name}")
