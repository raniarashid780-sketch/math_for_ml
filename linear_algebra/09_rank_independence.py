"""Day 9: Rank, Linear Independence, Span"""
import numpy as np

# Task 1 — two independent vectors and two dependent vectors
v1 = np.array([1, 2])
v2 = np.array([3, 4])        # independent from v1
v3 = np.array([2, 4])        # dependent — v3 = 2 * v1

# Task 2 — check rank of matrices built from these
A_independent = np.array([v1, v2])   # stack as rows
A_dependent = np.array([v1, v3])

print(f"Rank of independent matrix: {np.linalg.matrix_rank(A_independent)}")
print(f"Rank of dependent matrix: {np.linalg.matrix_rank(A_dependent)}")

# Task 3 — assert ranks are what you expect
assert np.linalg.matrix_rank(A_independent) == 2, "Expected rank 2 for independent vectors"
assert np.linalg.matrix_rank(A_dependent) == 1, "Expected rank 1 for dependent vectors"

# Task 4 — determinant connection: det=0 ↔ rank < full
print(f"det(A_independent): {np.linalg.det(A_independent):.4f}")
print(f"det(A_dependent): {np.linalg.det(A_dependent):.4f}")
assert np.isclose(np.linalg.det(A_dependent), 0), "Dependent matrix should have det=0"

# Task 5 — 3x3 matrix with rank 2 (one redundant row)
B = np.array([[1, 2, 3],
              [4, 5, 6],
              [5, 7, 9]])    # row3 = row1 + row2
print(f"Rank of B (one redundant row): {np.linalg.matrix_rank(B)}")
assert np.linalg.matrix_rank(B) == 2, "Expected rank 2"

# Task 6 — one comment: what does rank tell you about the data in ML terms?
# Rank tells you the number of linearly independent features in your dataset.
# In machine learning, a lower rank than the number of features indicates redundancy, which can lead to multicollinearity issues and may affect model performance.