"""Day 5: Matrix inverse"""
import numpy as np

A = np.array([[1, 2], [3, 4]])
print("Matrix A:\n", A)

# An identity matrix has 1s on the main diagonal and 0s elsewhere;
# multiplying any matrix by I leaves the original matrix unchanged (A @ I = A).
I = np.eye(2, dtype=int)
print("\nIdentity Matrix (I):\n", I)

A_inv = np.linalg.inv(A)
print("\nInverse Matrix (A_inv):\n", A_inv)

# np.allclose accounts for tiny floating-point rounding errors (e.g., 1.0000000000000002 vs 1.0)
is_identity = np.allclose(A @ A_inv, I)
print(f"\nVerification (A @ A_inv == I): {is_identity}")
assert is_identity, "Matrix multiplication with inverse failed to produce identity matrix."

# Row 1 ([2, 4]) is exactly 2 * Row 2 ([1, 2])
A_singular = np.array([[2, 4], [1, 2]])
print("\nSingular Matrix:\n", A_singular)

try:
    np.linalg.inv(A_singular)
except np.linalg.LinAlgError as e:
    print(f"\nCaught expected LinAlgError: {e}")

# 6. Print one-line explanation
print("\nExplanation: A singular matrix has no inverse because its rows are linearly dependent (determinant = 0), compressing space into fewer dimensions so the original input cannot be recovered.")