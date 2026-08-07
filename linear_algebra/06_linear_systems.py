"""Day 6: System of linear equations"""
import numpy as np
A = np.array([[2, 1], [1, 3]])
b = np.array([[5], [7]])
x1 = np.linalg.inv(A) @ b
x2 = np.linalg.solve(A, b)
assert np.allclose(x1, x2),"Inverse method failed to match"
assert np.allclose(A @ x2, b),"Solution doesn't satisfy original equation A @ x = b"
print(f"Solution via inverse: x={x1.flatten()}")
print(f"Solution via solve:   x={x2.flatten()}")
print(f"Verification A @ x2 == b: {np.allclose(A @ x2, b)}")
A_singular = np.array([[2, 4], [1, 2]])
try:
    np.linalg.solve(A_singular, b)
except np.linalg.LinAlgError as e:
    print(f"\nCaught expected LinAlgError: {e}")
# np.linalg.solve is more efficient and numerically stable than computing the inverse, especially for large systems.
# Both give same answer for small systems, but np.linalg.solve is preferred in practice.