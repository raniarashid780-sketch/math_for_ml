"""Day 8: Eigenvalues and Eigenvectors"""
import numpy as np

A = np.array([[3, 1], [1, 3]])

# Task 1 — verify A is symmetric
assert np.array_equal(A, A.T), "Matrix is not symmetric"

# Task 2 — compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
# eigenvectors: each COLUMN is one eigenvector, not each row

# Task 3 — print clearly
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{eigenvectors}")

# Task 4 — verify definition A @ v = λ * v for each eigenvector
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]      # column i = eigenvector i
    lam = eigenvalues[i]
    assert np.allclose(A @ v, lam * v), f"Eigen definition failed for eigenvector {i}"
    print(f"Eigenvector {i}: A@v={A@v}, λ*v={lam*v} ✓")

# Task 5 — one comment: what does a larger eigenvalue mean geometrically?
# A larger eigenvalue means that the corresponding eigenvector is stretched more when the linear transformation represented by matrix A is applied.
#  Geometrically, it indicates that the direction of the eigenvector is preserved, but its magnitude is scaled by the eigenvalue.