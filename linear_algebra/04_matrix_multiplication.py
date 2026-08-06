"""Day 4: Matrix multiplication """
import numpy as np
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[2, 4], [6, 8], [10, 12]])

def matmul(A, B):
    m, n = A.shape
    n2, p = B.shape
    result = np.zeros((m, p), dtype=int)
    for i in range(m):
        for j in range(p):
            result[i, j] = np.dot(A[i], B[:, j])
    return result
result = matmul(A, B)
print(f"Manual matrix multiplication :{result}")
print(f"A @ B: {A@B}")

assert np.array_equal(matmul(A, B), A@B), "Manual multiplication doesn't match A * B"

X = np.array([[1, 2, 7], [3, 6, 9]])
Y = np.array([[4, 6, 8, 9],[1, 4, 9, 0]])

try:
    print(matmul(X, Y))
except ValueError as e:
    print(e)

print(f"A.shape: {A.shape}, B.shape: {B.shape}, result.shape: {result.shape} | ({A.shape[0]},{A.shape[1]}) @ ({B.shape[0]},{B.shape[1]}) = ({result.shape[0]},{result.shape[1]})")