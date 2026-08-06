"""Day 3: Matrix Basics"""
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

def add_matrices(a, b):
    return np.array([
        [a[0, 0] + b[0, 0], a[0, 1] + b[0, 1]],
        [a[1, 0] + b[1, 0], a[1, 1] + b[1, 1]],
    ])

print(f"Manual addition: {add_matrices(A, B)}")
print(f"A + B: {A + B}")

def scale_matrix(a, k):
    return np.array([
        [a[0, 0] * k, a[0, 1] * k],
        [a[1, 0] * k, a[1, 1] * k],
    ])
print(f"Manual scale of matrix: {scale_matrix(A, 5)}")
print(f"A * k: {A * 5}")

C = np.array([[7, 8, 9], [4, 5, 6]])

def transpose_matrix(a):
    return np.array([
    [a[0, 0], a[1, 0]],
    [a[0, 1], a[1, 1]],
    [a[0, 2], a[1, 2]]
    ])
print(f"shape befoore transpose: {C.shape}")
print(f"Manual transpose of matrix: {transpose_matrix(C)}")
D = C.T
print(f"C.T: {D}")
print(f" Shape after transpose: {D.shape}")
print(f"Shape flipped from {C.shape} to {D.shape} because transpose swaps rows and columns")

assert np.array_equal(add_matrices(A, B), A + B), "Manual addition doesn't match A + B"
assert np.array_equal(scale_matrix(A, 5), A * 5), "Manual scale doesn't match A * 5"
assert np.array_equal(transpose_matrix(C), C.T), "Manual transpose doesn't match C.T"