"""Day 7: Determinants"""
import numpy as np
A = np.array([[1, 2], [2, 8]])
B = np.array([[2, 4], [1, 2]])

def determinant2x2(a):
    return (a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0])

print(f"Manual determinant: {determinant2x2(A)}")
print(f"np.linalg.det(A):{np.linalg.det(A)}")

assert np.isclose(determinant2x2(A), np.linalg.det(A)), "Manual determinant does not match "
assert np.isclose(determinant2x2(B), 0), "Manual determinant does not match for singular matrix B"
# why does det = 0 mean no inverse exists?
# A singular matrix has no inverse because its rows are linearly dependent (determinant = 0),
# compressing space into fewer dimensions so the original input cannot be recovered.