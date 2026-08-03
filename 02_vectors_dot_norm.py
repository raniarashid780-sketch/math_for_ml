"""Day 2: Dot Product, Norms, Cosine Similarity """
import numpy as np

a = np.array([1, 0])
b = np.array([2, 0])

x = np.array([1, 0])
y = np.array([0, 1])

c = np.array([1, 0])
d = np.array([-1, 0])

def dotproduct(a, b):
    return a[0] * b[0] + a[1] * b[1]

print(f"dotproduct(a,b) manual = {dotproduct(a, b)}")
print(f"np.dot(a,b) = {np.dot(a, b)}")

def norm(a):
    return np.sqrt(a[0]**2 + a[1]**2)

print(f"norm(a) manual = {norm(a)}")
print(f"np.linalg.norm(a) = {np.linalg.norm(a)}")

def cosine_similarity(a, b):
    return dotproduct(a, b) / (norm(a) * norm(b))

print(f"cosine_similarity(a, b) = {cosine_similarity(a, b)}")
print(f"cosine_similarity(x, y) = {cosine_similarity(x, y)}")
print(f"cosine_similarity(c, d) = {cosine_similarity(c, d)}")
angle = np.arccos(cosine_similarity(a, b))
check = norm(a) * norm(b) * np.cos(angle)
print(f"angle between a,b (radians) = {angle}")
print(f"dot product = {dotproduct(a, b)}, reconstructed from angle = {check}")
assert np.isclose(dotproduct(a, b), check), "Dot product doesn't match norm*norm*cos(angle)"
print("Task 5 verified: dot product formula holds.")