"""Day 1: Vectors — addition, scalar multiplication, geometric intuition."""
import numpy as np
import matplotlib.pyplot as plt
v1 = np.array([2, 4])
v2 = np.array([6, 3])
v_sum = v1 + v2
# A Python list is itself a contiguous array — but of pointers, not values.
# A NumPy array is a contiguous block of the raw values themselves.
# So NumPy arrays are more efficient for numerical computations.

def add_vectors(v1, v2):
    """Add two vectors."""
    return v1 + v2

def scale_vector(v, k):
    return k * v

print(f"v1 + v2 = {add_vectors(v1, v2)}")
print(f"v1 * 3 = {scale_vector(v1, 3)}")
print(f"v1 * -5 = {scale_vector(v1, -5)}")
print(f"v1 * 0 = {scale_vector(v1, 0)}")

fig, ax = plt.subplots()

origin = np.array([0, 0])
head_of_v1 = v1
start_of_v2 = head_of_v1
end_of_v2 = start_of_v2 + v2

ax.quiver(origin[0], origin[1], v1[0], v1[1], color='blue', angles='xy', scale_units='xy', scale=1, label='v1')
ax.quiver(start_of_v2[0], start_of_v2[1], v2[0], v2[1], color='red', angles='xy', scale_units='xy', scale=1, label='v2')
ax.quiver(origin[0], origin[1], v_sum[0], v_sum[1], color='green', angles='xy', scale_units='xy', scale=1, label='v1+v2')

ax.set_xlim(-1, 8)
ax.set_ylim(-1, 8)
ax.set_aspect('equal')
assert np.array_equal(end_of_v2, v_sum), "Tip-to-tail path doesn't match v1+v2 — geometry is broken"
print(f"Tip-to-tail endpoint {end_of_v2} matches v1+v2 = {v_sum}")
ax.legend()
plt.show()