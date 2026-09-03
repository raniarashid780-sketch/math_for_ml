
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
# Task 1
# Pick a simple 2-variable function, e.g. f(x, y) = x**2 + 3*x*y + y**2.

# df/dx = 2x + 3y
# df/dy = 3x + 2y
def f(x, y):
    return x**2 + 3*x*y + y**2

# Task 2
# Implement a numerical partial derivative function: numerical_partial(f, x, y, wrt, h=1e-5) that estimates the partial using the limit definition ((f(x+h, y) - f(x, y)) / h for wrt='x').

def numerical_partial(f, x, y, wrt, h=1e-5):
    if wrt == 'x':
        return (f(x + h, y) - f(x, y)) / h
    elif wrt == 'y':
        return (f(x, y + h) - f(x, y)) / h


# Task 3
# Implement an analytical partial derivative function: analytical_partial(f, x, y, wrt)
# Evaluate your numerical partial and your hand-derived analytical formula at a specific point (e.g. x=2, y=3), assert they're close with np.isclose.

x0, y0 = 2, 3

# analytic (plug numbers into your hand-derived formulas)
df_dx_analytic = 2*x0 + 3*y0
df_dy_analytic = 3*x0 + 2*y0

# numerical (call the checker function)
df_dx_numeric = numerical_partial(f, x0, y0, 'x')
df_dy_numeric = numerical_partial(f, x0, y0, 'y')

print(f"df/dx: analytic={df_dx_analytic}, numeric={df_dx_numeric:.5f}")
print(f"df/dy: analytic={df_dy_analytic}, numeric={df_dy_numeric:.5f}")

import numpy as np
assert np.isclose(df_dx_analytic, df_dx_numeric)
assert np.isclose(df_dy_analytic, df_dy_numeric)

# Task 4
# Plot the surface f(x,y) in 3D (matplotlib's Axes3D or ax.plot_surface) — first 3D plot in this track, new syntax, look it up if needed rather than guessing.

x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)   # your function works here too, since it's just +, *, ** — numpy handles arrays fine

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
SCRIPT_DIR = Path(__file__).resolve().parent
output_path = SCRIPT_DIR.parent / "assets" / "partial_derivatives_surface.png"
plt.savefig(output_path)