"""Day 20: gradient and chain rule."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# df/dx = 2x + 3y
# df/dy = 3x + 2y
def f(x, y):
    return x**2 + 3*x*y + y**2

def gradient(x, y):
    return np.array([2*x + 3*y,    # ∂f/∂x
                     3*x + 2*y])   # ∂f/∂y

def numerical_partial(f, x, y, wrt, h=1e-5):
    if wrt == 'x':
        return (f(x + h, y) - f(x, y)) / h
    elif wrt == 'y':
        return (f(x, y + h) - f(x, y)) / h

x0, y0 = 2, 3

grad = gradient(x0, y0)                          # [13, 12]
dx   = numerical_partial(f, x0, y0, 'x')         # ≈ 13
dy   = numerical_partial(f, x0, y0, 'y')         # ≈ 12

assert np.isclose(grad[0], dx)
assert np.isclose(grad[1], dy)

print(f"Analytic gradient:  {grad}")
print(f"Numeric dx: {dx:.5f}, dy: {dy:.5f}")

def g(x):
    return x**2          # inside

def composed(x):
    return np.sin(g(x)) # outside wrapping inside

def chain_rule_derivative(x):
    return np.cos(x**2) * 2*x   # step 2 × step 3

def numerical_derivative(func, x, h=1e-5):
    return (func(x + h) - func(x)) / h   # checker

x0 = 1

analytic = chain_rule_derivative(x0)      # 1.0806
numeric  = numerical_derivative(composed, x0)    # ≈ 1.0806

assert np.isclose(analytic, numeric, atol=1e-4)

print(f"Chain rule analytic: {analytic:.5f}")
print(f"Chain rule numeric:  {numeric:.5f}")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- left plot: gradient arrows ---
x_vals = np.linspace(-3, 3, 10)
y_vals = np.linspace(-3, 3, 10)
X, Y   = np.meshgrid(x_vals, y_vals)

U = 2*X + 3*Y    # ∂f/∂x at every grid point
V = 3*X + 2*Y    # ∂f/∂y at every grid point

ax1.quiver(X, Y, U, V)
ax1.set_title("Gradient of f(x,y)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# --- right plot: sin(x²) and derivative ---
x_line = np.linspace(-3, 3, 300)

ax2.plot(x_line, np.sin(x_line**2),          label="sin(x²)")
ax2.plot(x_line, np.cos(x_line**2) * 2*x_line, label="derivative")
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_title("sin(x²) and its derivative")
ax2.set_xlabel("x")
ax2.legend()

plt.tight_layout()
SCRIPT_DIR = Path(__file__).resolve().parent
output_path = SCRIPT_DIR.parent / "assets" / "gradient_chain.png"
plt.savefig(output_path)
plt.show()