"""Day 21: gradient descent."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def f(x):
    return (x-3)**2

def grad_f(x):
    return 2*(x-3)

def gradient_descent(start, learning_rate, n_steps):
    x = start
    x_history = [x]
    f_history = [f(x)]
    for _ in range(n_steps):
        x -= learning_rate * grad_f(x)
        x_history.append(x)
        f_history.append(f(x))
    return x, x_history, f_history

print(f"Gradient descent: {gradient_descent(10.0, 0.01, 50)}")
print(f"Gradient descent: {gradient_descent(10.0, 0.1, 50)}")
print(f"Gradient descent: {gradient_descent(10.0, 0.9, 50)}")

assert np.isclose(gradient_descent(10.0, 0.01, 50)[0], 5.54, atol=1e-2)
assert np.isclose(gradient_descent(10.0, 0.1, 50)[0], 3.0, atol=1e-2)
assert np.isclose(gradient_descent(10.0, 0.9, 50)[0], 3.0, atol=1e-2)

fig, ax = plt.subplots(figsize=(6, 5))
steps = np.arange(51)
for lr in [0.01, 0.1, 0.9]:
    _, _, f_history = gradient_descent(10.0, lr, 50)
    ax.plot(steps, f_history, label=f"lr={lr}")

ax.set_xlabel("Step")
ax.set_ylabel("f(x)")
ax.set_title("Gradient Descent")
ax.legend()
plt.tight_layout()
SCRIPT_DIR = Path(__file__).resolve().parent
output_path = SCRIPT_DIR.parent / "assets" / "gradient_descent.png"
plt.savefig(output_path)
plt.show()
