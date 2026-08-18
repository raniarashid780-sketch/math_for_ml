"""Day 17: Central Limit Theorem"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)

def sample_means(n_rolls, n_trials):
    means = []
    for _ in range(n_trials):
        rolls = np.random.randint(1, 7, size=n_rolls)
        means.append(rolls.mean())
    return means

n_trials = 1000

means_n1 = sample_means(1, n_trials)
means_n30 = sample_means(30, n_trials)

std_n1 = np.std(means_n1)
std_n30 = np.std(means_n30)
theoretical_std_n30 = std_n1 / np.sqrt(30)

print(f"Mean of sample means (N=1): {np.mean(means_n1):.4f}")
print(f"Std of sample means (N=1): {std_n1:.4f}")
print(f"Mean of sample means (N=30): {np.mean(means_n30):.4f}")
print(f"Std of sample means (N=30): {std_n30:.4f}")
print(f"Theoretical std (N=30) = std_N1 / sqrt(30): {theoretical_std_n30:.4f}")

matches = np.isclose(std_n30, theoretical_std_n30, rtol=0.1)
print(f"Simulated std matches theoretical prediction (10% tolerance): {matches}")
assert matches, "Simulated std deviates too far from CLT prediction"

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(means_n1, bins=6, edgecolor='black')
axes[0].set_title("N=1 (single roll)")
axes[0].set_xlabel("Sample Mean")
axes[0].set_ylabel("Frequency")

axes[1].hist(means_n30, bins=30, edgecolor='black')
axes[1].set_title("N=30 (averaged)")
axes[1].set_xlabel("Sample Mean")
axes[1].set_ylabel("Frequency")

plt.tight_layout()

SCRIPT_DIR = Path(__file__).resolve().parent
output_path = SCRIPT_DIR.parent / "assets" / "central_limit_theorem.png"
plt.savefig(output_path)
print(f"Plot saved to {output_path}")

plt.show()