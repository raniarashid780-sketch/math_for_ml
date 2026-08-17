"""Day 17: Central Limit Theorem"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
np.random.randint(1, 7)
def sample_means(n_rolls, n_trials):
    means = []
    for _ in range(n_trials):
        rolls = np.random.randint(1, 7, size=n_rolls)
        means.append(rolls.mean())
    return means

n_rolls = 30
n_trials = 1000
means = sample_means(n_rolls, n_trials)

print(f"Mean of sample means (N={n_rolls}): {np.mean(means):.4f}")
print(f"Standard deviation of sample means (N={n_rolls}): {np.std(means):.4f}")

# Theoretical standard deviation for a single die roll
std_N1 = np.std(np.random.randint(1, 7, size=100000))
# Theoretical standard deviation for the sample mean
theoretical_std = std_N1 / np.sqrt(n_rolls)
print(f"Theoretical standard deviation (N={n_rolls}): {theoretical_std:.4f}")
np.isclose(np.std(means), theoretical_std, rtol=0.1)  # Allowing 10% relative tolerance

plt.hist(means, bins=30, edgecolor='black')
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.title('Distribution of Sample Means')
plt.savefig("D:/math_for_ml/assets/central_limit_theorem.png")
plt.show()

