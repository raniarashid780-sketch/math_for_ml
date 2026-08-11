"""Day 13: Distributions"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=1000)    # mean=0, std=1
binomial_data = np.random.binomial(n=20, p=0.3, size=1000)   # 20 trials, 30% success
uniform_data = np.random.uniform(low=0, high=1, size=1000)    # between 0 and 1

print(f"Normal data mean:{normal_data.mean()}")
print(f"Binomial data mean:{binomial_data.mean()}")
print(f"Uniform data mean:{uniform_data.mean()}")

within_1std = np.sum(np.abs(normal_data - normal_data.mean()) < normal_data.std()) / len(normal_data)
assert np.isclose(within_1std, 0.68, atol=0.05), "68% rule failed"

fig, axes = plt.subplots(1, 3, figsize=(12, 3))

axes[0].hist(normal_data, bins=30, color='steelblue', edgecolor='black')
axes[0].set_title('Normal Distribution')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

axes[1].hist(binomial_data, bins=20, color='salmon', edgecolor='black')
axes[1].set_title('Binomial Distribution (n=20, p=0.3)')
axes[1].set_xlabel('Number of Successes')
axes[1].set_ylabel('Frequency')

axes[2].hist(uniform_data, bins=30, color='mediumseagreen', edgecolor='black')
axes[2].set_title('Uniform Distribution (0 to 1)')
axes[2].set_xlabel('Value')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('D:/math_for_ml/assets/distributions.png', dpi=100, bbox_inches='tight')
plt.show()
