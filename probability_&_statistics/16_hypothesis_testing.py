"""Day 16: Hypothesis Testing"""
import numpy as np
from scipy.stats import ttest_ind

class_a = [78, 82, 75, 90, 85, 79, 88, 76, 91, 83]
class_b = [74, 70, 68, 80, 72, 75, 69, 77, 71, 73]

def manual_t_test(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    t = (mean_a - mean_b) / np.sqrt(pooled_var * (1/n_a + 1/n_b))
    return t

manual_t = manual_t_test(class_a, class_b)
t_stat, p_value = ttest_ind(class_a, class_b)
print(f"Manual t = {manual_t:.4f} | scipy t = {t_stat:.4f}")
assert np.isclose(manual_t, t_stat), "t-statistic mismatch"

print(f"p-value = {p_value:.4f}")
# H0: true mean scores of Class A and Class B are equal
if p_value < 0.05:
    print(f"p = {p_value:.4f} < 0.05 -> reject H0: difference is statistically significant")
else:
    print(f"p = {p_value:.4f} >= 0.05 -> fail to reject H0: not enough evidence of a difference")