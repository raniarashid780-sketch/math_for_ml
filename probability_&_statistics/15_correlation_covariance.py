"""Day 15: Correlation & Covariance"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5  # strongly correlated with x
z = np.random.randn(100)                  # independent of x

def manual_covariance(a, b):
    """Population covariance, matching np.cov's [0,1] entry when bias=True."""
    a, b= np.asarray(a), np.asarray(b)
    mean_a, mean_b =a.mean(), b.mean()
    return np.mean((a - mean_a) * (b - mean_b))

def manual_correlation(a, b):
    """Pearson correlation coefficient."""
    a, b = np.asarray(a), np.asarray(b)
    cov = manual_covariance(a, b)
    std_a, std_b = a.std(), b.std()
    return cov/(std_a * std_b)

manual_cov_xy = manual_covariance(x, y)
numpy_cov_xy = np.cov(x, y, bias=True)[0, 1]
print(f"Manual cov(x, y) = {manual_cov_xy:.4f} | NumPy cov (x, y) = {numpy_cov_xy:.4f}")
assert np.isclose(manual_cov_xy, numpy_cov_xy),"Covariance mismatch"

manual_corr_xy = manual_correlation(x, y)
numpy_corr_xy = np.corrcoef(x, y)[0, 1]
print(f"Manual corr(x, y) = {manual_corr_xy:.4f} | NumPy corr (x, y) = {numpy_corr_xy:.4f}")
assert np.isclose(manual_corr_xy, numpy_corr_xy),"Correlation mismatch"

manual_corr_xz = manual_correlation(x, z)
numpy_corr_xz = np.corrcoef(x, z)[0, 1]
print(f"Manual corr(x, z) = {manual_corr_xz:.4f} | NumPy corr (x, z) = {numpy_corr_xz:.4f}")
assert np.isclose(manual_corr_xz, numpy_corr_xz),"Correlation mismatch"


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(x, y, alpha=0.6)
axes[0].set_title(f"Correlated (r = {manual_corr_xy:.2f})")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

axes[1].scatter(x, z, alpha=0.6)
axes[1].set_title(f"Independent (r = {manual_corr_xz:.2f})")
axes[1].set_xlabel("x")
axes[1].set_ylabel("z")

plt.tight_layout()
plt.show()
SCRIPT_DIR = Path(__file__).resolve().parent
output_path = SCRIPT_DIR.parent / "assets" / "correlation_covariance.png"
plt.savefig(output_path)

# Real correlation-without-causation example:
# Countries' per-capita chocolate consumption correlates positively with
# number of Nobel laureates per capita. The actual driver is national
# wealth/GDP — richer countries both afford more chocolate imports AND
# fund more research infrastructure. Chocolate doesn't cause Nobel prizes;
# a shared confound (economic development) drives both.