"""Day 10: PCA from scratch — linear algebra capstone"""
import numpy as np
import matplotlib.pyplot as plt

# Step 0 — create some fake 2D data (50 points, 2 features, correlated)
np.random.seed(42)
X = np.random.randn(50, 2)
X[:, 1] = X[:, 0] * 2 + np.random.randn(50) * 0.5  # feature 2 correlates with feature 1

# Step 1 — center the data (YOUR CODE)
X_centered = X - X.mean(axis=0)
# number of samples and features
n, d = X.shape

# Step 2 — covariance matrix (YOUR CODE)
covariance_matrix = (X_centered.T @ X_centered) / (n - 1)

# Step 3 — eigenvalues and eigenvectors (YOUR CODE)
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

# Step 4 — sort by eigenvalue descending (YOUR CODE)
# hint: np.argsort(eigenvalues)[::-1] gives sorted indices
sort_idx = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sort_idx]
eigenvectors_sorted = eigenvectors[:, sort_idx]

# Step 5 — project onto top 1 principal component (YOUR CODE)
# result should be shape (50, 1)
X_pca = X_centered @ eigenvectors_sorted[:, :1]
explained_variance = eigenvalues_sorted / eigenvalues_sorted.sum()

#Verification
print(f"Original data shape: {X.shape}")
print(f"Centered data mean (should be ~0): {X_centered.mean(axis=0)}")
print(f"Eigenvalues sorted: {eigenvalues_sorted}")
print(f"Explained variance ratio: {explained_variance}")
print(f"PCA projection shape: {X_pca.shape}")

assert X_pca.shape == (50, 1), "Projection shape wrong"
assert np.allclose(X_centered.mean(axis=0), np.zeros(d), atol=1e-10), "Data not centered"

fig, ax = plt.subplots()
ax.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.5, label='Data')
pc1 = eigenvectors_sorted[:, 0]
ax.quiver(0, 0, pc1[0]*2, pc1[1]*2, color='red', angles='xy',
          scale_units='xy', scale=1, label='PC1')
ax.set_aspect('equal')
ax.legend()
plt.title('PCA — data cloud + first principal component')
plt.savefig("D:/math_for_ml/assets/pca_visualization.png", dpi=100, bbox_inches='tight')
plt.show()
