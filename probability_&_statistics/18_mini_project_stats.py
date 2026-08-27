"""Day 18: Mini Project - Statistics on Freelance Job Data"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_ind

np.random.seed(42)

# 1. Load and clean
project_dir = Path(__file__).resolve().parent.parent
data = pd.read_csv(project_dir / "sample.csv")

data["budget"] = pd.to_numeric(data["budget"], errors="coerce")
data = data.dropna(subset=["budget", "category", "skills"])
data["skills_count"] = data["skills"].str.split(",").str.len()

# 2. Descriptive statistics
print(data["budget"].describe())
print(f"Mean: {data['budget'].mean():.2f}")
print(f"Median: {data['budget'].median():.2f}")
print(f"Standard deviation: {data['budget'].std():.2f}")

# 3. Correlation
correlation = data[["budget", "skills_count"]].corr()
print(correlation)

# 4. Hypothesis test
# H0: mean budget for Web Developer jobs equals mean budget for Data Analyst jobs
web_developer = data.loc[data["category"] == "Web Developer", "budget"]
data_analyst = data.loc[data["category"] == "Data Analyst", "budget"]

t_stat, p_value = ttest_ind(web_developer, data_analyst, equal_var=False)

print(f"Web Developer mean: {web_developer.mean():.2f}")
print(f"Data Analyst mean: {data_analyst.mean():.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print(f"p = {p_value:.4f} < 0.05 -> reject H0: budget difference is statistically significant")
else:
    print(f"p = {p_value:.4f} >= 0.05 -> fail to reject H0: not enough evidence of a real difference")

# 5. Central Limit Theorem simulation
def sample_means(values, sample_size, trials=1000):
    means = []
    for _ in range(trials):
        sample = np.random.choice(values, size=sample_size, replace=True)
        means.append(sample.mean())
    return np.array(means)

budget_values = data["budget"].to_numpy()
means_5 = sample_means(budget_values, 5)
means_30 = sample_means(budget_values, 30)

std_5 = means_5.std()
std_30 = means_30.std()
theoretical_std_30 = std_5 / np.sqrt(30 / 5)

print(f"Std of means, n=5: {std_5:.2f}")
print(f"Std of means, n=30: {std_30:.2f}")
print(f"Theoretical std, n=30 (from n=5 std): {theoretical_std_30:.2f}")
matches = np.isclose(std_30, theoretical_std_30, rtol=0.2)
print(f"Simulated shrink matches CLT prediction (20% tolerance): {matches}")

# 6. Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].hist(data["budget"], bins=12, color='tab:red', edgecolor="black")
axes[0, 0].set_title("Budget Distribution")
axes[0, 0].set_xlabel("Budget")

axes[0, 1].scatter(data["skills_count"], data["budget"], color='tab:orange', alpha=0.6)
axes[0, 1].set_title("Budget vs Number of Skills")
axes[0, 1].set_xlabel("Number of Skills")
axes[0, 1].set_ylabel("Budget")

axes[1, 0].hist(means_5, bins=20, alpha=0.7, label="n=5", color='tab:green')
axes[1, 0].hist(means_30, bins=20, alpha=0.7, label="n=30", color='tab:pink')
axes[1, 0].set_title("CLT: Sample Means")
axes[1, 0].legend()

category_means = data.groupby("category")["budget"].mean()
category_means.plot(kind="bar", ax=axes[1, 1], color='tab:purple', edgecolor="black")
axes[1, 1].set_title("Average Budget by Category")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
SCRIPT_DIR = Path(__file__).resolve().parent
output_path = SCRIPT_DIR.parent / "assets" / "mini_project_stats_overview.png"
plt.savefig(output_path)
plt.show()

# Insight:
# Web Developer jobs have a slightly higher mean budget ($766 vs $733), but
# the t-test result (p = 0.9391) shows this gap is NOT statistically
# significant -- a difference this small is easily explained by random
# variation in which jobs got posted, not a real pattern in the market.
# skills_count has a weak positive correlation with budget (~0.29),
# meaning the number of skills listed only modestly relates to what a
# job pays -- most of the budget variation comes from something else.