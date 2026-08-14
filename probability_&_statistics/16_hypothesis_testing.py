"""Day 16: Hypothesis Testing"""
from scipy.stats import ttest_ind

class_a_scores = [78, 82, 75, 90, 85, 79, 88, 76, 91, 83]
class_b_scores = [74, 70, 68, 80, 72, 75, 69, 77, 71, 73]

t_stat, p_value = ttest_ind(class_a_scores, class_b_scores)
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

# Is p_value above or below 0.05?
# Yes, the p-value is below 0.05, which means we reject the null hypothesis (H₀) that there is no difference in the mean scores between Class A and Class B. This suggests that there is a statistically significant difference in the average scores of the two classes. In plain English, it means that one class performed better than the other on average, and this difference is unlikely to be due to random chance.
# Based on that, do you reject or fail to reject H₀?
# We reject H₀, indicating that there is a significant difference in the mean scores between Class A and Class B.
# What does that conclusion mean about Class A vs Class B — plain English, no jargon?
# In plain English, it means that Class A performed better than Class B on average, and this difference is unlikely to be due to random chance.
