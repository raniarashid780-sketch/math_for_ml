"""Day 12: Probability basics"""
import numpy as np
rolls = np.random.randint(1, 7, size=10000)

p_six = np.sum(rolls == 6) / len(rolls)

even_rolls = rolls[rolls % 2 == 0]        # keep only even rolls
p_gt3_given_even = np.sum(even_rolls > 3) / len(even_rolls)

p_gt3 = np.sum(rolls > 3) / len(rolls)
p_even = np.sum(rolls % 2 == 0) / len(rolls)
p_both = np.sum((rolls > 3) & (rolls % 2 == 0)) / len(rolls)

print(f"P(>3) * P(even) = {p_gt3 * p_even:.4f}")
print(f"P(>3 and even)  = {p_both:.4f}")
# Why does conditional probability matter in ML?
# Conditional probability is important in machine learning because it helps us understand the relationship between different features and outcomes.
# By analyzing how the probability of one event changes when we know that another event has occurred, we can make more informed predictions and decisions.
# This is particularly useful in classification problems, where we want to predict the likelihood of a certain class given specific input features.
# Understanding conditional probabilities allows us to build models that can capture complex dependencies in the data, leading to better performance and more accurate predictions.