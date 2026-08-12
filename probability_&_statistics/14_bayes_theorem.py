"""Day 14: Bayes' Theorem"""

def bayes(p_a, p_b_given_a, p_b_given_not_a):
    p_b = p_b_given_a * p_a + p_b_given_not_a * (1 - p_a)
    result = (p_b_given_a * p_a) / p_b
    assert 0 <= result <= 1
    return result

posterior = bayes(0.20, 0.95, 0.05)
assert 0 <= posterior <= 1
print(posterior)
posterior1 = bayes(0.05, 0.95, 0.05)
assert 0 <= posterior1 <= 1
print(posterior1)
posterior2 = bayes(0.50, 0.95, 0.05)
assert 0 <= posterior2 <= 1
print(posterior2)

p_a = 0.10
p_b_given_a = 0.70
p_b_given_not_a = 0.25

result = bayes(p_a, p_b_given_a, p_b_given_not_a)
assert 0 <= result <= 1
print(f"P(pays >$50/hr | 5+ skills required) = {result:.4f}")
# Naive Bayes in sklearn uses this exact formula I wrote above — but instead of
# one piece of evidence (like one flag or one skill-count check), it multiplies
# together the likelihoods from MANY features at once (every word in an email,
# every column in a row of data) to get one posterior probability per class.
# It's called "naive" because it assumes all those features are independent of
# each other, which usually isn't fully true in real data — but the shortcut
# still works well in practice.