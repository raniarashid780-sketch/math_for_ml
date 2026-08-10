"""Day 11 : Descriptive statistics"""
import numpy as np
A = np.array([12, 15, 14, 10, 18, 200, 13, 16, 14, 11])

def manual_mean(a):
    return np.sum(a) / a.size

def manual_variance(a):
    a = np.asarray(a).ravel()
    m = manual_mean(a)
    return ((a - m) ** 2).sum() / a.size

def manual_std(a):
    return np.sqrt(manual_variance(a))

print(f"Manual mean:{manual_mean(A)}")
print(f"np.mean :{np.mean(A)}")
print(f"Manual variance :{manual_variance(A)}")
print(f"np.var :{np.var(A)}")
print(f"Manual std :{manual_std(A)}")
print(f"np.std :{np.std(A)}")

assert np.isclose(manual_mean(A), np.mean(A)), "Manual mean doesn't match"
assert np.isclose(manual_variance(A), np.var(A)), "Manual variance doesn't match"
assert np.isclose(manual_std(A), np.std(A)), "Manual std doesn't match"

print(f"Mean :{np.mean(A)}  Median :{np.median(A)}")
# The mean is significantly affected by the outlier value of 200 in the dataset, which skews it higher than where most of the data points are concentrated.
#  The median, on the other hand, is less sensitive to extreme values and better represents the central tendency of the majority of the data.
# Therefore, in this case, the median is closer to where most of the data actually sits.

p25, p50, p75 = np.percentile(A, [25, 50, 75])
print(f"25th percentile: {p25}, Median (50th): {p50}, 75th percentile: {p75}")

# Why does high variance in a feature matter before training a model?
# High variance in a feature indicates that the data points are spread out over a wide range of values. This can lead to instability in model training, as the model may struggle to learn patterns from features that have inconsistent or extreme values.
# Features with high variance can dominate the learning process, potentially overshadowing other important features and leading to overfitting.
# Normalizing or standardizing features with high variance is often necessary to ensure that all features contribute equally to the model's learning process.