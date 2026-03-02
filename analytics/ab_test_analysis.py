import pandas as pd
import numpy as np
from scipy import stats

# Load processed data
df = pd.read_csv("../campaign_data/clean_data/influencer_metrics.csv")

# Split into two groups based on engagement score
median_score = df["engagement_score"].median()

group_A = df[df["engagement_score"] <= median_score]
group_B = df[df["engagement_score"] > median_score]

# Calculate conversion rates
conv_A = group_A["purchase"].mean()
conv_B = group_B["purchase"].mean()

# Number of samples
n_A = len(group_A)
n_B = len(group_B)

# Number of successes
success_A = group_A["purchase"].sum()
success_B = group_B["purchase"].sum()

# Perform Z-test
p_pool = (success_A + success_B) / (n_A + n_B)
z = (conv_B - conv_A) / np.sqrt(p_pool * (1 - p_pool) * ((1/n_A) + (1/n_B)))
p_value = 1 - stats.norm.cdf(z)

print("----- A/B Test Results -----")
print(f"Conversion Rate A: {conv_A:.4f}")
print(f"Conversion Rate B: {conv_B:.4f}")
print(f"Z-score: {z:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: Statistically Significant Difference 🚀")
else:
    print("Result: No Significant Difference")