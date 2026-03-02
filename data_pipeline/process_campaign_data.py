import pandas as pd
import os

# Paths
RAW_PATH = "../campaign_data/raw_events/influencer_events.csv"
CLEAN_PATH = "../campaign_data/clean_data/influencer_metrics.csv"

# Make sure clean_data folder exists
os.makedirs("../campaign_data/clean_data", exist_ok=True)

# Load raw data
df = pd.read_csv(RAW_PATH)

# -------------------------
# FEATURE ENGINEERING
# -------------------------

# Engagement Score (weighted)
df["engagement_score"] = (
    (0.2 * df["likes"]) +
    (0.5 * df["comments"]) +
    (0.7 * df["shares"]) +
    (0.8 * df["saves"])
)

# Click Through Rate (CTR)
df["ctr"] = df["link_clicks"] / df["profile_visits"]

# Conversion Rate
df["conversion_rate"] = df["purchase"] / df["link_clicks"]

# Replace infinite values (if link_clicks = 0)
df.replace([float("inf"), -float("inf")], 0, inplace=True)
df.fillna(0, inplace=True)

# Save cleaned data
df.to_csv(CLEAN_PATH, index=False)

print("Campaign data processed successfully!")