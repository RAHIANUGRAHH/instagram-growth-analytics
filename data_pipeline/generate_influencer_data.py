import pandas as pd
import numpy as np
import os

np.random.seed(42)

rows = 3000
data = []

for i in range(rows):
    
    influencer_id = np.random.randint(1, 100)
    followers = np.random.randint(5000, 500000)
    
    likes = int(followers * np.random.uniform(0.02, 0.1))
    comments = int(likes * np.random.uniform(0.01, 0.05))
    shares = int(likes * np.random.uniform(0.01, 0.03))
    saves = int(likes * np.random.uniform(0.02, 0.06))
    
    profile_visits = int(likes * np.random.uniform(0.1, 0.3))
    link_clicks = int(profile_visits * np.random.uniform(0.1, 0.4))
    
    purchase_prob = (link_clicks * 0.002) + (saves * 0.0005)
    purchase = np.random.binomial(1, min(purchase_prob, 0.9))
    
    data.append([
        influencer_id,
        followers,
        likes,
        comments,
        shares,
        saves,
        profile_visits,
        link_clicks,
        purchase
    ])

columns = [
    "influencer_id",
    "followers",
    "likes",
    "comments",
    "shares",
    "saves",
    "profile_visits",
    "link_clicks",
    "purchase"
]

df = pd.DataFrame(data, columns=columns)

os.makedirs("../campaign_data/raw_events", exist_ok=True)
df.to_csv("../campaign_data/raw_events/influencer_events.csv", index=False)

print("Influencer dataset generated successfully!")