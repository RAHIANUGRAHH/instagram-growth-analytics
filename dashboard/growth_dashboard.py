import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import stats

st.set_page_config(page_title="Instagram Growth Analytics", layout="wide")

# --------------------------
# LOAD DATA SAFELY
# --------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "campaign_data", "clean_data", "influencer_metrics.csv")

df = pd.read_csv(DATA_PATH)

st.title("🚀 Instagram Campaign Growth Dashboard")

# --------------------------
# KPI SECTION
# --------------------------

st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Influencers", len(df))
col2.metric("Total Purchases", int(df["purchase"].sum()))
col3.metric("Average CTR", f"{df['ctr'].mean():.2f}")
col4.metric("Avg Conversion Rate", f"{df['conversion_rate'].mean():.2f}")

# --------------------------
# ENGAGEMENT DISTRIBUTION
# --------------------------

st.subheader("📈 Engagement Score Distribution")
st.bar_chart(df["engagement_score"])

# --------------------------
# A/B TEST SECTION
# --------------------------

st.subheader("🧪 A/B Test: Low vs High Engagement")

median_score = df["engagement_score"].median()
group_A = df[df["engagement_score"] <= median_score]
group_B = df[df["engagement_score"] > median_score]

conv_A = group_A["purchase"].mean()
conv_B = group_B["purchase"].mean()

n_A = len(group_A)
n_B = len(group_B)

success_A = group_A["purchase"].sum()
success_B = group_B["purchase"].sum()

p_pool = (success_A + success_B) / (n_A + n_B)
z = (conv_B - conv_A) / np.sqrt(p_pool * (1 - p_pool) * ((1/n_A) + (1/n_B)))
p_value = 1 - stats.norm.cdf(z)

colA, colB, colC = st.columns(3)

colA.metric("Conversion A", f"{conv_A:.2f}")
colB.metric("Conversion B", f"{conv_B:.2f}")
colC.metric("P-value", f"{p_value:.4f}")

if p_value < 0.05:
    st.success("Statistically Significant Difference 🚀")
else:
    st.warning("No Significant Difference")

# --------------------------
# MODEL SECTION
# --------------------------

st.subheader("🤖 Purchase Prediction Model")

features = [
    "likes", "comments", "shares", "saves",
    "profile_visits", "link_clicks",
    "engagement_score", "ctr"
]

X = df[features]
y = df["purchase"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

st.subheader("📊 Feature Importance")
st.bar_chart(importance.set_index("Feature"))

# --------------------------
# PREDICTION TOOL
# --------------------------

st.subheader("🔮 Predict Purchase Probability")

likes = st.number_input("Likes", value=1000)
comments = st.number_input("Comments", value=50)
shares = st.number_input("Shares", value=30)
saves = st.number_input("Saves", value=40)
profile_visits = st.number_input("Profile Visits", value=200)
link_clicks = st.number_input("Link Clicks", value=50)

engagement_score = (
    (0.2 * likes) +
    (0.5 * comments) +
    (0.7 * shares) +
    (0.8 * saves)
)

ctr = link_clicks / profile_visits if profile_visits > 0 else 0

input_data = [[
    likes, comments, shares, saves,
    profile_visits, link_clicks,
    engagement_score, ctr
]]

prediction = model.predict_proba(input_data)[0][1]

st.metric("Predicted Purchase Probability", f"{prediction:.2f}")