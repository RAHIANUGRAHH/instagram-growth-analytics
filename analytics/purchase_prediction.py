import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load processed data
df = pd.read_csv("../campaign_data/clean_data/influencer_metrics.csv")

# -------------------------
# SELECT FEATURES
# -------------------------

features = [
    "likes",
    "comments",
    "shares",
    "saves",
    "profile_visits",
    "link_clicks",
    "engagement_score",
    "ctr"
]

X = df[features]
y = df["purchase"]

# -------------------------
# TRAIN TEST SPLIT
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# MODEL TRAINING
# -------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------
# PREDICTIONS
# -------------------------

y_pred = model.predict(X_test)

# -------------------------
# EVALUATION
# -------------------------

print("----- Model Evaluation -----")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------
# FEATURE IMPORTANCE
# -------------------------

importance = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
})

importance = importance.sort_values(by="Coefficient", ascending=False)

print("\n----- Feature Importance -----")
print(importance)