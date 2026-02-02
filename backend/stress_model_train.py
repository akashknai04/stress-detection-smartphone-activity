import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Sample data
data = {
    "screen_time": [120, 300, 90, 400, 30, 500, 250, 180, 60, 350],
    "unlocks": [40, 120, 20, 200, 10, 220, 130, 90, 25, 160],
    "night_usage": [30, 90, 10, 120, 5, 150, 80, 50, 8, 110],
    "call_duration": [5, 20, 2, 35, 1, 40, 22, 15, 3, 30],
    "stress_level": [1, 3, 1, 4, 0, 4, 3, 2, 0, 3]
}

df = pd.DataFrame(data)

# Features and target
X = df[["screen_time", "unlocks", "night_usage", "call_duration"]]
y = df["stress_level"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "stress_model.pkl")

print("Model created successfully! File saved as stress_model.pkl")
