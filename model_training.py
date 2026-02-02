import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = {
    "screen_time": [100, 200, 300, 50, 80, 350, 400],
    "unlocks": [20, 40, 60, 10, 15, 70, 80],
    "notification_count": [50, 120, 200, 30, 40, 250, 300],
    "call_minutes": [10, 25, 30, 5, 8, 40, 45],
    "stress": [0, 1, 2, 0, 0, 2, 2]
}

df = pd.DataFrame(data)

X = df.drop("stress", axis=1)
y = df["stress"]

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
print("Model saved as model.pkl")
