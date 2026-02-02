from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained ML model
model = joblib.load("stress_model.pkl")

# ------------ Main API used by Android -------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        
        features = np.array([
            data.get("screen_time", 0),
            data.get("unlocks", 0),
            data.get("night_usage", 0),
            data.get("call_duration", 0)
        ]).reshape(1, -1)

        prediction = int(model.predict(features)[0])

        return jsonify({
            "status": "success",
            "stress_level": prediction
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# --------------------------------------------
# Keep your older route — just for alternative use
@app.route("/mobile_activity", methods=["POST"])
def mobile_activity():
    return predict()

@app.route("/upload", methods=["POST"])
def upload_data():
    return predict()

# --------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
