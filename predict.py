import joblib
import pandas as pd
from flask import Flask, request, jsonify

MODEL_PATH = "netflix_type_rf_model.pkl"
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Netflix Classifier API is running"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        return jsonify({"input": data, "prediction": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
