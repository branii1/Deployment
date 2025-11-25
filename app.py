import joblib
import pandas as pd
from flask import Flask, request, jsonify
import numpy as np

# Load the model
MODEL_PATH = "netflix_type_rf_model.pkl"
model = joblib.load(MODEL_PATH)

# Create Flask app - FIXED: Use consistent naming
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Netflix Classifier API is running 🎬",
        "endpoints": {
            "health": "/health (GET)",
            "predict": "/predict (POST)"
        },
        "example_request": {
            "features": [2020, 90, 5, 0, 0, 0]  # Adjust based on your model
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model_loaded": True}), 200

@app.route('/predict', methods=['POST'])
def predict_endpoint():  # FIXED: Renamed to avoid conflict
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Convert to DataFrame for model prediction
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0].tolist()
        
        return jsonify({
            "input": data, 
            "prediction": str(prediction),
            "probability": probability,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict', methods=['GET'])
def predict_get():
    return """
    <html>
        <head><title>Netflix Predictor</title></head>
        <body>
            <h1>🎬 Netflix Content Predictor</h1>
            <p>Send a POST request with JSON data to get predictions</p>
            <h3>Example using curl:</h3>
            <code>
                curl -X POST http://localhost:8080/predict \<br>
                -H "Content-Type: application/json" \<br>
                -d '{"feature1": 2020, "feature2": 90, "feature3": 5, "feature4": 0, "feature5": 0, "feature6": 0}'
            </code>
        </body>
    </html>
    """

if __name__ == '__main__':
    print("🚀 Starting Netflix Prediction API...")
    print("✅ Model loaded successfully!")
    app.run(host="0.0.0.0", port=8080, debug=True)