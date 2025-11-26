import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify

MODEL_PATH = "netflix_type_rf_model.pkl"
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

# Try to get expected features from the model
def get_expected_features():
    """Extract expected feature names from the model if available."""
    try:
        # For scikit-learn models (version 1.0+), feature names are stored
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        # For pipeline models, check the final estimator
        elif hasattr(model, 'steps') and len(model.steps) > 0:
            final_estimator = model.steps[-1][1]
            if hasattr(final_estimator, 'feature_names_in_'):
                return list(final_estimator.feature_names_in_)
    except Exception:
        pass
    return None

EXPECTED_FEATURES = get_expected_features()

def validate_input(data):
    """
    Validate input data for prediction.
    Returns (is_valid, error_message)
    """
    # Check if data is provided
    if data is None:
        return False, "Request body is missing or invalid JSON"
    
    # Check if data is a dictionary
    if not isinstance(data, dict):
        return False, "Input data must be a JSON object (dictionary)"
    
    # Check if data is empty
    if len(data) == 0:
        return False, "Input data cannot be empty"
    
    # If we know the expected features, validate them
    if EXPECTED_FEATURES is not None:
        provided_features = set(data.keys())
        expected_features_set = set(EXPECTED_FEATURES)
        
        # Check for missing features
        missing_features = expected_features_set - provided_features
        if missing_features:
            return False, f"Missing required features: {', '.join(sorted(missing_features))}"
        
        # Check for extra features (warn but don't fail - they'll be ignored by pandas)
        extra_features = provided_features - expected_features_set
        if extra_features:
            # Log warning but continue (pandas will ignore extra columns)
            pass
    
    # Validate that values are not None and are numeric/convertible
    for key, value in data.items():
        if value is None:
            return False, f"Feature '{key}' cannot be None"
        
        # Try to convert to numeric (allows strings that represent numbers)
        try:
            float(value)
        except (ValueError, TypeError):
            # Allow string values (for categorical features)
            if not isinstance(value, (str, bool)):
                return False, f"Feature '{key}' has invalid type. Expected numeric or string, got {type(value).__name__}"
    
    return True, None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Netflix Classifier API is running",
        "endpoints": {
            "/": "Health check",
            "/predict": "POST - Make a prediction",
            "/features": "GET - Get expected feature names"
        }
    }), 200

@app.route('/features', methods=['GET'])
def get_features():
    """Endpoint to get the expected feature names for the model."""
    if EXPECTED_FEATURES is None:
        return jsonify({
            "message": "Feature names not available from model",
            "note": "Please check your model training code or documentation for required features"
        }), 200
    
    return jsonify({
        "expected_features": EXPECTED_FEATURES,
        "count": len(EXPECTED_FEATURES)
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Check Content-Type
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    try:
        data = request.get_json()
        
        # Validate input
        is_valid, error_message = validate_input(data)
        if not is_valid:
            return jsonify({"error": error_message}), 400
        
        # Convert to DataFrame
        try:
            df = pd.DataFrame([data])
        except Exception as e:
            return jsonify({"error": f"Failed to process input data: {str(e)}"}), 400
        
        # Ensure columns are in the correct order if we know expected features
        if EXPECTED_FEATURES is not None:
            # Reorder columns to match expected order
            df = df.reindex(columns=EXPECTED_FEATURES, fill_value=np.nan)
        
        # Make prediction
        try:
            prediction = model.predict(df)[0]
            prediction_proba = None
            
            # Try to get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(df)[0]
                classes = model.classes_ if hasattr(model, 'classes_') else None
                if classes is not None:
                    prediction_proba = dict(zip(classes, proba.tolist()))
            
            response = {
                "input": data,
                "prediction": str(prediction)
            }
            
            if prediction_proba:
                response["prediction_probabilities"] = prediction_proba
            
            return jsonify(response), 200
            
        except Exception as e:
            return jsonify({
                "error": f"Prediction failed: {str(e)}",
                "hint": "Please verify that all feature values are in the correct format"
            }), 400
            
    except Exception as e:
        return jsonify({
            "error": f"Unexpected error: {str(e)}",
            "hint": "Please check your request format and try again"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=0)  # Let OS choose available port
