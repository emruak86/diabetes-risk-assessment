from flask import render_template, request, jsonify
import numpy as np
import flask
from catboost import CatBoostClassifier

app = flask.Flask(__name__)

# Load the trained CatBoost model
model = CatBoostClassifier()
model.load_model("Diabetes_Prediction/Model/diabetes_model.cbm")

@app.route('/')
def home():
    """Serve the UI template"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get JSON data from the request

        # Extract and convert data types from the JSON data
        input_features = [
            data["gender"],
            float(data["age"]),
            int(data["hypertension"]),
            int(data["heart_disease"]),
            data["smoking_history"],
            float(data["bmi"]),
            float(data["hba1c"]),
            float(data["blood_glucose_level"])
        ]

        final_input = np.array(input_features).reshape(1, -1)

        # Get the probability of being diabetic (class 1)
        diabetes_prob = model.predict_proba(final_input)[0][1]

        # Convert probability to risk percentage
        risk_percent = round(diabetes_prob * 100, 2)

        # Classify risk level based on percentage
        if risk_percent < 30:
            risk_level = "Low Risk"
        elif risk_percent < 60:
            risk_level = "Moderate Risk"
        elif risk_percent < 80:
            risk_level = "High Risk"
        else:
            risk_level = "Very High Risk"

        # Return a JSON response
        return jsonify({
            'risk_percent': risk_percent,
            'risk_level': risk_level
        })

    except Exception as e:
        # Return a JSON error message for better debugging
        return jsonify({'error': 'Invalid input or server error.', 'details': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)