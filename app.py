from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model (adjust filename if needed)
model = pickle.load(open("model.pkl", "rb"))


@app.route('/', methods=['GET'])
def index():
    return (
        "<h1>Diabetes Prediction API</h1>"
        "<p>Use <code>/predict</code> (POST) with a JSON payload containing the features:</p>"
        "<pre>[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]</pre>"
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Extract features from JSON
    features = [
        data['Pregnancies'],
        data['Glucose'],
        data['BloodPressure'],
        data['SkinThickness'],
        data['Insulin'],
        data['BMI'],
        data['DiabetesPedigreeFunction'],
        data['Age']
    ]

    prediction = model.predict([features])
    probability = model.predict_proba([features])

    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    confidence = round(np.max(probability) * 100, 2)

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
