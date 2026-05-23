import gradio as gr
import pandas as pd
import joblib
import numpy as np
import os

# load model (try a couple common filenames)
MODEL_FILES = ['diabetes-dataset_pipeline.pkl', 'diabetes_pipeline.pkl']
model_path = next((p for p in MODEL_FILES if os.path.exists(p)), MODEL_FILES[0])
model = joblib.load(model_path)


def predict_diabetes(
    Pregnancies,
    Glucose,
    BloodPressure,
    SkinThickness,
    Insulin,
    BMI,
    DiabetesPedigreeFunction,
    Age
):
    data = pd.DataFrame([[
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ]], columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])

    # Get raw prediction and probabilities (if available) for debugging
    pred = model.predict(data)
    prob = None
    if hasattr(model, 'predict_proba'):
        try:
            prob = model.predict_proba(data)[0]
        except Exception:
            prob = None

    # Log to console for debugging when running locally
    print('INPUTS:', data.to_dict(orient='records'))
    print('PRED:', pred, 'PROB:', prob)

    label = 'Diabetic' if int(pred[0]) == 1 else 'Non-Diabetic'
    if prob is not None:
        confidence = round(float(np.max(prob)) * 100, 2)
        return f"{label} (confidence: {confidence}%)"
    return label


app = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age")
    ],
    outputs="text",
    title="Diabetes Prediction App"
)

if __name__ == '__main__':
    # Use a different port to avoid conflicts if 7860 is already in use
    app.launch(server_name="0.0.0.0", server_port=7870)
