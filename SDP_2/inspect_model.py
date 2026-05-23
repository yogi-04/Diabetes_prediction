import os
import joblib
import numpy as np

candidates = ['diabetes_pipeline.pkl', 'diabetes-dataset_pipeline.pkl', 'diabetes-dataset_pipeline.pkl']
model_path = None
for c in candidates:
    if os.path.exists(c):
        model_path = c
        break
if model_path is None:
    print('No model file found. Checked:', candidates)
    raise SystemExit(1)

print('Loading model from', model_path)
model = joblib.load(model_path)

print('Model type:', type(model))
try:
    print('Model repr:', repr(model)[:1000])
except Exception:
    pass

# If sklearn pipeline
if hasattr(model, 'named_steps'):
    print('Pipeline steps:', list(model.named_steps.keys()))
    final = model.named_steps[list(model.named_steps.keys())[-1]]
    print('Final estimator type:', type(final))
else:
    final = model

if hasattr(final, 'classes_'):
    print('Classes:', getattr(final, 'classes_'))
else:
    print('No classes_ attribute on final estimator')

# Does model support predict_proba?
print('Has predict_proba:', hasattr(final, 'predict_proba'))
print('Has predict:', hasattr(final, 'predict'))

# Try sample inputs
samples = [
    [0, 110, 70, 20, 79, 25.6, 0.5, 25],  # typical non-diabetic-ish
    [3, 180, 90, 35, 200, 40.0, 1.2, 50],  # diabetic-ish
]

X = np.array(samples)

try:
    pred = model.predict(X)
    print('predict:', pred)
except Exception as e:
    print('predict failed:', e)

try:
    prob = model.predict_proba(X)
    print('predict_proba shape:', np.array(prob).shape)
    print('predict_proba:', prob)
except Exception as e:
    print('predict_proba failed:', e)

# If pipeline expects DataFrame with columns, try with DataFrame
import pandas as pd
cols = [
    'Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'
]
try:
    df = pd.DataFrame(samples, columns=cols)
    pred_df = model.predict(df)
    print('predict with DataFrame:', pred_df)
    try:
        prob_df = model.predict_proba(df)
        print('predict_proba with DataFrame:', prob_df)
    except Exception as e:
        print('predict_proba with DataFrame failed:', e)
except Exception as e:
    print('DataFrame predict failed:', e)

print('Done')
