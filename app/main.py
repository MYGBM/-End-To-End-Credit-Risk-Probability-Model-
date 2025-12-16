from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from app.schema import CreditScoringInput

# Initialize FastAPI app
app = FastAPI()

# Load model, encoder, and scaler
model_path = 'model/model.pkl'
scaler_path = 'model/scaler.pkl'
encoder_path = 'model/encoder.pkl'

try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    with open(encoder_path, 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)
except Exception as e:
    raise RuntimeError(f"Error loading model, encoder, or scaler: {str(e)}")

# Preprocessing function
def preprocess_input(data):
    input_data = [
        data.RFMS_Score,
        data.RecencyScore,
        data.PricingStrategy,
        data.ProductCategory
    ]

    numerical_features = input_data[:2]
    categorical_features = input_data[2:]

    scaled_numerical = scaler.transform(np.array(numerical_features).reshape(1, -1))
    encoded_categorical = encoder.transform([categorical_features])

    processed_input = np.hstack([scaled_numerical, encoded_categorical])
    return processed_input

# Prediction endpoint
@app.post("/predict")
def predict_credit_score(input_data: CreditScoringInput):
    try:
        processed_data = preprocess_input(input_data)
        prediction = model.predict(processed_data)
        label = "Good" if prediction == 1 else "Bad"
        return {"prediction": label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")