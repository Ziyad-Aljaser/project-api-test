from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load your models
try:
    model = joblib.load('kmens_model.joblib')
    scaler = joblib.load('kmens_scaler.joblib')
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")


# Define input data model
class InputFeatures(BaseModel):
    Provider: str
    Level: str
    Type: str
    Duration_Weeks: str


def preprocessing(input_features: InputFeatures):
    # Create a dictionary with input features in the required order
    dict_f = {
        'Provider': input_features.Provider,
        'Level': input_features.Level,
        'Type': input_features.Type,
        'Duration_Weeks': input_features.Duration_Weeks
    }

    # Ensure the features list is created in the correct order expected by the model
    # The order of keys here should match the order expected by the scaler/model
    features_list = [dict_f[key] for key in
                     ['Provider', 'Level', 'Type', 'Duration_Weeks']]

    # Convert features to a 2D array to be compatible with scaler.transform
    try:
        scaled_features = scaler.transform(
            [features_list])  # Ensure this is a 2D array
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Error in feature scaling: {e}")

    return scaled_features


@app.get("/")
def read_root():
    return {"message": "Coursera Prediction"}


# Define the /predict endpoint to handle POST requests
@app.post("/predict")
async def predict(input_features: InputFeatures):
    try:
        data = preprocessing(input_features)
        y_pred = model.predict(data)
        return {"pred": y_pred.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
