from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load your models
try:
    model = joblib.load('kmens_model.joblib')
    scaler = joblib.load('kmens_scaler.joblib')
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise RuntimeError(f"Error loading models: {e}")


# Define input data model
class InputFeatures(BaseModel):
    Provider: str
    Level: str
    Type: str
    Duration_Weeks: str


def preprocessing(input_features: InputFeatures):
    # Map input features to expected order and types
    dict_f = {
        'Provider': input_features.Provider,
        'Level': input_features.Level,
        'Type': input_features.Type,
        'Duration_Weeks': input_features.Duration_Weeks
    }

    # Check and log feature values
    logger.info(f"Received features: {dict_f}")

    # Convert categorical data to numerical values or encode them if necessary
    # Example: Simple encoding (adjust as per your training method)
    feature_mapping = {
        'Provider': {'IBM': 0, 'Coursera': 1},  # Example mapping, adjust as needed
        'Level': {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2},
        'Type': {'Course': 0, 'Specialization': 1},
        'Duration_Weeks': {'1-4': 0, '5-8': 1, '9-12': 2}  # Example
    }

    try:
        # Encode the features using the mapping
        features_list = [feature_mapping[key][dict_f[key]] for key in
                         ['Provider', 'Level', 'Type', 'Duration_Weeks']]
        logger.info(f"Encoded features: {features_list}")
    except KeyError as e:
        logger.error(f"Invalid feature value: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid feature value: {e}")

    # Scale the input features
    try:
        scaled_features = scaler.transform(
            [features_list])  # Ensure 2D array format
        logger.info(f"Scaled features: {scaled_features}")
    except Exception as e:
        logger.error(f"Error in feature scaling: {e}")
        raise HTTPException(status_code=400,
                            detail=f"Error in feature scaling: {e}")

    return scaled_features


@app.get("/")
def read_root():
    return {"message": "Coursera Prediction"}


@app.post("/predict")
async def predict(input_features: InputFeatures):
    try:
        data = preprocessing(input_features)
        y_pred = model.predict(data)
        logger.info(f"Prediction: {y_pred}")
        return {"pred": y_pred.tolist()[0]}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
