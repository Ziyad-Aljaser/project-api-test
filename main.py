from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel

app = FastAPI()

# GET request
@app.get("/")
def read_root():
    return {"message": "Coursera Prediction"}

# Load your models
model = joblib.load('kmens_model.joblib')
scaler = joblib.load('kmens_scaler.joblib')

# Define input data model
class InputFeatures(BaseModel):
    Provider: str
    Level: str
    Type: str
    Duration_Weeks: str

def preprocessing(input_features: InputFeatures):
    dict_f = {
        'Provider': input_features.Provider,
        'Level': input_features.Level,
        'Type': input_features.Type,
        'Duration_Weeks': input_features.Duration_Weeks
    }
    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    # Scale the input features
    scaled_features = scaler.transform([features_list])
    return scaled_features

@app.get("/predict")
def predict(input_features: InputFeatures):
    return preprocessing(input_features)


# Define the /predict endpoint to handle POST requests
@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]}