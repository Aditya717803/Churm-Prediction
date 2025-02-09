from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("model.joblib")

# Define request body schema
class FormData(BaseModel):
    gender: str
    seniorCitizen: str
    dependents: str
    phoneService: str
    onlineSecurity: str
    onlineBackup: str
    techsupport: str
    contract: str
    paperlessbilling: str
    monthlycharge: float  # Ensure it's a float

# Encoding map with default fallback
encoding_map = {
    "Male": 0, "Female": 1,
    "no": 0, "yes": 1,
    "Month-to-month": 1, "One year": 12, "Two year": 24
}

@app.post("/ml-model")
async def process_data(data: FormData):
    input_data = np.array([[
        encoding_map.get(data.gender, -1),
        encoding_map.get(data.seniorCitizen, -1),
        encoding_map.get(data.dependents, -1),
        encoding_map.get(data.phoneService, -1),
        encoding_map.get(data.onlineSecurity, -1),
        encoding_map.get(data.onlineBackup, -1),
        encoding_map.get(data.techsupport, -1),
        encoding_map.get(data.contract, -1),
        encoding_map.get(data.paperlessbilling, -1),
        data.monthlycharge  # Use as a float instead of encoding
    ]])
    # Check feature count before prediction
    # if input_data.shape[1] != model.n_features_in_:
    #     return {"error": f"Model expects {model.n_features_in_} features, but received {input_data.shape[1]}"}
    print("------------",input_data)
    
    try:
        prediction = model.predict(input_data)

    except Exception as e:
        print(e)

    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
