from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.types import conlist
import joblib
import numpy as np

class Request(BaseModel):
    features: conlist(float, min_length=4, max_length=4)

app = FastAPI()
model = joblib.load("model.joblib")

@app.post("/predict")
def predict(req: Request):
    prediction = model.predict(np.array([req.features]))
    return {"prediction": int(prediction[0])}