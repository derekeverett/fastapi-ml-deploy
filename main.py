from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.types import conlist
# from typing import Annotated
import joblib
import numpy as np

class IrisRequest(BaseModel):
    features: conlist(float, min_length=4, max_length=4)

app = FastAPI()
model = joblib.load("model.joblib")

@app.post("/predict")
def predict(req: IrisRequest):
    prediction = model.predict(np.array([req.features]))
    return {"prediction": int(prediction[0])}