from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np 
import shap
# SHAP has different explainers for different model types:

# TreeExplainer → Random Forest, XGBoost, LightGBM ✅ (our case)
# LinearExplainer → Logistic Regression, Linear Regression
# DeepExplainer → Neural Networks
# KernelExplainer → any model but very slow
app=FastAPI()
from fastapi.middleware.cors import CORSMiddleware
# FastAPI is running on http://localhost:8000 
# and your HTML file will open from the file system. 
# When JS tries to call the API from the HTML file,
# 
# the browser will block it for security reasons — 
# this is called CORS (Cross Origin Resource Sharing).

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
model = joblib.load("random_forest_model.pkl")

class CustomerData(BaseModel):
    credit_score: int
    age: int
    tenure: int
    balance: float
    products_number: int
    credit_card: int
    active_member: int
    estimated_salary: float
#This is basically telling FastAPI — "when someone sends data to our API, 
# it must have exactly these fields with these data types."
#→ Pydantic is a data validation library.
# We'll use it to define what input data our API expects — 
# like "age must be a number, balance must be a float". If 
# someone sends wrong data, Pydantic automatically rejects it with an error.
explainer=shap.TreeExplainer(model)
@app.post("/predict")

def predict(customer: CustomerData):
    data = pd.DataFrame([customer.dict()])
    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]
    shap_values = explainer.shap_values(data)
    print(shap_values.shape)
    print(shap_values)
    # feature_impact = {k: round(float(v), 4) for k, v in zip(data.columns, shap_values[0])}
    feature_impact = {k: round(float(v), 4) for k, v in zip(data.columns, shap_values[0, :, 1])}
    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": round(float(probability), 2),
        "feature_impact":feature_impact
        
        
    }
