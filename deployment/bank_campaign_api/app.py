import os
from typing import List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
import uvicorn

# Initialize the FastAPI app
app = FastAPI(
    title="Bank Marketing Model API",
    description="API for predicting customer subscription to certificate of deposit",
    version="1.0.0"
)

# Define the input data model based on your features
class CustomerData(BaseModel):
    customer_age: int = Field(..., description="Age of the customer in years", example=41)
    tenure: int = Field(..., description="How long the customer has been with the bank in months", example=24)
    account_balance: float = Field(..., description="Account balance in currency units", example=2567.89)
    previous_campaign_number_of_contacts: int = Field(0, description="Number of contacts performed during previous campaign", example=3)
    employment_type: str = Field(..., description="Type of employment", example="private")
    marital_status: str = Field(..., description="Marital status", example="married")
    education_level: str = Field(..., description="Education level", example="tertiary")
    credit_default: str = Field(..., description="Has credit in default?", example="no")
    housing_loan: str = Field(..., description="Has housing loan?", example="yes")
    personal_loan: str = Field(..., description="Has personal loan?", example="no")
    contact_method: str = Field(..., description="Contact communication type", example="cellular")
    previous_campaign_outcome: str = Field("unknown", description="Outcome of the previous marketing campaign", example="success")

# Define the prediction response model
class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Prediction (1 = will subscribe, 0 = will not subscribe)")
    probability: float = Field(..., description="Probability of subscription")
    explanation: dict = Field(..., description="Feature importance for this prediction")

# Load the model at startup
@app.on_event("startup")
async def load_model():
    global model, feature_names

    try:
        # Load model from MLflow or direct pickle file
        try:
            # First try loading from MLflow
            # Update this path to your MLflow model location
            # model_uri = "model.pkl"
            # app.model = mlflow.sklearn.load_model(model_uri)
            
            import joblib
            app.model = joblib.load("model.pkl")
        except Exception as e:
            # If MLflow loading fails, try direct pickle loading
            # import joblib
            # app.model = joblib.load("model.pkl")
            print("Model load unsuccessful")
            
        # Try to extract feature names from the model's preprocessing step
        try:
            preprocessor = app.model.named_steps['preprocessor']
            onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
            
            # Get feature names
            cat_feature_names = list(onehot.get_feature_names_out(
                preprocessor.transformers_[1][2]
            ))
            numeric_features = preprocessor.transformers_[0][2]
            app.feature_names = list(numeric_features) + cat_feature_names
        except:
            # If feature names can't be extracted, use generic names
            app.feature_names = None
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Define the prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    try:
        # Convert the input data to a pandas DataFrame
        input_df = pd.DataFrame([customer.dict()])
        
        # Make prediction
        prediction = int(app.model.predict(input_df)[0])
        probability = float(app.model.predict_proba(input_df)[0][1])
        
        # Get feature importances if available
        explanation = {}
        try:
            if hasattr(app.model, 'named_steps') and 'classifier' in app.model.named_steps:
                clf = app.model.named_steps['classifier']
                if hasattr(clf, 'feature_importances_'):
                    importances = clf.feature_importances_
                    if app.feature_names:
                        for i, importance in enumerate(importances):
                            if i < len(app.feature_names):
                                explanation[app.feature_names[i]] = float(importance)
                    else:
                        # Use generic feature names
                        for i, importance in enumerate(importances):
                            explanation[f"feature_{i}"] = float(importance)
        except Exception as e:
            explanation = {"error": str(e)}
            
        return {
            "prediction": prediction,
            "probability": probability,
            "explanation": explanation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    if hasattr(app, 'model') and app.model is not None:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

# Endpoint to get model information
@app.get("/model-info")
async def model_info():
    if not hasattr(app, 'model'):
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    info = {
        "model_type": str(type(app.model)),
        "features": {}
    }
    
    try:
        # Extract preprocessing information
        if hasattr(app.model, 'named_steps') and 'preprocessor' in app.model.named_steps:
            preprocessor = app.model.named_steps['preprocessor']
            
            # Get numeric features
            if len(preprocessor.transformers_) > 0:
                num_transformer = preprocessor.transformers_[0]
                if num_transformer[0] == 'num':
                    info["features"]["numeric"] = list(num_transformer[2])
            
            # Get categorical features
            if len(preprocessor.transformers_) > 1:
                cat_transformer = preprocessor.transformers_[1]
                if cat_transformer[0] == 'cat':
                    info["features"]["categorical"] = list(cat_transformer[2])
        
        # Get classifier information
        if hasattr(app.model, 'named_steps') and 'classifier' in app.model.named_steps:
            clf = app.model.named_steps['classifier']
            info["classifier"] = {
                "type": str(type(clf)),
                "parameters": clf.get_params(),
            }
            
    except Exception as e:
        info["error"] = str(e)
    
    return info

# Batch prediction endpoint
@app.post("/batch-predict")
async def batch_predict(customers: List[CustomerData]):
    if not hasattr(app, 'model'):
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert the input data to a pandas DataFrame
        input_df = pd.DataFrame([customer.dict() for customer in customers])
        
        # Make predictions
        predictions = [int(pred) for pred in app.model.predict(input_df)]
        probabilities = [float(prob[1]) for prob in app.model.predict_proba(input_df)]
        
        return {
            "predictions": predictions,
            "probabilities": probabilities
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# if __name__ == "__main__":
#     # Run the app with uvicorn when this script is executed directly
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
