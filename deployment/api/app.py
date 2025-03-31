import os
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger("bank-model-api")

# Initialize FastAPI app
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="API for predicting term deposit subscriptions using a LightGBM model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Input data model for single prediction
class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str
    
    class Config:
        schema_extra = {
            "example": {
                "age": 30,
                "job": "management",
                "marital": "married",
                "education": "tertiary",
                "default": "no",
                "balance": 1476,
                "housing": "yes",
                "loan": "yes",
                "contact": "unknown",
                "day": 3,
                "month": "jun",
                "duration": 199,
                "campaign": 4,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown"
            }
        }

# Input for batch predictions
class BatchCustomerData(BaseModel):
    customers: List[CustomerData]

# Response model for predictions
class PredictionResponse(BaseModel):
    customer_id: Optional[int] = None
    prediction: bool
    probability: float
    prediction_label: str
    
# Batch prediction response
class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict[str, Union[int, float]]

# Model state to be loaded on startup
class ModelState:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.scaler = None

model_state = ModelState()

def get_model():
    """Dependency to get the loaded model"""
    if model_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return model_state

def preprocess_data(data, single=True):
    """Preprocess input data to match model training format"""
    try:
        # Convert to DataFrame
        if single:
            df = pd.DataFrame([data.dict()])
        else:
            df = pd.DataFrame([item.dict() for item in data])
            
        # Make a copy of the original data
        X = df.copy()
        
        # Get categorical and numerical columns based on data types
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        # One-hot encode categorical variables
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Ensure all columns expected by the model are present
        for col in model_state.feature_names:
            if col not in X.columns:
                X[col] = 0
                
        # Keep only columns that the model knows
        X = X[model_state.feature_names]
        
        # Scale numerical features
        if model_state.scaler:
            common_cols = list(set(numerical_cols) & set(model_state.feature_names))
            if common_cols:
                X[common_cols] = model_state.scaler.transform(X[common_cols])
                
        return X, df
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise HTTPException(
            status_code=422, 
            detail=f"Data preprocessing error: {str(e)}"
        )

def make_predictions(X, df, single=True):
    """Make predictions using the loaded model"""
    try:
        # Get predictions
        probabilities = model_state.model.predict(X)
        predictions = (probabilities > 0.5).astype(bool)
        
        # Format response
        if single:
            return PredictionResponse(
                prediction=bool(predictions[0]),
                probability=float(probabilities[0]),
                prediction_label="yes" if predictions[0] else "no"
            )
        else:
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                results.append(
                    PredictionResponse(
                        customer_id=i,
                        prediction=bool(pred),
                        probability=float(prob),
                        prediction_label="yes" if pred else "no"
                    )
                )
                
            # Create summary statistics
            positive_count = sum(predictions)
            summary = {
                "total_predictions": len(predictions),
                "positive_predictions": int(positive_count),
                "positive_percentage": float(positive_count / len(predictions) * 100),
                "average_probability": float(np.mean(probabilities))
            }
            
            return BatchPredictionResponse(
                predictions=results,
                summary=summary
            )
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """Load model and resources on startup"""
    try:
        # Get MLflow run ID from environment variable or use default
        run_id = os.getenv("MLFLOW_RUN_ID")
        
        if not run_id:
            logger.warning("No MLFLOW_RUN_ID provided, trying to find latest run")
            # Try to find the latest run
            try:
                from mlflow.tracking import MlflowClient
                
                # Get tracking URI
                tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
                mlflow.set_tracking_uri(tracking_uri)
                
                client = MlflowClient()
                experiments = client.search_experiments()
                
                if not experiments:
                    raise ValueError("No MLflow experiments found")
                
                # Sort experiments by creation time and get the latest
                experiments.sort(key=lambda x: x.creation_time, reverse=True)
                latest_experiment = experiments[0]
                
                # Get runs for the experiment
                runs = client.search_runs(
                    experiment_ids=[latest_experiment.experiment_id],
                    order_by=["start_time DESC"]
                )
                
                if not runs:
                    raise ValueError(f"No runs found for latest experiment: {latest_experiment.name}")
                
                # Get the latest run
                run_id = runs[0].info.run_id
                logger.info(f"Found latest run: {run_id} from experiment: {latest_experiment.name}")
            except Exception as e:
                logger.error(f"Error finding latest run: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load model: {str(e)}"
                )
        
        # Load model from MLflow
        logger.info(f"Loading model from run: {run_id}")
        model_uri = f"runs:/{run_id}/lightgbm_model"
        
        # Set model state
        model_state.model = mlflow.lightgbm.load_model(model_uri)
        model_state.feature_names = model_state.model.feature_name()
        
        # Load a sample of the training data to initialize preprocessing
        # This step assumes the CSV is in the same directory as the API
        try:
            # Try to load the original dataset to learn features
            df = pd.read_csv('bank.csv', delimiter=';')
            
            # Initialize preprocessing components
            X = df.drop('y', axis=1)
            
            # Get categorical and numerical columns
            model_state.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            model_state.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            # Initialize scaler
            model_state.scaler = StandardScaler()
            if model_state.numerical_cols:
                model_state.scaler.fit(X[model_state.numerical_cols])
                
            logger.info("Preprocessing components initialized from training data")
        except Exception as e:
            logger.warning(f"Could not load original dataset for preprocessing: {str(e)}")
            logger.warning("Continuing without preprocessing initialization")
            
        logger.info(f"Model loaded successfully with {len(model_state.feature_names)} features")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Continue app initialization, but endpoints will return errors until model is loaded
        
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Bank Marketing Prediction API",
        "status": "active",
        "model_loaded": model_state.model is not None,
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    customer: CustomerData,
    background_tasks: BackgroundTasks,
    model=Depends(get_model)
):
    """Predict subscription probability for a single customer"""
    # Log request (asynchronously)
    background_tasks.add_task(
        logger.info, 
        f"Prediction request received: {json.dumps(customer.dict())}"
    )
    
    # Preprocess the input data
    X, original_df = preprocess_data(customer, single=True)
    
    # Make prediction
    result = make_predictions(X, original_df, single=True)
    
    # Log result (asynchronously)
    background_tasks.add_task(
        logger.info, 
        f"Prediction result: {json.dumps(result.dict())}"
    )
    
    return result

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch: BatchCustomerData,
    background_tasks: BackgroundTasks,
    model=Depends(get_model)
):
    """Predict subscription probabilities for multiple customers"""
    # Log batch size (asynchronously)
    background_tasks.add_task(
        logger.info, 
        f"Batch prediction request received with {len(batch.customers)} customers"
    )
    
    if len(batch.customers) == 0:
        raise HTTPException(status_code=400, detail="Batch request must contain at least one customer")
    
    if len(batch.customers) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limited to 1000 customers")
    
    # Preprocess the batch data
    X, original_df = preprocess_data(batch.customers, single=False)
    
    # Make batch predictions
    results = make_predictions(X, original_df, single=False)
    
    # Log summary (asynchronously)
    background_tasks.add_task(
        logger.info, 
        f"Batch prediction complete: {json.dumps(results.summary)}"
    )
    
    return results

@app.get("/model/info")
async def model_info(model=Depends(get_model)):
    """Get information about the loaded model"""
    # Get model metadata - be cautious with feature names to prevent memory issues
    return {
        "status": "loaded",
        "feature_count": len(model_state.feature_names),
        "feature_sample": model_state.feature_names[:10] if len(model_state.feature_names) > 10 else model_state.feature_names,
        "categorical_features": model_state.categorical_cols,
        "numerical_features": model_state.numerical_cols,
        "preprocessing_initialized": model_state.scaler is not None
    }

@app.post("/reload_model")
async def reload_model(run_id: Optional[str] = None):
    """Reload the model (admin endpoint)"""
    try:
        # Set environment variable if run_id provided
        if run_id:
            os.environ["MLFLOW_RUN_ID"] = run_id
            
        # Reset model state
        model_state.model = None
        model_state.feature_names = None
        
        # Call startup function to reload model
        await startup_event()
        
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {str(e)}"
        )

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)