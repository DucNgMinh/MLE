import mlflow
import mlflow.lightgbm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_model_and_predict(run_id, data_path):
    """
    Load a trained model from MLflow and use it to make predictions on new data.
    
    Parameters:
    run_id (str): The MLflow run ID where the model is stored
    data_path (str): Path to the CSV file with new data for prediction
    
    Returns:
    pandas.DataFrame: Original data with prediction results
    """
    print(f"Loading model from run: {run_id}")
    
    # Load the model from MLflow
    model = mlflow.lightgbm.load_model(f"runs:/{run_id}/lightgbm_model")
    
    # Load and preprocess the data similar to training
    df = pd.read_csv(data_path, delimiter=';')
    
    # Keep a copy of the original data
    original_df = df.copy()
    
    # Check if target column exists in the data
    if 'y' in df.columns:
        X = df.drop('y', axis=1)
    else:
        X = df.copy()
    
    # Handle categorical variables with one-hot encoding
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Make predictions
    predictions_proba = model.predict(X)
    predictions = (predictions_proba > 0.5).astype(int)
    
    # Add predictions to the original data
    original_df['prediction_probability'] = predictions_proba
    original_df['prediction'] = predictions.astype(int)
    original_df['prediction_label'] = np.where(predictions == 1, 'yes', 'no')
    
    print(f"Made predictions for {len(original_df)} records")
    print(f"Positive predictions: {sum(predictions)}")
    
    return original_df

def get_latest_run_id(experiment_name=None):
    """
    Get the run ID of the latest MLflow run in the specified experiment.
    If no experiment name is provided, use the latest experiment.
    
    Parameters:
    experiment_name (str, optional): Name of the MLflow experiment
    
    Returns:
    str: Run ID of the latest run
    """
    if experiment_name:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found")
            return None
        experiment_id = experiment.experiment_id
    else:
        # Get all experiments and choose the latest one
        experiments = mlflow.search_experiments()
        if not experiments:
            print("No experiments found")
            return None
        # Sort by creation time and get the latest
        experiments.sort(key=lambda x: x.creation_time, reverse=True)
        experiment_id = experiments[0].experiment_id
        print(f"Using latest experiment: {experiments[0].name}")
    
    # Get runs for the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    if runs.empty:
        print(f"No runs found for experiment {experiment_id}")
        return None
    
    # Sort by start time and get the latest run
    latest_run = runs.sort_values("start_time", ascending=False).iloc[0]
    run_id = latest_run.run_id
    
    print(f"Found latest run: {run_id}")
    return run_id

if __name__ == "__main__":
    # Example usage
    # Option 1: Automatically use the latest run
    run_id = get_latest_run_id()
    
    # Option 2: Specify a run ID directly
    # run_id = "your-run-id-here"
    
    if run_id:
        # Make predictions using the loaded model
        predictions_df = load_model_and_predict(run_id, 'bank.csv')
        
        # Save the predictions to a CSV file
        output_path = 'bank_predictions.csv'
        predictions_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
