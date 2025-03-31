import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
import itertools
import argparse

# Set experiment name
experiment_name = f"bank_marketing_lightgbm_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
mlflow.set_experiment(experiment_name)

def load_and_preprocess_data(file_path):
    """Load and preprocess the bank dataset"""
    # Load data
    df = pd.read_csv(file_path, delimiter=';')
    
    # Separate features and target
    X = df.drop('y', axis=1)
    y = df['y'].apply(lambda x: 1 if x == 'yes' else 0)  # Convert to binary
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Apply one-hot encoding for categorical variables
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test, params):
    """Train LightGBM model and evaluate performance"""
    # Create dataset for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "best_iteration": model.best_iteration
    }
    
    return model, metrics

def hyperparameter_tuning(data_path):
    """Perform hyperparameter tuning with LightGBM and track with MLflow"""
    # Log data path parameter
    mlflow.log_param("data_path", data_path)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    
    # Define hyperparameter search space
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 63, 127],
        'max_depth': [5, 10, -1],  # -1 means no limit
        'min_data_in_leaf': [10, 20, 50],
        'feature_fraction': [0.7, 0.8, 0.9],
        'bagging_fraction': [0.7, 0.8, 0.9],
        'lambda_l1': [0, 0.1, 1.0],
        'lambda_l2': [0, 0.1, 1.0]
    }
    
    # Base parameters that stay constant
    base_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # For computational efficiency, let's use a subset of parameter combinations
    # We'll use a subset of the parameters that tend to have the most impact
    keys_to_tune = ['learning_rate', 'num_leaves', 'max_depth', 'feature_fraction']
    param_combinations = list(itertools.product(
        param_grid['learning_rate'],
        param_grid['num_leaves'],
        param_grid['max_depth'], 
        param_grid['feature_fraction']
    ))
    
    best_score = 0
    best_params = None
    best_run_id = None
    
    print(f"Starting hyperparameter tuning with {len(param_combinations)} combinations")
    
    for i, (lr, num_leaves, max_depth, feat_frac) in enumerate(param_combinations):
        # Create parameters dictionary for this run
        params = base_params.copy()
        params.update({
            'learning_rate': lr,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'feature_fraction': feat_frac,
            'min_data_in_leaf': 20,  # Default from remaining params
            'bagging_fraction': 0.8,  # Default
            'lambda_l1': 0.1,  # Default
            'lambda_l2': 0.1   # Default
        })
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"run_{i+1}"):
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
                
            # Train and evaluate model
            model, metrics = train_and_evaluate(X_train, X_test, y_train, y_test, params)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log the model
            mlflow.lightgbm.log_model(model, f"model_run_{i+1}")
            
            # Track best model based on F1 score (you can change this to another metric if needed)
            current_score = metrics['f1_score']
            if current_score > best_score:
                best_score = current_score
                best_params = params.copy()
                best_run_id = mlflow.active_run().info.run_id
                
            print(f"Run {i+1}/{len(param_combinations)}: f1_score={metrics['f1_score']:.4f}, "
                  f"accuracy={metrics['accuracy']:.4f}, AUC={metrics['roc_auc']:.4f}")
    
    print("\nHyperparameter tuning completed!")
    print(f"Best F1 Score: {best_score:.4f}")
    print(f"Best Run ID: {best_run_id}")
    print("Best Parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
        
    return best_run_id, best_params

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for LightGBM with MLflow tracking')
    parser.add_argument('--data_path', type=str, default='bank.csv',
                        help='Path to the CSV file (default: bank.csv)')
    args = parser.parse_args()
    
    print(f"Using data from: {args.data_path}")
    best_run_id, best_params = hyperparameter_tuning(args.data_path)
    
    # Optionally, train a final model with the best parameters on all data
    print("\nTraining final model with best parameters...")
    
    with mlflow.start_run(run_name="final_model"):
        # Load and preprocess all data (no test split)
        X_train, X_test, y_train, y_test = load_and_preprocess_data(args.data_path)
        
        # Combine train and test for final model
        X_all = pd.concat([X_train, X_test])
        y_all = pd.concat([y_train, y_test])
        
        # Log best parameters
        for key, value in best_params.items():
            mlflow.log_param(key, value)
            
        # Create dataset
        train_data = lgb.Dataset(X_all, label=y_all)
        
        # Train final model
        final_model = lgb.train(
            best_params,
            train_data,
            num_boost_round=1000
        )
        
        # Log the final model
        mlflow.lightgbm.log_model(final_model, "final_model")
        
        print("Final model trained and logged to MLflow")
        print(f"Final run ID: {mlflow.active_run().info.run_id}")
