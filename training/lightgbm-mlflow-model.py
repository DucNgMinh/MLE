import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import argparse

# Set the MLflow tracking URI (optional - use if you have a specific server)
mlflow.set_tracking_uri("http://localhost:5001")

# Create an experiment name with timestamp
experiment_name = f"bank_marketing_lightgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
mlflow.set_experiment(experiment_name)

# Data loading and preprocessing
def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path, delimiter=';')
    
    print(f"Dataset loaded with shape: {df.shape}")
    print(f"Target distribution:\n{df['y'].value_counts()}")
    
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, X.columns

# Create feature importance plot
def plot_feature_importance(model, feature_names, output_path):
    plt.figure(figsize=(10, 8))
    
    # Get feature importance from the model
    importance = model.feature_importance(importance_type='gain')
    
    # Create a DataFrame for easier manipulation
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    # Plot top 20 features
    top_features = feature_imp.head(20)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    return feature_imp

# Main function for model training and evaluation
def train_and_evaluate_model(data_path):
    with mlflow.start_run() as run:
        # Log data path
        mlflow.log_param("data_path", data_path)
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(data_path)
        
        # Define LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1
        }
        
        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        # Create dataset for LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model
        print("Training LightGBM model...")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, test_data],
            callbacks=[lgb.early_stopping(50, verbose=True)]
        )
        
        # Make predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Create and save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        cm_path = "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        
        # Create and save feature importance plot
        feature_imp_path = "feature_importance.png"
        feature_imp_df = plot_feature_importance(model, feature_names, feature_imp_path)
        mlflow.log_artifact(feature_imp_path)
        
        # Save feature importance as CSV
        feature_imp_csv = "feature_importance.csv"
        feature_imp_df.to_csv(feature_imp_csv, index=False)
        mlflow.log_artifact(feature_imp_csv)
        
        # Log the model
        mlflow.lightgbm.log_model(model, "lightgbm_model")
        
        # Clean up temporary files after logging to MLflow
        try:
            os.remove(cm_path)
            print(f"Deleted temporary file: {cm_path}")
            os.remove(feature_imp_path)
            print(f"Deleted temporary file: {feature_imp_path}")
            os.remove(feature_imp_csv)
            print(f"Deleted temporary file: {feature_imp_csv}")
        except Exception as e:
            print(f"Warning: Could not delete temporary files: {str(e)}")
       
        # Print results
        print("\nModel Evaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        print(f"\nModel and artifacts logged to MLflow with run ID: {run.info.run_id}")
        print(f"Experiment name: {experiment_name}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train LightGBM model with MLflow tracking')
    parser.add_argument('--data_path', type=str, default='bank.csv',
                        help='Path to the CSV file (default: bank.csv)')
    args = parser.parse_args()
    
    print(f"Using data from: {args.data_path}")
    train_and_evaluate_model(args.data_path)
