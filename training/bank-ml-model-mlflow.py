import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

MLFLOW_PATH = "http://localhost:5001/"

# Set up MLflow experiment
def setup_mlflow_experiment(experiment_name="Certificate of Deposit Campaign"):
    """
    Set up MLflow experiment and return experiment ID
    
    Parameters:
    -----------
    experiment_name : str, optional
        Name of the MLflow experiment
    
    Returns:
    --------
    str
        Experiment ID
    """
    # Set tracking URI to local mlruns directory
    mlflow.set_tracking_uri(MLFLOW_PATH)
    
    # Create or get existing experiment
    mlflow.set_experiment(experiment_name)
    
    return mlflow.get_experiment_by_name(experiment_name).experiment_id

# Load data
def load_data(filepath):
    """
    Load data from Excel file
    
    Parameters:
    -----------
    filepath : str
        Path to the Excel file
    
    Returns:
    --------
    X : DataFrame
        Features
    y : Series
        Target variable
    """
    # Read the Excel file
    df = pd.read_excel(filepath)
    
    # Remove unique identifier columns
    df = df.drop(['CIF', 'date'], axis=1)
    
    # Separate features and target
    X = df.drop('y', axis=1)
    y = df['y']
    
    return X, y

# Prepare preprocessing
def create_preprocessor():
    """
    Create a preprocessing pipeline for both numeric and categorical features
    
    Returns:
    --------
    ColumnTransformer
        Preprocessor for features
    """
    # Identify numeric and categorical columns
    numeric_features = [
        'customer_age', 
        'tenure', 
        'account_balance', 
        'previous_campaign_number_of_contacts'
    ]
    
    categorical_features = [
        'employment_type', 
        'marital_status', 
        'education_level', 
        'credit_default',
        'housing_loan', 
        'personal_loan', 
        'contact_method', 
        'previous_campaign_outcome'
    ]
    
    # Preprocessing for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# Train the model with MLflow tracking
def train_and_log_model(X, y, experiment_id, run_name='Random Forest Classifier'):
    """
    Train a Random Forest Classifier with MLflow tracking
    
    Parameters:
    -----------
    X : DataFrame
        Features
    y : Series
        Target variable
    experiment_id : str
        MLflow experiment ID
    run_name : str, optional
        Name of the MLflow run
    
    Returns:
    --------
    tuple
        Trained model, test features, test target, predictions
    """
    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create preprocessor
        preprocessor = create_preprocessor()
        
        # Hyperparameters to log
        rf_params = {
            'n_estimators': 100,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        
        # Log parameters
        mlflow.log_params(rf_params)
        
        # Create model pipeline
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(**rf_params))
        ])
        
        # Fit the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        mlflow.log_metric('cv_mean_score', cv_scores.mean())
        mlflow.log_metric('cv_score_std', cv_scores.std())
        
        # Generate and save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        
        # Log confusion matrix as artifact
        mlflow.log_artifact('confusion_matrix.png')
        
        # Generate feature importance plot
        preprocessor = model.named_steps['preprocessor']
        onehot = preprocessor.named_transformers_['cat'].named_steps['onehot']
        
        # Get feature names
        cat_feature_names = list(onehot.get_feature_names_out(
            preprocessor.transformers_[1][2]
        ))
        numeric_features = preprocessor.transformers_[0][2]
        feature_names = list(numeric_features) + cat_feature_names
        
        # Get feature importances
        clf = model.named_steps['classifier']
        importances = clf.feature_importances_
        
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Create feature importance plot
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances in Bank Marketing Prediction")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                   [feature_names[i] for i in indices], 
                   rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importances.png')
        
        # Log feature importance plot
        mlflow.log_artifact('feature_importances.png')
        
        # Log classification report as artifact
        with open('classification_report.txt', 'w') as f:
            f.write(classification_report(y_test, y_pred))
        mlflow.log_artifact('classification_report.txt')
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        return model, X_test, y_test, y_pred

# Main execution
def main():
    # Set up MLflow experiment
    experiment_id = setup_mlflow_experiment()
    
    # File path (modify as needed)
    filepath = r"C:\Users\admin\Documents\Projects\MLE\data\bank.xlsx"
    
    # Load data
    X, y = load_data(filepath)
    
    # Train and log model
    model, X_test, y_test, y_pred = train_and_log_model(X, y, experiment_id)
    
    return model

# Run the main function
if __name__ == '__main__':
    trained_model = main()
    print("Model training completed. Check MLflow UI for details.")
