import requests
import json
import pandas as pd
from typing import List, Dict
import sys

def predict_single(api_url: str, customer_data: Dict) -> Dict:
    """
    Make a prediction for a single customer
    
    Args:
        api_url: Base URL of the API
        customer_data: Dictionary with customer data
        
    Returns:
        Prediction response
    """
    response = requests.post(
        f"{api_url}/predict",
        json=customer_data
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def predict_batch(api_url: str, customers_data: List[Dict]) -> Dict:
    """
    Make predictions for multiple customers
    
    Args:
        api_url: Base URL of the API
        customers_data: List of dictionaries with customer data
        
    Returns:
        Batch prediction response
    """
    response = requests.post(
        f"{api_url}/predict/batch",
        json={"customers": customers_data}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def load_sample_data(csv_path: str, num_samples: int = 5) -> List[Dict]:
    """
    Load sample data from CSV
    
    Args:
        csv_path: Path to the CSV file
        num_samples: Number of samples to load
        
    Returns:
        List of dictionaries with customer data
    """
    try:
        df = pd.read_csv(csv_path, delimiter=';')
        # Drop target column if present
        if 'y' in df.columns:
            df = df.drop('y', axis=1)
        
        # Convert to list of dictionaries
        samples = df.head(num_samples).to_dict(orient='records')
        return samples
    except Exception as e:
        print(f"Error loading sample data: {str(e)}")
        return None

def main():
    # Set API URL
    api_url = "http://localhost:8000"
    
    # Check API health
    try:
        health_response = requests.get(f"{api_url}/health")
        if health_response.status_code == 200:
            print("API is healthy!")
        else:
            print(f"API health check failed: {health_response.status_code}")
            print(health_response.text)
            return
    except Exception as e:
        print(f"Could not connect to API: {str(e)}")
        return
    
    # Example 1: Single prediction with hardcoded data
    print("\n--- Example 1: Single Prediction ---")
    
    sample_customer = {
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
    
    result = predict_single(api_url, sample_customer)
    if result:
        print(f"Prediction: {result['prediction_label']}")
        print(f"Probability: {result['probability']:.4f}")
    
    # Example 2: Load sample data from CSV and make batch prediction
    print("\n--- Example 2: Batch Prediction ---")
    
    # Check if CSV path is provided as command line argument
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "bank.csv"  # Default path
    
    samples = load_sample_data(csv_path, num_samples=10)
    if samples:
        print(f"Loaded {len(samples)} samples from {csv_path}")
        
        batch_result = predict_batch(api_url, samples)
        if batch_result:
            print("\nBatch Prediction Summary:")
            summary = batch_result["summary"]
            print(f"Total predictions: {summary['total_predictions']}")
            print(f"Positive predictions: {summary['positive_predictions']} ({summary['positive_percentage']:.2f}%)")
            print(f"Average probability: {summary['average_probability']:.4f}")
            
            print("\nIndividual Predictions:")
            for i, pred in enumerate(batch_result["predictions"][:5]):  # Show first 5
                print(f"Customer {i+1}: {pred['prediction_label']} (Probability: {pred['probability']:.4f})")
            
            if len(batch_result["predictions"]) > 5:
                print(f"... and {len(batch_result['predictions']) - 5} more")

if __name__ == "__main__":
    main()
