import requests
import json
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("bank_api_client")

def test_single_prediction():
    """Test the single prediction endpoint"""
    url = "http://localhost:8000/predict"
    
    # Example customer data
    customer_data = {
        "customer_age": 41,
        "tenure": 36,
        "account_balance": 2567.89,
        "previous_campaign_number_of_contacts": 2,
        "employment_type": "private",
        "marital_status": "married",
        "education_level": "tertiary",
        "credit_default": "no",
        "housing_loan": "yes",
        "personal_loan": "no",
        "contact_method": "cellular",
        "previous_campaign_outcome": "success"
    }
    
    # Make the request
    logger.info("Making single prediction request...")
    response = requests.post(url, json=customer_data)
    
    # Log the response
    logger.info("=== Single Prediction Test ===")
    logger.info(f"Status code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Prediction: {result['prediction']}")
        logger.info(f"Probability: {result['probability']:.4f}")
        logger.info("Feature Importance:")
        for feature, importance in sorted(
            result['explanation'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]:
            logger.info(f"  {feature}: {importance:.4f}")
    else:
        logger.error(f"Error: {response.text}")

def test_batch_prediction():
    """Test the batch prediction endpoint"""
    url = "http://localhost:8000/batch-predict"
    
    # Example batch of customer data
    customers_data = [
        {
            "customer_age": 41,
            "tenure": 36,
            "account_balance": 2567.89,
            "previous_campaign_number_of_contacts": 2,
            "employment_type": "private",
            "marital_status": "married",
            "education_level": "tertiary",
            "credit_default": "no",
            "housing_loan": "yes",
            "personal_loan": "no",
            "contact_method": "cellular",
            "previous_campaign_outcome": "success"
        },
        {
            "customer_age": 25,
            "tenure": 12,
            "account_balance": 456.78,
            "previous_campaign_number_of_contacts": 0,
            "employment_type": "student",
            "marital_status": "single",
            "education_level": "secondary",
            "credit_default": "no",
            "housing_loan": "no",
            "personal_loan": "no",
            "contact_method": "cellular",
            "previous_campaign_outcome": "unknown"
        }
    ]
    
    # Make the request
    logger.info("Making batch prediction request...")
    response = requests.post(url, json=customers_data)
    
    # Log the response
    logger.info("=== Batch Prediction Test ===")
    logger.info(f"Status code: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Predictions: {result['predictions']}")
        logger.info(f"Probabilities: {[round(p, 4) for p in result['probabilities']]}")
    else:
        logger.error(f"Error: {response.text}")

def test_model_info():
    """Test the model info endpoint"""
    url = "http://localhost:8000/model-info"
    
    # Make the request
    logger.info("Getting model info...")
    response = requests.get(url)
    
    # Log the response
    logger.info("=== Model Info Test ===")
    logger.info(f"Status code: {response.status_code}")
    if response.status_code == 200:
        info = response.json()
        logger.info(f"Model type: {info['model_type']}")
        if 'features' in info:
            logger.info("Features:")
            if 'numeric' in info['features']:
                logger.info(f"  Numeric: {info['features']['numeric']}")
            if 'categorical' in info['features']:
                logger.info(f"  Categorical: {info['features']['categorical']}")
    else:
        logger.error(f"Error: {response.text}")

def test_health_check():
    """Test the health check endpoint"""
    url = "http://localhost:8000/health"
    
    # Make the request
    logger.info("Checking API health...")
    response = requests.get(url)
    
    # Log the response
    logger.info("=== Health Check Test ===")
    logger.info(f"Status code: {response.status_code}")
    if response.status_code == 200:
        status = response.json()
        logger.info(f"Status: {status['status']}")
        logger.info(f"Model loaded: {status['model_loaded']}")
    else:
        logger.error(f"Error: {response.text}")

if __name__ == "__main__":
    logger.info("Starting Bank ML API tests...")
    try:
        test_health_check()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.exception(f"An error occurred during testing: {str(e)}")