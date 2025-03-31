#!/bin/bash

# Define variables
IMAGE_NAME="bank-model-api"
CONTAINER_NAME="bank-model-container"
PORT=8000
MLFLOW_RUN_ID=$1

# Check if MLflow run ID is provided
if [ -z "$MLFLOW_RUN_ID" ]; then
    echo "Warning: No MLflow run ID provided. The API will try to use the latest run."
    echo "Usage: ./deploy.sh <mlflow_run_id>"
fi

# Stop and remove existing container if it exists
echo "Stopping and removing existing container if it exists..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Run the container
echo "Starting container..."
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8000 \
    -e MLFLOW_RUN_ID=$MLFLOW_RUN_ID \
    -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
    -v $(pwd)/bank.csv:/app/bank.csv \
    $IMAGE_NAME

echo "Container is running!"
echo "API is available at http://localhost:$PORT"
echo "API documentation is available at http://localhost:$PORT/docs"
