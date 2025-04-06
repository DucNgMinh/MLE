# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY ./app.py /app

COPY ./requirements.txt /app

COPY ./models /app

RUN pip install --no-cache-dir -r requirements.txt



# Expose the port the app runs on
EXPOSE 8000

# Use uvicorn to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]