"""
MLflow Configuration with DAGsHub Integration
===========================================

This file configures MLflow to work with DAGsHub for experiment tracking.
"""

import os
import mlflow
from mlflow.tracking import MlflowClient

# DAGsHub configuration
DAGSHUB_USERNAME = "ganeshml15"
REPO_NAME = "my-first-repo"
DAGSHUB_TOKEN = "28903ab8086ebbc7a60408636e19f8130615b419"  # Replace with your actual token if needed

# Set MLflow tracking URI to DAGsHub
os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{DAGSHUB_USERNAME}/{REPO_NAME}.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# Set experiment name
EXPERIMENT_NAME = "diabetes_prediction"
mlflow.set_experiment(EXPERIMENT_NAME)

# Print config information for debugging
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"MLflow experiment: {mlflow.get_experiment_by_name(EXPERIMENT_NAME).name}")

# Initialize MLflow client
client = MlflowClient()

# For local testing (comment out when using DAGsHub)
# os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5005" 