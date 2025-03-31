"""
Preprocessing Script for Diabetes Prediction Pipeline
===================================================

This script is the FIRST step in our ML pipeline. It handles data preprocessing tasks:
1. Loads raw data from data/raw/pima_diabetes.csv
2. Handles missing values and outliers
3. Scales features for better model performance
4. Saves processed data to data/preprocessed/pima_diabetes_clean.csv
5. Logs preprocessing steps to MLflow for experiment tracking

Flow in Pipeline:
----------------
1. This script runs FIRST in the pipeline
2. Output is used by train.py (SECOND step)
3. Creates clean, processed dataset for model training

Dependencies:
------------
- pandas: For data manipulation
- sklearn: For data scaling
- mlflow: For experiment tracking
- yaml: For configuration management
"""

# preprocess.py

import pandas as pd
import numpy as np
import yaml
import os
from sklearn.preprocessing import StandardScaler
import mlflow
# Import MLflow configuration
import mlflow_config

def load_params():
    """Load parameters from params.yaml file"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

def preprocess_data():
    """Preprocess the diabetes dataset and save results"""
    # Start MLflow run
    with mlflow.start_run(run_name="preprocessing") as run:
        # Load parameters
        params = load_params()
        
        # Get file paths from params
        input_file = params["preprocessing"]["input"]
        output_file = params["preprocessing"]["output"]
        
        # Load data
        data = pd.read_csv(input_file)
        
        # Log data shape as a metric
        mlflow.log_param("input_shape", str(data.shape))
        
        # Fill missing values with mean
        data.fillna(data.mean(), inplace=True)
        
        # Split features and target
        X = data.drop(["Outcome"], axis=1)
        y = data["Outcome"]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Combine scaled features with target
        data_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        data_scaled["Outcome"] = y.values
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save preprocessed data
        data_scaled.to_csv(output_file, index=False)
        
        # Log metrics
        mlflow.log_param("output_shape", str(data_scaled.shape))
        mlflow.log_artifact(output_file)
        
        # Print information
        print(f"Preprocessed data shape: {data_scaled.shape}")
        print(f"Saved preprocessed data to: {output_file}")
        
        # Get MLflow run URL and print it
        client = mlflow.tracking.MlflowClient()
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        
        print(f"🏃 View run preprocessing at: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}/runs/{run_id}")
        print(f"🧪 View experiment at: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}")

if __name__ == "__main__":
    preprocess_data()
