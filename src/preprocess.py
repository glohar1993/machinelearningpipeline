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
import json
# Import MLflow configuration
import mlflow_config

def load_params():
    """Load parameters from params.yaml file"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

def preprocess_data():
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri("https://dagshub.com/ganeshml15/my-first-repo.mlflow")
    mlflow.set_experiment("diabetes_prediction")
    
    # Start MLflow run
    with mlflow.start_run(run_name="preprocessing"):
        # Load data
        data = pd.read_csv('data/raw/pima_diabetes.csv')
        
        # Get initial shape
        initial_shape = data.shape
        
        # Handle missing values (0 values in certain columns)
        columns_to_process = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for column in columns_to_process:
            data[column] = data[column].replace(0, np.nan)
            data[column] = data[column].fillna(data[column].mean())
        
        # Scale features
        scaler = StandardScaler()
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        data[features] = scaler.fit_transform(data[features])
        
        # Save preprocessed data
        os.makedirs('data/preprocessed', exist_ok=True)
        output_file = 'data/preprocessed/pima_diabetes_clean.csv'
        data.to_csv(output_file, index=False)
        
        # Log metrics
        metrics = {
            "initial_rows": initial_shape[0],
            "initial_columns": initial_shape[1],
            "final_rows": data.shape[0],
            "final_columns": data.shape[1],
            "missing_values_filled": sum([len(data[data[col] == 0]) for col in columns_to_process])
        }
        
        # Save metrics to file
        os.makedirs('metrics', exist_ok=True)
        with open('metrics/preprocess_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(output_file)
        
        print(f"Preprocessed data shape: {data.shape}")
        print(f"Saved preprocessed data to: {output_file}")

if __name__ == "__main__":
    preprocess_data()
