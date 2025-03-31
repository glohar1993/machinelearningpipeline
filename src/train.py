"""
Training Script for Diabetes Prediction Pipeline
==============================================

This script is the SECOND step in our ML pipeline. It handles model training:
1. Loads preprocessed data from data/preprocessed/pima_diabetes_clean.csv
2. Splits data into training and testing sets
3. Trains a Random Forest model with specified parameters
4. Saves the trained model to models/random_forest_model.pkl
5. Logs training metrics and model to MLflow

Flow in Pipeline:
----------------
1. This script runs SECOND in the pipeline
2. Uses preprocessed data from preprocess.py (FIRST step)
3. Output is used by evaluate.py (THIRD step)
4. Creates trained model for evaluation

Dependencies:
------------
- sklearn: For Random Forest model and metrics
- pandas: For data handling
- mlflow: For experiment tracking
- yaml: For configuration management
- pickle: For model serialization
"""

# train.py

import pandas as pd
import numpy as np
import pickle
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.models.signature import infer_signature
# Import MLflow configuration
import mlflow_config
import json

def load_params():
    """Load parameters from params.yaml file"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

def train():
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri("https://dagshub.com/ganeshml15/my-first-repo.mlflow")
    mlflow.set_experiment("diabetes_prediction")
    
    # Start MLflow run
    with mlflow.start_run(run_name="training"):
        # Load preprocessed data
        data = pd.read_csv('data/preprocessed/pima_diabetes_clean.csv')
        
        # Split features and target
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        # Save metrics
        metrics = {
            "training_accuracy": float(train_accuracy),
            "testing_accuracy": float(test_accuracy),
            "n_estimators": model.n_estimators,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        # Save metrics to file
        os.makedirs('metrics', exist_ok=True)
        with open('metrics/train_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Log metrics and parameters to MLflow
        mlflow.log_metrics(metrics)
        mlflow.log_params({
            "n_estimators": model.n_estimators,
            "random_state": 42
        })
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = 'models/random_forest_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "sklearn_model")
        mlflow.log_artifact(model_path)
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Testing accuracy: {test_accuracy:.4f}")
        print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    train()
