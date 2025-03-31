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

def load_params():
    """Load parameters from params.yaml file"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

def train():
    """Train a random forest model on preprocessed data and log to MLflow"""
    # Start MLflow run
    with mlflow.start_run(run_name="training") as run:
        # Load parameters
        params = load_params()
        
        # Get file paths and model params from config
        input_file = params["training"]["input"]
        model_output = params["training"]["model_output"]
        test_size = params["training"]["test_size"]
        random_state = params["training"]["random_state"]
        n_estimators = params["training"]["n_estimators"]
        max_depth = params["training"]["max_depth"]
        
        # Log parameters
        mlflow.log_params({
            "test_size": test_size,
            "random_state": random_state,
            "n_estimators": n_estimators,
            "max_depth": max_depth
        })
        
        # Load data
        data = pd.read_csv(input_file)
        
        # Split features and target
        X = data.drop(["Outcome"], axis=1)
        y = data["Outcome"]
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        
        model.fit(X_train, y_train)
        
        # Get predictions and accuracy
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_preds)
        test_accuracy = accuracy_score(y_test, test_preds)
        
        # Log metrics
        mlflow.log_metric("training_accuracy", train_accuracy)
        mlflow.log_metric("testing_accuracy", test_accuracy)
        
        # Save feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in feature_importance.iterrows():
            mlflow.log_metric(f"importance_{row['feature']}", row['importance'])
        
        # Create directory for model if it doesn't exist
        os.makedirs(os.path.dirname(model_output), exist_ok=True)
        
        # Save model
        with open(model_output, "wb") as f:
            pickle.dump(model, f)
        
        # Log model as artifact
        mlflow.log_artifact(model_output)
        
        # Infer model signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Log scikit-learn model with signature
        mlflow.sklearn.log_model(
            model, 
            "sklearn_model",
            signature=signature,
            input_example=X_train.iloc[0:5],
            registered_model_name="diabetes_prediction_model"
        )
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Testing accuracy: {test_accuracy:.4f}")
        print(f"Model saved to: {model_output}")
        print(f"Model registered as: diabetes_prediction_model")
        
        # Get MLflow run URL and print it
        client = mlflow.tracking.MlflowClient()
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        
        print(f"🏃 View run training at: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}/runs/{run_id}")
        print(f"🧪 View experiment at: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}")

if __name__ == "__main__":
    train()
