"""
Evaluation Script for Diabetes Prediction Pipeline
===============================================

This script is the THIRD and FINAL step in our ML pipeline. It handles model evaluation:
1. Loads trained model from models/random_forest_model.pkl
2. Makes predictions on test data
3. Calculates performance metrics (accuracy, precision, recall, F1-score)
4. Generates visualizations (confusion matrix, feature importance)
5. Logs evaluation results to MLflow

Flow in Pipeline:
----------------
1. This script runs THIRD in the pipeline
2. Uses trained model from train.py (SECOND step)
3. This is the FINAL step in the pipeline
4. Provides comprehensive model evaluation

Dependencies:
------------
- sklearn: For metrics and model loading
- pandas: For data handling
- matplotlib: For visualization
- mlflow: For experiment tracking
- yaml: For configuration management
- pickle: For model loading
"""

# evaluate.py

import pandas as pd
import numpy as np
import yaml
import os
import json
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import mlflow
from mlflow.tracking import MlflowClient

# Import MLflow configuration
import mlflow_config

def load_params():
    """Load parameters from params.yaml file"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params

def evaluate():
    """Evaluate the trained model and log results to MLflow"""
    # Start MLflow run
    with mlflow.start_run(run_name="model_evaluation") as run:
        # Load parameters
        params = load_params()
        
        # Load model
        model_path = params["training"]["model_output"]
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Load data
        data = pd.read_csv(params["training"]["input"])
        
        # Split features and target
        X = data.drop(["Outcome"], axis=1)
        y = data["Outcome"]
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        clf_report = classification_report(y, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Log metrics to MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_class_0", clf_report["0"]["precision"])
        mlflow.log_metric("recall_class_0", clf_report["0"]["recall"])
        mlflow.log_metric("precision_class_1", clf_report["1"]["precision"])
        mlflow.log_metric("recall_class_1", clf_report["1"]["recall"])
        
        # Create and save confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=["No Diabetes", "Diabetes"],
                   yticklabels=["No Diabetes", "Diabetes"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        
        # Create directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        
        # Save confusion matrix
        plt.savefig("reports/confusion_matrix.png")
        mlflow.log_artifact("reports/confusion_matrix.png")
        plt.close()
        
        # Create and save feature importance plot
        feature_importance = model.feature_importances_
        features = X.columns
        
        # Sort feature importances in descending order
        indices = np.argsort(feature_importance)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(X.shape[1]), feature_importance[indices], align="center")
        plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        
        # Save feature importance plot
        plt.savefig("reports/feature_importance.png")
        mlflow.log_artifact("reports/feature_importance.png")
        plt.close()
        
        # Save metrics as JSON for DVC
        metrics_dict = {
            "accuracy": float(accuracy),
            "precision_class_0": float(clf_report["0"]["precision"]),
            "recall_class_0": float(clf_report["0"]["recall"]),
            "precision_class_1": float(clf_report["1"]["precision"]),
            "recall_class_1": float(clf_report["1"]["recall"]),
            "f1_class_0": float(clf_report["0"]["f1-score"]),
            "f1_class_1": float(clf_report["1"]["f1-score"])
        }
        
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        # Print feature importance
        print("\nFeature Importance:")
        for i, feature in enumerate(features[indices]):
            print(f"{feature}: {feature_importance[indices[i]]:.6f}")
        
        # Get MLflow run URL and print it
        client = MlflowClient()
        run_id = run.info.run_id
        experiment_id = run.info.experiment_id
        
        print(f"\n🏃 View run at: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}/runs/{run_id}")
        print(f"🧪 View experiment at: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}")

if __name__ == "__main__":
    evaluate()
