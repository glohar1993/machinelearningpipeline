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
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri("https://dagshub.com/ganeshml15/my-first-repo.mlflow")
    mlflow.set_experiment("diabetes_prediction")
    
    # Start MLflow run
    with mlflow.start_run(run_name="model_evaluation"):
        # Load the model
        with open('models/random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load test data
        data = pd.read_csv('data/preprocessed/pima_diabetes_clean.csv')
        X = data.drop('Outcome', axis=1)
        y = data['Outcome']
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Generate classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Save metrics
        metrics = {
            "accuracy": report['accuracy'],
            "precision_0": report['0']['precision'],
            "recall_0": report['0']['recall'],
            "f1_score_0": report['0']['f1-score'],
            "precision_1": report['1']['precision'],
            "recall_1": report['1']['recall'],
            "f1_score_1": report['1']['f1-score']
        }
        
        # Save metrics to file
        os.makedirs('metrics', exist_ok=True)
        with open('metrics/evaluate_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        
        # Create and save confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Create and save feature importance plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.savefig('feature_importance.png')
        plt.close()
        
        # Log artifacts
        mlflow.log_artifact('confusion_matrix.png')
        mlflow.log_artifact('feature_importance.png')
        
        # Save feature importance to CSV
        feature_importance.to_csv('feature_importance.csv', index=False)
        
        # Print results
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        print("\nFeature Importance:")
        for _, row in feature_importance.iterrows():
            print(f"{row['feature']}: {row['importance']:.6f}")

if __name__ == "__main__":
    evaluate()
