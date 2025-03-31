# Diabetes Prediction ML Pipeline

This repository contains a complete machine learning pipeline for diabetes prediction using the Pima Indians Diabetes dataset.

## Project Structure
```
Machine-Learning-Pipeline/
├── data/
│   ├── raw/                    # Original dataset
│   └── preprocessed/           # Cleaned data
├── models/                     # Trained models
├── src/                        # Source code
│   ├── preprocess.py          # Data cleaning
│   ├── train.py              # Model training
│   └── evaluate.py           # Model evaluation
├── mlruns/                    # MLflow tracking
├── params.yaml                # Configuration
└── requirements.txt           # Dependencies
```

## Setup Instructions

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Pipeline

1. Preprocess data:
```bash
python src/preprocess.py
```

2. Train model:
```bash
python src/train.py
```

3. Evaluate model:
```bash
python src/evaluate.py
```

4. View results:
```bash
mlflow ui --port 5001
```

## Model Performance
- Training accuracy: 86.32%
- Testing accuracy: 76.62%
- Overall accuracy: 84%

## Feature Importance
1. Glucose (34.27%)
2. BMI (18.15%)
3. Age (16.04%)

## MLflow Integration
- Tracks experiments
- Logs parameters
- Stores metrics
- Saves artifacts

## Dependencies
- scikit-learn
- pandas
- numpy
- matplotlib
- mlflow
- pyyaml
