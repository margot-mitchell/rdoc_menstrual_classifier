#!/bin/bash

# Prediction script for making predictions on unlabeled hormone data
# This script loads trained models and predicts menstrual phases

echo "Starting prediction pipeline..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found. Please run training first."
    exit 1
fi

# Check if models exist
if [ ! -d "outputs/models" ] || [ -z "$(ls -A outputs/models 2>/dev/null)" ]; then
    echo "Error: No trained models found in outputs/models/"
    echo "Please run training first: ./scripts/run_training.sh"
    exit 1
fi

# Run prediction
echo "Running predictions..."
python src/main/predict_model.py

echo "Prediction pipeline completed!"
echo "Check outputs/predictions/ for prediction results"
echo "Check outputs/reports/ for prediction summaries" 