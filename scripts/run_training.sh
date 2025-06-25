#!/bin/bash

# Training script for menstrual pattern classification models
# This script trains models on labeled data and saves them for later use

echo "Starting model training pipeline..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run training
echo "Running model training..."
python src/main/train_model.py

echo "Training pipeline completed!"
echo "Check outputs/models/ for saved models"
echo "Check outputs/reports/ for training results" 