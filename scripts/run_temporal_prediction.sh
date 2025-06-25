#!/bin/bash

# Temporal prediction script for making rule-based predictions
# This script uses survey responses and period data to predict menstrual phases

echo "Starting temporal prediction pipeline..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found. Please run training first."
    exit 1
fi

# Check if required data files exist
if [ ! -f "outputs/data/hormone_data_unlabeled.csv" ]; then
    echo "Error: Unlabeled hormone data not found at outputs/data/hormone_data_unlabeled.csv"
    echo "Please run simulation first: ./scripts/run_simulation.sh"
    exit 1
fi

if [ ! -f "outputs/data/survey_responses.csv" ]; then
    echo "Error: Survey data not found at outputs/data/survey_responses.csv"
    echo "Please run simulation first: ./scripts/run_simulation.sh"
    exit 1
fi

if [ ! -f "outputs/data/period_sleep_data.csv" ]; then
    echo "Error: Period data not found at outputs/data/period_sleep_data.csv"
    echo "Please run simulation first: ./scripts/run_simulation.sh"
    exit 1
fi

# Run temporal prediction
echo "Running temporal prediction..."
python src/main/temporal_predict.py

echo "Temporal prediction pipeline completed!"
echo "Check outputs/predictions/ for rule-based predictions"
echo "Check outputs/reports/ for prediction summaries" 