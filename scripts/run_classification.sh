#!/bin/bash

# Script to run the menstrual cycle classification
# Usage: ./scripts/run_classification.sh

set -e  # Exit on any error

echo "Starting menstrual cycle classification..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install requirements if needed
echo "Checking dependencies..."
pip install -r requirements.txt

# Check if simulation data exists
if [ ! -f "outputs/data/hormone_data_unlabeled.csv" ]; then
    echo "Simulation data not found. Running simulation first..."
    python src/main/simulation.py
fi

# Run classification
echo "Running classification..."
python src/main/classification.py

echo "Classification completed successfully!"
echo "Check the outputs/reports/ directory for results." 