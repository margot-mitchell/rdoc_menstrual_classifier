#!/bin/bash

# Script to run the cross-validation experiments
# Usage: ./scripts/run_cross_validation.sh

set -e  # Exit on any error

echo "Starting cross-validation experiments..."

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

# Run cross-validation
echo "Running cross-validation..."
python src/main/cross_validation.py

echo "Cross-validation completed successfully!"
echo "Check the outputs/reports/ directory for results." 