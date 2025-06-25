#!/bin/bash

# Script to run the menstrual cycle simulation
# Usage: ./scripts/run_simulation.sh

set -e  # Exit on any error

echo "Starting menstrual cycle simulation..."

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

# Run simulation
echo "Running simulation..."
python src/main/simulation.py

echo "Simulation completed successfully!"
echo "Check the outputs/ directory for results." 