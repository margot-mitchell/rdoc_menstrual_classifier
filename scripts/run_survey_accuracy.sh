#!/bin/bash

# Script to run survey accuracy analysis
# This script checks the accuracy of survey responses against actual period data

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Survey Accuracy Analysis ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Change to project root directory
cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: Virtual environment not found. Using system Python."
fi

# Check if required data files exist
DATA_DIR="outputs/data"
SURVEY_FILE="$DATA_DIR/survey_responses.csv"
PERIOD_FILE="$DATA_DIR/period_sleep_data.csv"

if [ ! -f "$SURVEY_FILE" ]; then
    echo "Error: Survey file not found at $SURVEY_FILE"
    echo "Please run simulation first: ./scripts/run_simulation.sh"
    exit 1
fi

if [ ! -f "$PERIOD_FILE" ]; then
    echo "Error: Period file not found at $PERIOD_FILE"
    echo "Please run simulation first: ./scripts/run_simulation.sh"
    exit 1
fi

echo "Found required data files:"
echo "  - Survey data: $SURVEY_FILE"
echo "  - Period data: $PERIOD_FILE"
echo ""

# Run the survey accuracy analysis
echo "Running survey accuracy analysis..."
python src/main/survey_accuracy.py

echo ""
echo "=== Survey Accuracy Analysis Complete ==="
echo "Results saved in outputs/reports/"
echo ""
echo "Files generated:"
echo "  - survey_accuracy_analysis.csv: Detailed analysis results"
echo ""
echo "Check the console output above for summary statistics." 