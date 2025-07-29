#!/bin/bash

# ML Regression Dashboard - Unix Launcher
echo "======================================================================"
echo "   ML REGRESSION DASHBOARD - UNIX LAUNCHER"
echo "======================================================================"
echo ""
echo "Starting the ML Regression Dashboard..."
echo "This will automatically install all required packages."
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Try different Python commands
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found! Please install Python 3.7+ and try again."
    read -p "Press Enter to exit..."
    exit 1
fi

echo "🐍 Using: $PYTHON_CMD"
echo ""

# Run the Python launcher
$PYTHON_CMD start.py

# Check exit status
if [ $? -ne 0 ]; then
    echo ""
    echo "An error occurred. Press Enter to exit..."
    read
fi 