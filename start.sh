#!/bin/bash

# Cybersecurity Agent Startup Script
# This script sets up the environment and starts the CLI

echo "🚀 Starting Cybersecurity Agent..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python version $PYTHON_VERSION is too old. Required: $REQUIRED_VERSION or higher"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Check if requirements are installed
echo "🔍 Checking dependencies..."
if ! python3 -c "import pandas, numpy, cryptography" &> /dev/null; then
    echo "⚠️  Some dependencies are missing. Installing..."
    pip3 install -r requirements.txt
fi

# Add bin directory to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}/bin"

echo "✅ Environment configured"
echo "🚀 Starting CLI..."

# Start the CLI
python3 cs_util_lg.py "$@"
