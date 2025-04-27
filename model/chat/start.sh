#!/bin/bash

# Step 1: Check and select Python version
if command -v python3 &>/dev/null; then
  PYTHON=python3
elif command -v python &>/dev/null; then
  PYTHON=python
else
  echo "Python is not installed on your system."
  echo "Redirecting you to the download page..."
  open "https://www.python.org/downloads/"
  exit 1
fi

echo "Using $PYTHON"

# Step 2: Set up virtual environment
$PYTHON -m venv venv
source venv/bin/activate

# Step 3: Install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Step 4: Start Flask app in background
echo "Starting Flask app..."
$PYTHON app.py &
echo $! > flask.pid  # Save PID to file
sleep 3
open http://127.0.0.1:5000/
