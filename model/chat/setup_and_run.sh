#!/bin/bash

# chmod +x setup_and_run.sh
# ./setup_and_run.sh



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
FLASK_PID=$!
sleep 3
open http://127.0.0.1:5000/

# Step 5: Wait for user to hit ENTER to clean up
echo "Press ENTER to stop the server and clean up..."
read

# Step 6: Kill Flask process
echo "Stopping Flask app (PID: $FLASK_PID)..."
kill "$FLASK_PID"

# Step 7: Deactivate virtual environment
deactivate

# Step 8: Clean up __pycache__, static folder and chat history
echo "Cleaning up..."
rm -rf __pycache__ static/*
> chat_history.json

# Step 9: Remove virtual environment
rm -rf venv

echo "Done! Environment fully cleaned up."